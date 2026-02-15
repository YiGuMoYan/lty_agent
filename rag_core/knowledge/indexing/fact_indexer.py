import os
import re
import uuid
import json
import asyncio
from typing import List, Dict, Any, Optional, Callable
import config
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models
import jieba
from rank_bm25 import BM25Okapi
from rag_core.utils.logger import logger

class FactIndexer:
    def __init__(self, persist_directory=None):
        """
        Initialize FactIndexer with Qdrant (Local Mode) and BM25.
        """
        if persist_directory is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            persist_directory = os.path.join(base_dir, "dataset", "vector_store", "qdrant_lty")

        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        logger.info(f"[FactIndexer] Initializing Qdrant at {persist_directory}")
        # Initialize Qdrant in local mode (path-based)
        self.client = QdrantClient(path=persist_directory)
        self.collection_name = "lty_facts"

        # Determine vector dimension from config
        from rag_core.llm.embeddings import get_embedding_function
        self.embedding_fn = get_embedding_function()
        self.vector_dim = config.EMBEDDING_DIM

        # Create collection if not exists
        if not self.client.collection_exists(self.collection_name):
            logger.info(f"[FactIndexer] Creating collection {self.collection_name} with dim={self.vector_dim}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE
                )
            )

        # Initialize BM25
        self.bm25 = None
        self.doc_map = [] # List of {'id': id, 'content': text, 'metadata': meta}
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Build BM25 index from Qdrant data"""
        if not self.client.collection_exists(self.collection_name):
            return

        logger.info("[FactIndexer] Loading documents for BM25...")
        try:
            # Scroll all points
            points = []
            offset = None
            while True:
                res = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=None,
                    limit=200,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset
                )
                batch, offset = res
                points.extend(batch)
                if offset is None:
                    break

            if not points:
                return

            self.doc_map = []
            corpus = []

            for p in points:
                text = p.payload.get('text', '')
                self.doc_map.append({
                    'id': p.id,
                    'content': text,
                    'metadata': p.payload.get('full_metadata', {})
                })
                corpus.append(text)

            tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"[FactIndexer] BM25 index built with {len(corpus)} documents.")

        except Exception as e:
            logger.error(f"[FactIndexer] Failed to build BM25 index: {e}")

    def count(self):
        """Return number of entities in collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except:
            return 0

    def _parse_markdown(self, file_path):
        """
        Parse markdown file into sections based on headers.
        Returns list of dicts: {content, metadata}
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract Frontmatter
        frontmatter = {}
        fm_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if fm_match:
            fm_text = fm_match.group(1)
            for line in fm_text.split('\n'):
                if ':' in line:
                    k, v = line.split(':', 1)
                    frontmatter[k.strip()] = v.strip()
            content = content[fm_match.end():]

        # Split by Headers (##)
        parts = re.split(r'(^|\n)##\s+', content)

        sections = []
        if parts[0].strip():
             sections.append({
                "content": parts[0].strip(),
                "section": "Introduction",
                **frontmatter
            })

        lines = content.split('\n')
        buffer = []
        current_section = "Introduction"
        chunk_list = []

        for line in lines:
            if line.strip().startswith('## '):
                if buffer:
                    chunk_list.append((current_section, "\n".join(buffer)))
                buffer = []
                current_section = line.strip().replace('#', '').strip()
                buffer.append(line)
            else:
                buffer.append(line)

        if buffer:
             chunk_list.append((current_section, "\n".join(buffer)))

        results = []
        base_name = os.path.basename(file_path)
        for section_title, text in chunk_list:
            if len(text.strip()) < 10:
                continue

            results.append({
                "id": f"{base_name}#{section_title}",
                "document": text,
                "metadata": {
                    "source": base_name,
                    "section": section_title,
                    "category": frontmatter.get("category", "Unknown"),
                    "topic": frontmatter.get("topic", base_name.replace('.md',''))
                }
            })

        return results

    def _split_text_with_overlap(self, text, chunk_size=800, overlap=200):
        """
        Splits long text into overlapping chunks.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks

    def index_knowledge_base(self, kb_root=None, lyrics_path=None, progress_callback: Callable[[int, int], None] = None):
        """Walk KB directory and index all md files and lyrics.

        Args:
            kb_root: Path to knowledge base directory.
            lyrics_path: Path to lyrics JSONL file.
            progress_callback: 进度回调函数，签名为 callback(current, total)
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if kb_root is None:
            kb_root = os.path.join(base_dir, "dataset", "knowledge_base")

        if lyrics_path is None:
            lyrics_path = os.path.join(base_dir, "dataset", "song", "cleaned_lyrics.jsonl")

        logger.info(f"[FactIndexer] Scanning Knowledge Base: {kb_root}")

        texts_to_embed = []
        temp_metas = [] # List of (uuid, payload)

        # 1. Scan Markdown Files
        md_files = []
        for root, dirs, files in os.walk(kb_root):
            for file in files:
                if file.endswith(".md"):
                    md_files.append(os.path.join(root, file))

        logger.info(f"[FactIndexer] Found {len(md_files)} Markdown files.")

        # 计算总进度
        total_steps = 3  # 1.解析文档 2.生成Embedding 3.插入数据库
        current_step = 0

        for path in tqdm(md_files, desc="Parsing Markdown"):
            mtime = os.path.getmtime(path)
            chunks = self._parse_markdown(path)
            for chunk in chunks:
                # Apply Sliding Window Chunking
                sub_chunks = self._split_text_with_overlap(chunk['document'])

                for i, sub_text in enumerate(sub_chunks):
                    unique_id = str(uuid.uuid4())
                    meta = chunk['metadata'].copy()
                    meta['indexed_at'] = mtime
                    meta['chunk_index'] = i
                    meta['total_chunks'] = len(sub_chunks)

                    payload = {
                        "text": sub_text,
                        "source": meta.get("source", ""),
                        "category": meta.get("category", ""),
                        "topic": meta.get("topic", ""),
                        "full_metadata": meta
                    }

                    texts_to_embed.append(sub_text)
                    temp_metas.append((unique_id, payload))

        # 报告解析完成进度 (20%)
        if progress_callback:
            progress_callback(1, total_steps)

        # 2. Scan Lyrics
        if os.path.exists(lyrics_path):
            logger.info(f"[FactIndexer] Loading Lyrics from: {lyrics_path}")
            try:
                with open(lyrics_path, 'r', encoding='utf-8') as f:
                    lyrics_data = [json.loads(line) for line in f if line.strip()]

                logger.info(f"[FactIndexer] Found {len(lyrics_data)} songs.")

                for song in tqdm(lyrics_data, desc="Parsing Lyrics"):
                    title = song.get("song_title", "Unknown")
                    content = song.get("cleaned_lyrics", "")
                    if not content:
                        continue

                    # Prepare metadata
                    p_masters = song.get("p_masters", [])
                    if isinstance(p_masters, list):
                        p_masters_str = ", ".join(p_masters)
                    else:
                        p_masters_str = str(p_masters)

                    song_meta = song.get("song_metadata", {})
                    rag_text = f"歌曲：{title}\nP主/作者：{p_masters_str}\n\n{content}"

                    # Chunk lyrics too (just in case they are super long)
                    sub_chunks = self._split_text_with_overlap(rag_text)

                    for i, sub_text in enumerate(sub_chunks):
                        unique_id = str(uuid.uuid4())
                        payload = {
                            "text": sub_text,
                            "source": "LyricsDB",
                            "category": "Song",
                            "topic": title,
                            "full_metadata": {
                                "title": title,
                                "p_masters": p_masters,
                                "type": "lyrics",
                                "chunk_index": i
                            }
                        }
                        texts_to_embed.append(sub_text)
                        temp_metas.append((unique_id, payload))

            except Exception as e:
                logger.error(f"[FactIndexer] Error loading lyrics: {e}")

        if not temp_metas:
            logger.warning("[FactIndexer] No documents found.")
            if progress_callback:
                progress_callback(total_steps, total_steps)
            return

        logger.info(f"[FactIndexer] Total chunks to index: {len(texts_to_embed)}")
        logger.info("[FactIndexer] Generating Embeddings (Batch)...")

        try:
            vectors = []
            # Batch size - use config or environment variable
            batch_size = config.EMBEDDING_BATCH_SIZE
            total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size
            for i, idx in enumerate(tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding")):
                batch_text = texts_to_embed[idx:idx+batch_size]
                batch_vecs = self.embedding_fn(batch_text)
                vectors.extend(batch_vecs)
                # 报告Embedding进度 (20% -> 60%)
                if progress_callback and i % 10 == 0:
                    progress_callback(1 + int((i / total_batches) * 2), total_steps)
        except Exception as e:
            logger.error(f"[FactIndexer] Embedding generation failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Combine into Qdrant Points
        from qdrant_client.http.models import PointStruct
        points_to_upsert = []
        for i, (pid, payload) in enumerate(temp_metas):
            points_to_upsert.append(PointStruct(id=pid, vector=vectors[i], payload=payload))

        logger.info(f"[FactIndexer] Inserting {len(points_to_upsert)} points into Qdrant...")

        # 报告生成完成进度 (60%)
        if progress_callback:
            progress_callback(2, total_steps)

        # Batch upsert
        upsert_batch = 100
        total_upserts = (len(points_to_upsert) + upsert_batch - 1) // upsert_batch
        for i, idx in enumerate(tqdm(range(0, len(points_to_upsert), upsert_batch), desc="Upserting")):
            batch = points_to_upsert[idx:idx+upsert_batch]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            # 报告Upsert进度 (60% -> 100%)
            if progress_callback:
                progress_callback(2 + int((i / total_upserts) * 1), total_steps)

        logger.info("[FactIndexer] Indexing complete.")
        # Rebuild BM25 after indexing
        self._build_bm25_index()

    def search_facts(self, query, filter_dict=None, top_k=3):
        """
        Hybrid Search: Vector + BM25 with RRF Fusion
        """
        # 1. Vector Search
        vector_hits = self._search_vector(query, filter_dict, top_k=top_k*2)

        # 2. BM25 Search
        bm25_hits = self._search_bm25(query, filter_dict, top_k=top_k*2)

        # 3. RRF Fusion
        fused_results = self._rrf_fusion(vector_hits, bm25_hits, k=60)

        return fused_results[:top_k]

    async def search_facts_async(self, query: str, filter_dict: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        异步搜索方法 - 使用 run_in_executor 包装同步搜索
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.search_facts, query, filter_dict, top_k
        )

    def _search_bm25(self, query, filter_dict=None, top_k=5):
        if not self.bm25:
            return []

        tokenized_query = list(jieba.cut(query))
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Get top indices
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)

        results = []
        count = 0
        for i in top_indices:
            if doc_scores[i] <= 0:
                break

            doc = self.doc_map[i]

            # Apply Filter
            if filter_dict:
                match = True
                for k, v in filter_dict.items():
                    # Simple metadata check
                    if doc['metadata'].get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            results.append({
                "content": doc['content'],
                "metadata": doc['metadata'],
                "id": doc['id'],
                "score": doc_scores[i]
            })
            count += 1
            if count >= top_k:
                break
        return results

    def _search_vector(self, query, filter_dict=None, top_k=3):
        logger.debug(f"[FactIndexer] Vector Searching: {query}")

        # 1. Embed Query
        try:
            query_vector = self.embedding_fn([query])[0]
        except Exception as e:
            logger.error(f"[FactIndexer] Embedding failed: {e}")
            return []

        # 2. Build Filter
        query_filter = None
        if filter_dict:
            conditions = []
            for k, v in filter_dict.items():
                conditions.append(
                    models.FieldCondition(
                        key=k,
                        match=models.MatchValue(value=v)
                    )
                )
            if conditions:
                query_filter = models.Filter(must=conditions)

        # 3. Search
        try:
            if hasattr(self.client, "search"):
                 hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k
                )
            else:
                 hits = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k
                 ).points
        except Exception as e:
            logger.error(f"[FactIndexer] Qdrant search error: {e}")
            return []

        # 4. Format Results
        refs = []
        for hit in hits:
            refs.append({
                "content": hit.payload.get("text", ""),
                "metadata": hit.payload.get("full_metadata", {}),
                "distance": hit.score,
                "id": hit.id
            })

        return refs

    def _rrf_fusion(self, vector_results, bm25_results, k=60):
        """
        Reciprocal Rank Fusion
        """
        scores = {}

        # Helper to process results
        def process_list(results, weight=1.0):
            for rank, item in enumerate(results):
                doc_content = item['content'] # Use content as key for deduplication
                if doc_content not in scores:
                    scores[doc_content] = {
                        "score": 0.0,
                        "data": item
                    }
                scores[doc_content]["score"] += weight / (k + rank + 1)

        process_list(vector_results)
        process_list(bm25_results)

        # Sort by fused score
        sorted_docs = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["data"] for item in sorted_docs]

if __name__ == "__main__":
    indexer = FactIndexer()
    if indexer.count() == 0:
        indexer.index_knowledge_base()

    res = indexer.search_facts("洛天依代言过必胜客吗")
    for r in res:
        logger.debug(f"\n--- Result ({r['metadata'].get('source', '?')}) ---\n{r['content'][:100]}...")