import os
import re
import uuid
import json
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models

class FactIndexer:
    def __init__(self, persist_directory=None):
        """
        Initialize FactIndexer with Qdrant (Local Mode).
        """
        if persist_directory is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            persist_directory = os.path.join(base_dir, "dataset", "vector_store", "qdrant_lty")

        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        print(f"[FactIndexer] Initializing Qdrant at {persist_directory}")
        # Initialize Qdrant in local mode (path-based)
        self.client = QdrantClient(path=persist_directory)
        self.collection_name = "lty_facts"

        # Determine vector dimension dynamically
        from rag_core.embeddings import get_embedding_function, LocalBGEEmbeddingFunction
        self.embedding_fn = get_embedding_function()

        if isinstance(self.embedding_fn, LocalBGEEmbeddingFunction):
            self.vector_dim = 1024
        else:
            self.vector_dim = 1536 # DashScope defaults

        # Create collection if not exists
        if not self.client.collection_exists(self.collection_name):
            print(f"[FactIndexer] Creating collection {self.collection_name} with dim={self.vector_dim}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE
                )
            )

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

    def index_knowledge_base(self, kb_root=None, lyrics_path=None):
        """Walk KB directory and index all md files and lyrics."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if kb_root is None:
            kb_root = os.path.join(base_dir, "dataset", "knowledge_base")

        if lyrics_path is None:
            lyrics_path = os.path.join(base_dir, "dataset", "song", "cleaned_lyrics.jsonl")

        print(f"[FactIndexer] Scanning Knowledge Base: {kb_root}")

        texts_to_embed = []
        temp_metas = [] # List of (uuid, payload)

        # 1. Scan Markdown Files
        md_files = []
        for root, dirs, files in os.walk(kb_root):
            for file in files:
                if file.endswith(".md"):
                    md_files.append(os.path.join(root, file))

        print(f"[FactIndexer] Found {len(md_files)} Markdown files.")

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

        # 2. Scan Lyrics
        if os.path.exists(lyrics_path):
            print(f"[FactIndexer] Loading Lyrics from: {lyrics_path}")
            try:
                with open(lyrics_path, 'r', encoding='utf-8') as f:
                    lyrics_data = [json.loads(line) for line in f if line.strip()]

                print(f"[FactIndexer] Found {len(lyrics_data)} songs.")

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
                print(f"[FactIndexer] Error loading lyrics: {e}")

        if not temp_metas:
            print("[FactIndexer] No documents found.")
            return

        print(f"[FactIndexer] Total chunks to index: {len(texts_to_embed)}")
        print("[FactIndexer] Generating Embeddings (Batch)...")

        try:
            vectors = []
            # Batch size lowered to prevent CUDA OOM
            batch_size = 4
            for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding"):
                batch_text = texts_to_embed[i:i+batch_size]
                batch_vecs = self.embedding_fn(batch_text)
                vectors.extend(batch_vecs)
        except Exception as e:
            print(f"[FactIndexer] Embedding generation failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Combine into Qdrant Points
        from qdrant_client.http.models import PointStruct
        points_to_upsert = []
        for i, (pid, payload) in enumerate(temp_metas):
            points_to_upsert.append(PointStruct(id=pid, vector=vectors[i], payload=payload))

        print(f"[FactIndexer] Inserting {len(points_to_upsert)} points into Qdrant...")

        # Batch upsert
        upsert_batch = 100
        for i in tqdm(range(0, len(points_to_upsert), upsert_batch), desc="Upserting"):
            batch = points_to_upsert[i:i+upsert_batch]
            self.client.upsert(collection_name=self.collection_name, points=batch)

        print("[FactIndexer] Indexing complete.")

    def search_facts(self, query, filter_dict=None, top_k=3):
        """
        Search for facts.
        """
        print(f"[FactIndexer] Searching: {query} with filter {filter_dict}")

        # 1. Embed Query
        try:
            query_vector = self.embedding_fn([query])[0]
        except Exception as e:
            print(f"[FactIndexer] Embedding failed: {e}")
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
        # Explicit check for search method availability (Qdrant client version compatibility)
        if hasattr(self.client, "search"):
             hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k
            )
        else:
            # Fallback for some local clients or older versions
             hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k
             ).points

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

if __name__ == "__main__":
    indexer = FactIndexer()
    if indexer.count() == 0:
        indexer.index_knowledge_base()

    res = indexer.search_facts("洛天依代言过必胜客吗")
    for r in res:
        print(f"\n--- Result ({r['metadata'].get('source', '?')}) ---\n{r['content'][:100]}...")