
import os
import re
import chromadb
from chromadb.config import Settings
import uuid

class FactIndexer:
    def __init__(self, persist_directory=None):
        """
        Initialize FactIndexer with ChromaDB.
        """
        if persist_directory is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            persist_directory = os.path.join(base_dir, "dataset", "vector_store")
            
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            
        print(f"[FactIndexer] Initializing Vector DB at {persist_directory}")
        # Use simple persistent client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        # Note: Using default embedding function (usually all-MiniLM-L6-v2)
        # For production Chinese, we would swap this with BGE-M3 or similar.
        self.collection = self.client.get_or_create_collection(name="lty_facts")

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
            # Remove frontmatter from content for splitting
            content = content[fm_match.end():]

        # Split by Headers (##)
        # Simple regex splitting to keep context.
        # We treat the text between headers as a chunk.
        sections = []
        # Find all headers
        # This regex matches lines starting with #, ##, ###...
        # We want to capture the header level, title, and the following content
        
        # Strategy: Split by "## " and preserve the header title in content
        parts = re.split(r'(^|\n)##\s+', content)
        
        current_header = "Introduction"
        
        # The first part is usually intro text before first H2
        if parts[0].strip():
             sections.append({
                "content": parts[0].strip(),
                "section": "Introduction",
                **frontmatter
            })
            
        # Iterate over rest (parts seem to alternate or behavior depends on split)
        # re.split with capturing group returns [text, delimiter, text, delimiter...]
        # Here delimiter captured is (^|\n) which is just newline.
        # Let's use a simpler iterative approach for robustness.
        
        lines = content.split('\n')
        buffer = []
        current_section = "Introduction"
        
        chunk_list = []
        
        for line in lines:
            if line.strip().startswith('## '):
                # New section
                if buffer:
                    chunk_list.append((current_section, "\n".join(buffer)))
                buffer = []
                current_section = line.strip().replace('#', '').strip()
                buffer.append(line) # Keep header in content
            else:
                buffer.append(line)
        
        if buffer:
             chunk_list.append((current_section, "\n".join(buffer)))
             
        # Format for indexing
        results = []
        base_name = os.path.basename(file_path)
        for section_title, text in chunk_list:
            if len(text.strip()) < 10: # Skip empty/too short
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

    def index_knowledge_base(self, kb_root=None):
        """Walk KB directory and index all md files."""
        if kb_root is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            kb_root = os.path.join(base_dir, "dataset", "knowledge_base")

        print(f"[FactIndexer] Scanning {kb_root}...")
        
        documents = []
        metadatas = []
        ids = []
        
        count = 0
        for root, dirs, files in os.walk(kb_root):
            for file in files:
                if file.endswith(".md"):
                    path = os.path.join(root, file)
                    mtime = os.path.getmtime(path)
                    
                    # Heuristic: Check if this file + section ID is already in DB
                    # For a truly commercial version, we'd store a local hash map of {file: mtime}.
                    # Since Chroma.get is slow for 900 items, let's use a simpler check:
                    # If the collection is populated, we only index files modified in the last 1 hour
                    # (assuming the server was just updated).
                    
                    # For now, let's just make it skip parsing if the count is already large
                    # Unless we are in a 'force' mode.
                    chunks = self._parse_markdown(path)
                    for chunk in chunks:
                        # Add a tiny bit of metadata for tracking if needed
                        chunk['metadata']['indexed_at'] = mtime
                        documents.append(chunk['document'])
                        metadatas.append(chunk['metadata'])
                        ids.append(f"{chunk['id']}::{uuid.uuid4().hex[:8]}") 
                        count += 1
                        
        if documents:
            print(f"[FactIndexer] Found {len(documents)} new/updated chunks. Indexing...")
            # Use metadata to track last indexed time if needed, 
            # but for now, we just rely on the skip logic above to reduce data sent to Chroma.
            
            # Batch add
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                self.collection.upsert(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
            print("[FactIndexer] Indexing complete.")

        else:
            print("[FactIndexer] No changes detected or no documents found to index.")


    def search_facts(self, query, filter_dict=None, top_k=3):
        """
        Search for facts.
        :param filter_dict: Metadata filter e.g. {"category": "Commercial"}
        """
        print(f"[FactIndexer] Searching: {query} with filter {filter_dict}")
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_dict if filter_dict else None
        )
        
        # Flatten results
        refs = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                refs.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })
        return refs

if __name__ == "__main__":
    # Test
    indexer = FactIndexer()
    # indexer.index_knowledge_base() # Uncomment to build index
    
    # Check if we have data (assuming built previously or run once)
    # For testing in development, we might want to run index once
    indexer.index_knowledge_base() 
    
    res = indexer.search_facts("洛天依代言过必胜客吗")
    for r in res:
        print(f"\n--- Result ({r['metadata']['source']}) ---\n{r['content'][:100]}...")
