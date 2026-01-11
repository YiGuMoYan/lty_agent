
import json
from .indexing.lyrics_indexer import LyricsIndexer
from .indexing.fact_indexer import FactIndexer
from .indexing.graph_indexer import GraphIndexer

# Global instances (lazy loading recommended, but simpler here)
lyrics_idx = LyricsIndexer()
fact_idx = FactIndexer()
graph_idx = GraphIndexer()

# --- Tool Functions ---

def query_knowledge_graph(entity_name=None, relation_type=None, category=None, **kwargs):
    """
    Query the knowledge graph for structured facts.
    """
    # Robustness: Handle alias if model hallucinates 'query' or 'name' or 'question'
    target = entity_name or kwargs.get("name") or kwargs.get("query") or kwargs.get("question")
    if not target:
        return json.dumps({"status": "error", "message": "Missing 'entity_name' argument"})

    print(f"[Tool] query_knowledge_graph: {target}, {relation_type}")
    results = graph_idx.search_graph(target, relation_type)
    if not results:
        return json.dumps({"status": "not_found", "message": f"No graph node found for {target}"})
    return json.dumps(results[:10], ensure_ascii=False)

def search_lyrics(lyrics_snippet=None, song_title=None, **kwargs):
    """
    Search by lyrics snippet or song title.
    """
    # Robustness: Handle alias
    query = lyrics_snippet or song_title or kwargs.get("query") or kwargs.get("content")
    
    print(f"[Tool] search_lyrics: query='{query}'")
    
    if not query:
         return json.dumps([])

    # Heuristic: If short, assume title; if long, snippet? Or try both.
    # Try logic: exact title match first
    songs = lyrics_idx.get_song_by_title(query)
    if songs:
        return json.dumps(songs[:1], ensure_ascii=False)
        
    # Fallback to snippet search
    songs = lyrics_idx.search_lyrics(query, top_k=3)
    return json.dumps(songs, ensure_ascii=False)

def search_knowledge_base(query, filter_category=None):
    """
    Search vector knowledge base.
    """
    print(f"[Tool] search_knowledge_base: {query} (filter={filter_category})")
    filters = {"category": filter_category} if filter_category else None
    # Increase top_k to ensure we get the full table if it's split across chunks, or just more context
    results = fact_idx.search_facts(query, filters, top_k=5)
    
    # Compress output for LLM
    compressed = []
    for r in results:
        compressed.append({
            "content": r["content"], # Maybe truncate if too long
            "source": r["metadata"]["source"]
        })
    return json.dumps(compressed, ensure_ascii=False)

# --- Schema Definition for Qwen ---

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_graph",
            "description": "Query specific entities and relationships in the knowledge graph. Use this for questions like 'Who composed X?', 'When was Y released?', 'What brands did Z endorse?'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "The primary entity name (e.g. '必胜客', 'ilem', '66CCFF')"},
                    "relation_type": {"type": "string", "description": "Type of relationship to look for (optional)"},
                    "category": {"type": "string", "enum": ["Song", "Person", "Commercial", "Event"]}
                },
                "required": ["entity_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_lyrics",
            "description": "Search for song lyrics. Use this when user asks about specific lyrics lines or wants the lyrics of a song.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lyrics_snippet": {"type": "string", "description": "A snippet of lyrics to search for"},
                    "song_title": {"type": "string", "description": "The title of the song to search lyrics for"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Semantic search in the encyclopedia. Use this for general questions, backstories, detailed descriptions, or when graph search fails.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "filter_category": {"type": "string", "description": "Optional category filter", "enum": ["Timeline_DeepDive", "Discography_Famous", "Commercial_Deals", "Interpersonal_Relationships", "Producers"]}
                },
                "required": ["query"]
            }
        }
    }
]

AVAILABLE_TOOLS = {
    "query_knowledge_graph": query_knowledge_graph,
    "search_lyrics": search_lyrics,
    "search_knowledge_base": search_knowledge_base
}
