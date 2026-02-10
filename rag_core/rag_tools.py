import json
import re  # Added for keyword extraction
from .indexing.lyrics_indexer import LyricsIndexer
from .indexing.fact_indexer import FactIndexer
from .indexing.graph_indexer import GraphIndexer

# Global instances
lyrics_idx = LyricsIndexer()
fact_idx = FactIndexer()
graph_idx = GraphIndexer()

# --- Commercial Robustness: Startup Index Check ---
# Only perform full scan if the collection is empty
try:
    if fact_idx.count() == 0:
        print("[RAG Tools] Initializing Vector DB for the first time...")
        fact_idx.index_knowledge_base()
    else:
        print(f"[RAG Tools] Vector DB ready ({fact_idx.count()} chunks). Use scripts to refresh.")
except Exception as e:
    print(f"[RAG Tools] Auto-indexing warning: {e}")

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

    # New Feature: Artist Search
    artist = kwargs.get("artist_name")
    if artist:
        print(f"[Tool] search_lyrics: artist='{artist}'")
        songs = lyrics_idx.get_songs_by_artist(artist)
        if songs:
            # Return list of titles
            titles = [s.get("song_title") for s in songs[:10]]
            return json.dumps({"artist": artist, "songs": titles}, ensure_ascii=False)
        return json.dumps({"status": "not_found", "message": f"No songs found for artist {artist}"})

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
    
    # --- Commercial Robustness: Topic-Specific Priority Search ---
    # If the query contains a known topic name (e.g. "COP", "ilem"), 
    # and we find a file matching that topic, we prioritize it.
    topic_results = []
    # Clean query and extract potential topic keywords (only Nouns/Names)
    keywords = re.findall(r'[\u4e00-\u9fffA-Za-z0-9]+', query)
    stop_words = {"the", "and", "meaning", "perspective", "song", "who", "wrote", "of", "about", "for", "is", "was", "to", "this", "that", "it"}
    
    for kw in keywords:
        if len(kw) < 2 or kw.lower() in stop_words: continue
        
        # 1. Fuzzy Check via Graph (The "Did you mean?" layer)
        # We search graph for this keyword. If matches found, we use the MATCHED entity name.
        graph_matches = graph_idx.search_graph(kw)
        target_topic = kw # Default to asking strictly
        
        if graph_matches:
            # Use the "result" field from graph search which is the standardized node name
            # Graph search returns list of dicts. We look for 'DirectMatch' or best candidate.
            best_match = graph_matches[0]["result"]
            if best_match != kw:
                print(f"[RAG Tools] Auto-Correcting '{kw}' -> '{best_match}' (via Graph)")
                target_topic = best_match
        
        # 2. Topic Search in Vector DB (High Priority)
        # Now we search for the CORRECTED topic
        matches = fact_idx.search_facts(target_topic, filter_dict={"topic": target_topic}, top_k=2)
        if matches:
            topic_results.extend(matches)
    
    filters = {"category": filter_category} if filter_category else None
    vector_results = fact_idx.search_facts(query, filters, top_k=5)
    
    # Merge, prioritizing topic matches
    all_results = topic_results + vector_results
    
    # Deduplicate by content
    seen = set()
    unique_results = []
    for r in all_results:
        if r["content"] not in seen:
            seen.add(r["content"])
            unique_results.append(r)
    
    # Compress output for LLM
    compressed = []
    for r in unique_results[:5]: # Limit to top 5 even after merge
        compressed.append({
            "content": r["content"],
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
                    "song_title": {"type": "string", "description": "The title of the song to search lyrics for"},
                    "artist_name": {"type": "string", "description": "Search for songs by a specific artist/producer (e.g. 'ilem', 'COP')"}
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
