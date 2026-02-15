
import json
import os
import asyncio
from typing import List, Dict, Any
import jieba
from rank_bm25 import BM25Okapi
import numpy as np
from rag_core.utils.logger import logger

class LyricsIndexer:
    def __init__(self, data_path=None):
        """
        Initialize the LyricsIndexer.
        :param data_path: Path to the lyrics.jsonl file.
        """
        if data_path is None:
            # Default path relative to project root possibility, or passed explicitly
            # Assuming widely used default: dataset/song/lyrics.jsonl
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_path = os.path.join(base_dir, "dataset", "song", "lyrics.jsonl")
        
        self.data_path = data_path
        self.songs = []
        self.bm25 = None
        self.tokenized_corpus = []
        
        # Load and build if file exists
        if os.path.exists(self.data_path):
            self.load_data()
            self.build_index()
        else:
            logger.warning(f"[LyricsIndexer] Warning: Data file not found at {self.data_path}")

    def load_data(self):
        """Load lyrics from JSONL file."""
        self.songs = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.songs.append(json.loads(line))
            logger.info(f"[LyricsIndexer] Loaded {len(self.songs)} songs.")
        except Exception as e:
            logger.error(f"[LyricsIndexer] Error loading data: {e}")

    def _tokenize(self, text):
        """Tokenize Chinese text using jieba."""
        # Use simple precise mode
        return list(jieba.cut(text))

    def build_index(self):
        """Build BM25 index from lyrics."""
        logger.info("[LyricsIndexer] Building BM25 index...")
        self.tokenized_corpus = [self._tokenize(song.get('lyrics', '')) for song in self.songs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("[LyricsIndexer] Index built successfully.")

    def search_lyrics(self, query, top_k=3):
        """
        Search for songs containing the query in lyrics.
        :param query: The snippet of lyrics to search for.
        :param top_k: Number of results to return.
        :return: List of matches (dict with song info and score).
        """
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices
        top_n = np.argsort(doc_scores)[::-1][:top_k]

        results = []
        for idx in top_n:
            score = doc_scores[idx]
            if score > 0: # Filter out zero relevance
                song = self.songs[idx]
                results.append({
                    "song_title": song.get("song_title"),
                    "p_masters": song.get("p_masters"),
                    "lyrics_snippet": song.get("lyrics", "")[:100] + "...", # Preview
                    "full_lyrics": song.get("lyrics", ""),
                    "score": float(score)
                })

        return results

    async def search_lyrics_async(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        异步搜索歌词 - 使用 run_in_executor 包装同步搜索
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search_lyrics, query, top_k)

    def get_song_by_title(self, title):
        """
        Find exact or fuzzy match for song title.
        Using simple inclusion or overlap for "fuzzy" match in this version.
        """
        candidates = []
        for song in self.songs:
            db_title = song.get("song_title", "")
            if title == db_title:
                return [song] # Exact match priority
            if title in db_title or db_title in title:
                candidates.append(song)
        
        return candidates

    def get_songs_by_artist(self, artist_name):
        """
        Find songs by artist (P-Master).
        """
        results = []
        # Robust Logic: Check if artist_name is in p_masters list
        # p_masters is usually a list of strings, e.g. ["ilem", "Luo Tianyi"]
        # or maybe just a string in some dirty data?
        for song in self.songs:
            masters = song.get("p_masters", [])
            # Handle if masters is string or list
            if isinstance(masters, str):
                masters = [masters]
            
            # Case-insensitive check
            for m in masters:
                if artist_name.lower() in m.lower():
                    results.append(song)
                    break 
        return results

if __name__ == "__main__":
    # Simple test
    indexer = LyricsIndexer()
    
    # Test 1: Search Lyrics
    q = "好想吃布丁"
    logger.debug(f"\nSearching for: {q}")
    results = indexer.search_lyrics(q)
    for res in results:
        logger.debug(f"Title: {res['song_title']}, Score: {res['score']}")

    # Test 2: Search Title
    t = "66CCFF"
    logger.debug(f"\nSearching for title: {t}")
    songs = indexer.get_song_by_title(t)
    for s in songs:
        logger.debug(f"Found: {s['song_title']}")
