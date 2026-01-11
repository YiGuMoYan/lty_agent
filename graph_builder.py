import json
import os
import re
import networkx as nx
from glob import glob
from pypinyin import pinyin, Style

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'song')
LORE_DIR = os.path.join(os.path.dirname(__file__), 'agents', 'res')
GRAPH_PATH = os.path.join(os.path.dirname(__file__), 'lty_graph.json')

def get_pinyin(text):
    # Convert to pinyin without tones and join
    py_list = pinyin(text, style=Style.NORMAL)
    return "".join([item[0] for item in py_list]).lower()

def build_commercial_graph():
    G = nx.MultiDiGraph()
    print("=== Starting Enterprise GraphRAG Builder (with Pinyin) ===")

    # 1. Process Songs & Lyrics
    lyrics_file = os.path.join(DATA_DIR, 'lyrics.jsonl')
    songs_found = []
    if os.path.exists(lyrics_file):
        print(f"Ingesting lyrics from {lyrics_file}...")
        with open(lyrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    song_title = data.get('song_title', '').strip("- ")
                    p_masters = data.get('p_masters', [])
                    lyrics = data.get('lyrics', '')
                    
                    songs_found.append(song_title)
                    G.add_node(song_title, type="song", label=song_title)
                    
                    # Producers
                    for p in p_masters:
                        p_name = p.strip()
                        G.add_node(p_name, type="producer", label=p_name)
                        G.add_edge(song_title, p_name, relation="produced_by")
                        G.add_edge(p_name, song_title, relation="produced_song")
                    
                    # Lyrics Mapping (Sequential Link + Pinyin)
                    lines = [l.strip() for l in lyrics.split('\n') if len(l.strip()) > 5]
                    prev_node = None
                    for idx, l_text in enumerate(lines):
                        # Add Pinyin for typo-tolerant matching
                        py_text = get_pinyin(l_text)
                        
                        G.add_node(l_text, type="lyric_line", label=l_text, index=idx, pinyin=py_text)
                        G.add_edge(l_text, song_title, relation="belongs_to")
                        
                        # Link to previous line to form a chain
                        if prev_node:
                            G.add_edge(prev_node, l_text, relation="next_line")
                        prev_node = l_text
                except Exception as e: 
                    print(f"Error processing song: {e}")
                    continue

    # 2. Process Lore & Events
    print(f"Extracting factual events from {LORE_DIR}...")
    lore_files = glob(os.path.join(LORE_DIR, "**", "*.md"), recursive=True)
    
    # Common Patterns
    # Added "截至" filter in logic
    date_pattern = r"(\d{4}[年\-\.]\d{1,2}[月\-\.]\d{1,2}日?)"
    
    for f_path in lore_files:
        with open(f_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split by Question blocks primarily
            blocks = re.split(r"(?=问题 \d+)", content)
            for block in blocks:
                if not block.strip(): continue
                
                # Filter out statistical dates (preceded by 截至)
                # We use a greedy approach: find all dates, then check their context
                all_dates = []
                for m in re.finditer(date_pattern, block):
                    start = m.start()
                    # Check 5 chars before
                    context_before = block[max(0, start-10):start]
                    if "截至" in context_before:
                        # Skip statistical cutoff dates
                        continue
                    all_dates.append(m.group(1))
                
                if all_dates:
                    # Heuristic: the first non-"截至" date is usually the event date
                    target_date = all_dates[0]
                    # Clean date format
                    clean_date = target_date.replace('年','.').replace('月','.').replace('日','').strip()
                    # Ensure format is YYYY.M.D
                    clean_date = re.sub(r'(\d+)\.(\d+)\.(\d+)', lambda m: f"{m.group(1)}.{int(m.group(2))}.{int(m.group(3))}", clean_date)
                    
                    G.add_node(clean_date, type="date", label=clean_date)
                    
                    year_match = re.search(r"(\d{4})", target_date)
                    if year_match:
                        year = year_match.group(1)
                        G.add_node(year, type="year", label=year)
                        G.add_edge(year, clean_date, relation="contains_date")
                    
                    # Extract the first meaningful sentence as label
                    lines = block.strip().split('\n')
                    # Skip the "问题 X" line
                    desc_text = " ".join([l.strip() for l in lines[1:] if l.strip()])
                    event_desc = desc_text[:200]
                    
                    event_id = f"evt_{hash(event_desc) % 10**8}"
                    G.add_node(event_id, type="event", label=event_desc)
                    G.add_edge(clean_date, event_id, relation="happened")
                    
                    G.add_node("洛天依", type="character", label="洛天依")
                    G.add_edge(event_id, "洛天依", relation="involves")
                    
                    for song in songs_found:
                        if len(song) > 2 and song in block:
                            G.add_edge(event_id, song, relation="mentions_song")

    # 3. Finalize
    print(f"Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Save as JSON (Human Readable / Backup)
    data = nx.node_link_data(G)
    with open(GRAPH_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved graph JSON to {GRAPH_PATH}")

    # Save as Pickle (Binary / Fast Load)
    PICKLE_PATH = GRAPH_PATH.replace('.json', '.pkl')
    import pickle
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(G, f)
    print(f"Saved binary graph cache to {PICKLE_PATH} (High Performance Load Enabled)")
    
    print(f"Commercial Graph Built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"Output: {GRAPH_PATH}")

if __name__ == "__main__":
    build_commercial_graph()
