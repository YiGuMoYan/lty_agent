import chromadb
import networkx as nx
from config import DB_DIR, DEBUG
from rag_core.graph_loader import load_graph
from utils.text_tools import get_simplified_pinyin

class ResonanceEngine:
    def __init__(self):
        self.graph = load_graph()
        self.collection = None
        self._init_chroma()

    def _init_chroma(self):
        try:
            client = chromadb.PersistentClient(path=DB_DIR)
            self.collection = client.get_or_create_collection(name="lty_knowledge")
            if DEBUG: print("  [System] ChromaDB Connected (lty_knowledge).")
        except Exception as e:
            print(f"  [System] ChromaDB Init Failed: {e}")

    def _fuzzy_pinyin_search(self, q_text, q_pinyin):
        if not self.graph: return None
        # Exact Pinyin Match
        for node in self.graph.nodes:
            node_py = self.graph.nodes[node].get('pinyin', '')
            if q_pinyin == node_py: return node
        # Partial
        for node in self.graph.nodes:
            node_py = self.graph.nodes[node].get('pinyin', '')
            if q_pinyin in node_py: return node
        return None

    def deep_search(self, query):
        """
        Hybrid search: Graph + Vector.
        Includes logic for 'next_line' prediction and fact retrieval.
        """
        # Clean query using utils if needed, but here we keep it raw-ish
        q_norm = query.replace("心率", "心律").replace("计协", "机械")
        q_pinyin = get_simplified_pinyin(q_norm)
        report = {"facts": [], "context": [], "lore": [], "sequence": []}
        
        if self.graph:
            # 1. Node Matching
            found_nodes = set()
            for node in self.graph.nodes:
                label = self.graph.nodes[node].get('label', node)
                if q_norm.lower() in label.lower() or q_norm.lower() in node.lower():
                    found_nodes.add(node)
            
            # 2. Fuzzy fallback
            if len(found_nodes) < 2:
                fuzzy = self._fuzzy_pinyin_search(q_norm, q_pinyin)
                if fuzzy: found_nodes.add(fuzzy)
            
            # 3. Traverse
            for node in found_nodes:
                self._process_node(node, report)

        # 4. Vector Search
        if self.collection:
            res = self.collection.query(query_texts=[q_norm], n_results=5)
            if res['documents']:
                for idx, doc in enumerate(res['documents'][0]):
                    meta = res['metadatas'][0][idx]
                    if meta.get('type') == 'lyric': report["context"].append(doc)
                    else: report["lore"].append(doc)
        
        return self._format_report(report)

    def _process_node(self, node, report):
        label = self.graph.nodes[node].get('label', node)
        node_data = self.graph.nodes[node]
        
        # Lyric Sequence Logic
        if node_data.get('type') == 'lyric_line':
            self._trace_lyric_sequence(node, label, report)

        # Neighbor Relations
        rels = []
        for n in self.graph.neighbors(node):
            for key, data in self.graph.get_edge_data(node, n).items():
                target_label = self.graph.nodes[n].get('label', n)
                rels.append(f"{label} --[{data['relation']}]--> {target_label}")
        
        if rels:
            limit = 20 if node_data.get('type') in ['year', 'date'] else 15
            report["facts"].append("\n".join(rels[:limit]))

    def _trace_lyric_sequence(self, node, label, report):
        curr = node
        seq_chain = []
        for _ in range(2): # Look ahead 2 lines
            found_next = False
            for n in self.graph.neighbors(curr):
                for k, d in self.graph.get_edge_data(curr, n).items():
                    if d['relation'] == 'next_line':
                        seq_chain.append(self.graph.nodes[n].get('label', n))
                        curr = n
                        found_next = True
                        break
                if found_next: break
            if not found_next: break
        
        if seq_chain:
            report["sequence"].append(f"【歌词接龙预测】: {label} -> " + " -> ".join(seq_chain))

    def _format_report(self, r):
        lines = ["【Resonance 同步完成】音频背景资产已解密："]
        if r["sequence"]: lines.append("\n".join(r["sequence"]))
        if r["facts"]: lines.append("\n[图谱关联网络]:\n" + "\n".join(r["facts"][:5]))
        if r["context"]: lines.append("\n[增强旋律背景]:\n" + "\n".join(r["context"][:2]))
        if r["lore"]: lines.append("\n[瓦纳海姆设定存档]:\n" + "\n".join(r["lore"][:1]))
        return "\n".join(lines) if len(lines) > 1 else "【信号不足】无法穿透时空杂讯，请尝试提供确切歌名。"
