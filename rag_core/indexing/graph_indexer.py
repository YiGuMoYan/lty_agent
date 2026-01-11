
import json
import os
import networkx as nx

class GraphIndexer:
    def __init__(self, topics_path=None):
        """
        Initialize GraphIndexer.
        Loads taxonomy and builds a NetworkX graph.
        """
        if topics_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            topics_path = os.path.join(base_dir, "dataset", "data_gen", "topics_master.json")
        
        self.topics_path = topics_path
        self.graph = nx.DiGraph()
        
        if os.path.exists(self.topics_path):
            self.build_graph()
        else:
            print(f"[GraphIndexer] Warning: Taxonomy file not found at {self.topics_path}")

    def build_graph(self):
        """Build graph from topics_master.json."""
        print("[GraphIndexer] Building knowledge graph...")
        try:
            with open(self.topics_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Iterate categories
            # Original structure is {"taxonomy": [{"category": "...", "subtopics": []}]}
            items = data.get("taxonomy", [])
            for entry in items:
                category = entry.get("category")
                subtopics = entry.get("subtopics")
                
                cat_node = f"Category:{category}"
                self.graph.add_node(cat_node, type="Category")
                
                if isinstance(subtopics, list):
                    for item in subtopics:
                        if isinstance(item, str):
                            # Simple topic string
                            self._add_entity(item, category)
                        elif isinstance(item, dict):
                            # Complex timeline object or other struct
                            self._parse_complex_item(item, category)
                            
            print(f"[GraphIndexer] Graph built. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
            
        except Exception as e:
            print(f"[GraphIndexer] Error building graph: {e}")

    def _add_entity(self, name, category):
        """Add an entity node and link to category."""
        if not name: return
        
        # Simple heuristic to determine type/properties from string could go here
        # For now, we link Item -> Category
        
        # Node ID
        node_id = name
        self.graph.add_node(node_id, category=category, type="Topic")
        self.graph.add_edge(node_id, f"Category:{category}", relation="belongs_to")
        
        # Link to Year if present in text (e.g. "2018年...")
        import re
        year_match = re.search(r'(20\d{2})年', name)
        if year_match:
            year = year_match.group(1)
            year_node = f"Year:{year}"
            self.graph.add_node(year_node, type="Year")
            self.graph.add_edge(node_id, year_node, relation="happened_in")

    def _parse_complex_item(self, item, category):
        """Handle dictionary items in taxonomy."""
        # Check for common keys like 'year', 'events', or if it's just a structured topic
        # The current schema seems to be strings mostly, but Timeline might be dicts?
        # Based on previous view, Timeline_DeepDive is a LIST of STRINGS.
        # But let's be robust.
        pass

    def search_graph(self, entity_name, relation_type=None, hops=1):
        """
        Search for entities related to a given entity.
        :param entity_name: Start node.
        :param relation_type: Filter by edge attribute (optional).
        :return: List of related nodes/neighbors.
        """
        results = []
        
        # 1. Try Exact/Startswith Match on Nodes
        if self.graph.has_node(entity_name):
            start_node = entity_name
        else:
            # 2. Keyword Search (Fallback, with Multi-Keyword Support)
            # Support space-separated keywords treated as "AND" logic
            keywords = entity_name.split()
            candidates = []
            
            for node in self.graph.nodes():
                s_node = str(node)
                # Check if ALL keywords are present in the node name
                if all(k in s_node for k in keywords):
                    candidates.append(node)
            
            if candidates:
                # If we found matches, return them immediately as results (not just neighbors)
                # Because the node ITSELF is the fact (e.g. "2025年...演唱会")
                # We return the node content as "source" and its category as "target" (or vice versa)
                refined_results = []
                for cand in candidates[:10]: # Limit to avoid context overflow
                    # Get edges to find context (Category/Year)
                    context_edges = self.graph.out_edges(cand, data=True)
                    rel_info = "related"
                    category_found = "Unknown"
                    
                    for _, target, data in context_edges:
                        rel = data.get('relation')
                        rel_info = f"is_{rel}_of_{target}"
                        if rel == "belongs_to":
                            category_found = str(target).replace("Category:", "")
                        
                    refined_results.append({
                        "result": cand, # The full text of the node
                        "type": "DirectMatch", 
                        "category": category_found,
                        "context": rel_info
                    })
                return refined_results

        if not start_node:
            print(f"[GraphIndexer] No node keywords found for '{entity_name}'")
            return []

        # Get neighbors
        # For now, just direct neighbors (successors and predecessors)
        # In a directed graph, we might want both directions for "relatedness"
        
        neighbors = set(self.graph.successors(start_node)) | set(self.graph.predecessors(start_node))
        
        refined_results = []
        for n in neighbors:
            # Retrieve edge data
            # Edge could be (start, n) or (n, start)
            rel = "related_to"
            if self.graph.has_edge(start_node, n):
                rel = self.graph[start_node][n].get("relation", "related")
            elif self.graph.has_edge(n, start_node):
                rel = self.graph[n][start_node].get("relation", "related")
                
            refined_results.append({
                "source": start_node,
                "target": n,
                "relation": rel
            })
            
        return refined_results

if __name__ == "__main__":
    indexer = GraphIndexer()
    
    # Test
    q = "必胜客"
    print(f"\nSearching graph for: {q}")
    res = indexer.search_graph(q)
    for r in res:
        print(r)
        
    q2 = "2018"
    print(f"\nSearching graph for: {q2}")
    res2 = indexer.search_graph(q2)
    for r in res2:
        print(r)
