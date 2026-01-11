import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset.data_gen.agents import TaxonomyAgent
from dataset.data_gen.run_deep_dive import deep_dive

def run_taxonomy_phase():
    print("=== Phase 1: Taxonomy Generation ===")
    architect = TaxonomyAgent()
    
    # 1. Generate Taxonomy
    res = architect.generate_master_plan()
    if not res:
        print("Failed to generate taxonomy.")
        return
    
    try:
        data = json.loads(res)
        schema_path = os.path.join(os.path.dirname(__file__), 'topics_master.json')
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Taxonomy saved to {schema_path}")
        print("Please review this file before proceeding to mining.")
    except Exception as e:
        print(f"Error parsing taxonomy: {e}")
        print(res)

def run_mining_phase():
    print("=== Phase 2: Deep Mining (Batch Mode) ===")
    schema_path = os.path.join(os.path.dirname(__file__), 'topics_master.json')
    if not os.path.exists(schema_path):
        print("topics_master.json not found. Run taxonomy phase first.")
        return

def run_mining_phase():
    import concurrent.futures
    print("=== Phase 2: Deep Mining (Batch Mode - Parallel) ===")
    schema_path = os.path.join(os.path.dirname(__file__), 'topics_master.json')
    if not os.path.exists(schema_path):
        print("topics_master.json not found. Run taxonomy phase first.")
        return

    with open(schema_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    full_taxonomy = data.get("taxonomy", [])
    
    # Flatten all tasks
    tasks = []
    for cat_obj in full_taxonomy:
        category = cat_obj.get("category", "Uncategorized")
        subtopics = cat_obj.get("subtopics", [])
        for topic in subtopics:
            tasks.append((topic, category))
    
    print(f"� Found {len(tasks)} topics. Starting parallel mining with 5 threads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        futures = {executor.submit(deep_dive, topic, category=category): topic for topic, category in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            topic = futures[future]
            try:
                future.result() # deep_dive handles internal printing/saving
            except Exception as e:
                print(f"❌ Error extracting '{topic}': {e}")
                
    print("✨ Mining Phase Complete.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mine":
        run_mining_phase()
    else:
        run_taxonomy_phase()
