import sys
import os
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_core.companion_agent import CompanionAgent

def run_test_case(agent, name, query):
    print(f"\n[{name}] User: {query}")
    start = time.time()
    resp = agent.chat(query)
    duration = time.time() - start
    print(f"[{name}] Tianyi ({duration:.1f}s): {resp}")
    return resp

def main():
    print("=== Commercial Readiness Test (Comparison & Lists) ===")
    try:
        agent = CompanionAgent()
    except Exception as e:
        print(f"Init Failed: {e}")
        return

    # Turn 1
    run_test_case(agent, "Q1_WHO_WROTE", "你知道勾指起誓是谁写的吗")
    
    # Turn 2: Contextual Switch ("And this one?")
    # Should trigger query_knowledge_graph("达拉崩吧", "composed_by")
    run_test_case(agent, "Q2_CONTEXT_SWITCH", "那达拉崩吧呢")
    
    # List Check
    run_test_case(agent, "Q3_FULL_LIST", "2025 流光协奏 都在哪些城市举办了")

if __name__ == "__main__":
    main()
