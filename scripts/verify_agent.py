import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_core.companion_agent import CompanionAgent

def test_agent():
    print("Initializing Agent...")
    try:
        agent = CompanionAgent()
    except Exception as e:
        print(f"Agent Init Error: {e}")
        return

    # User Query 4: Ambiguous Concert Query
    # This previously failed because "My personal concert" doesn't exist.
    # Now it should keyword match "演唱会" or be routed better.
    q4 = "还有什么，你自己的个人演唱会呢" 
    print(f"\n--- User: {q4} ---")
    resp4 = agent.chat(q4)
    print(f"Luo Tianyi: {resp4}")

if __name__ == "__main__":
    test_agent()
