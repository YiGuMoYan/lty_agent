import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_core.agent import ActionAgent

def main():
    agent = ActionAgent()
    agent.chat_loop()

if __name__ == "__main__":
    main()
