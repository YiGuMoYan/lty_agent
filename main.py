
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_core.companion_agent import CompanionAgent

def main():
    print("=== 洛天依 LTY-Omni-Agent (Commercial Ver.) ===")
    print("正在连接共鸣雷达...")
    try:
        agent = CompanionAgent()
        print("天依上线啦！(输入 'exit' 退出)")
    except Exception as e:
        print(f"启动失败: {e}")
        return

    while True:
        try:
            user_input = input("\nUser: ")
            if not user_input.strip():
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("天依: 下次见哟！ByeBye~")
                break
                
            response = agent.chat(user_input)
            print(f"Luo Tianyi: {response}")
            
        except KeyboardInterrupt:
            print("\n天依: 下次见哟！ByeBye~")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
