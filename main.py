# -*- coding: utf-8 -*-

import sys
import os
import argparse

# 设置编码
if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul')

from rag_core.agent.companion_agent import CompanionAgent
from rag_core.emotions.emotional_memory import EmotionalMemory

def print_banner():
    """打印启动横幅"""
    print("=" * 60)
    print("洛天依 LTY-Omni-Agent (情感陪伴版)")
    print("=" * 60)
    print("新功能：智能情感陪伴 • 温暖回应 • 长期记忆")
    print("=" * 60)

def print_help():
    """打印帮助信息"""
    print("使用指南:\n")
    print("模式选择:")
    print("  --emotional, -e    情感陪伴模式 (默认)")
    print("  --regular, -r      普通RAG模式")
    print("  --style STYLE      设置回复风格 (casual/professional/concise)")
    print("  --help-ui         显示帮助信息\n")
    print("情感陪伴模式特色:")
    print("  • 智能情感识别 (开心/难过/焦虑/孤独等)")
    print("  • 长期情感记忆与用户画像")
    print("  • 温暖共情回应\n")
    print("普通RAG模式特色:")
    print("  • 知识图谱查询")
    print("  • 歌词信息检索")
    print("  • DeepSearch多跳检索")
    print("  • 事实核查与归因\n")
    print("回复风格选项:")
    print("  • casual (口语化): 日常交流风格，柔和自然")
    print("  • professional (专业): 知识查询风格，准确清晰")
    print("  • concise (简短): 高效交流风格，简洁明了\n")
    print("交互命令:")
    print("  help              显示帮助")
    print("  switch mode      切换模式")
    print("  set style [casual|professional|concise]  设置回复风格")
    print("  status           查看当前状态")
    print("  memory           查看情感记忆")
    print("  exit, quit       退出程序\n")
    print("开始你的洛天依陪伴之旅吧！")

def print_mode_info(emotional_mode):
    """显示当前模式信息"""
    if emotional_mode:
        print("情感陪伴模式已启动 - 天依将作为你的情感陪伴伙伴")
        print("   特色：温暖共情 • 个性化回应 • 长期记忆")
    else:
        print("普通RAG模式已启动 - 天依将作为知识查询助手")
        print("   特色：知识检索 • 事实核查 • DeepSearch")

def interactive_mode_switch(agent, current_mode):
    """交互式模式切换"""
    print("\n模式切换")
    print("1. 情感陪伴模式")
    print("2. 普通RAG模式")
    print("3. 返回主菜单")
    
    try:
        choice = input("请选择模式 (1-3): ").strip()
        
        if choice == "1":
            return True  # 情感陪伴模式
        elif choice == "2":
            return False  # 普通模式
        elif choice == "3":
            return current_mode
        else:
            print("无效选择，保持当前模式")
            return current_mode
    except Exception as e:
        print(f"模式切换失败: {e}")
        return current_mode

def show_memory_status(agent):
    """显示情感记忆状态"""
    if hasattr(agent, 'emotional_memory') and agent.emotional_memory:
        try:
            summary = agent.emotional_memory.get_profile_summary()
            print("\n情感记忆状态:")
            print(f"  总互动次数: {summary['total_interactions']}")
            print(f"  关系深度: {summary['relationship_depth']:.2f}")
            print(f"  信任度: {summary['trust_level']:.2f}")
            if summary['dominant_emotions']:
                emotions_text = ", ".join([f"{emo}({count})" for emo, count in summary['dominant_emotions']])
                print(f"  主要情感: {emotions_text}")
            print(f"  最后互动: {summary['last_interaction']}")
        except Exception as e:
            print(f"获取记忆状态失败: {e}")
    else:
        print("\n当前为普通模式，无情感记忆功能")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="洛天依 LTY-Omni-Agent 情感陪伴版")
    parser.add_argument('--emotional', '-e', action='store_true',
                       help='启动情感陪伴模式 (默认)')
    parser.add_argument('--regular', '-r', action='store_true',
                       help='启动普通RAG模式')
    parser.add_argument('--style', type=str, default=None,
                       choices=['casual', 'professional', 'concise'],
                       help='设置回复风格 (casual/professional/concise)')
    parser.add_argument('--help-ui', action='store_true',
                       help='显示帮助信息')

    args = parser.parse_args()
    
    if args.help_ui:
        print_help()
        return
    
    # 确定启动模式
    emotional_mode = True  # 默认情感陪伴模式
    if args.regular:
        emotional_mode = False
    
    print_banner()

    try:
        print("正在连接天依核心系统...")
        agent = CompanionAgent(use_emotional_mode=emotional_mode, style=args.style)
        print_mode_info(emotional_mode)
        if args.style:
            print(f"\n当前回复风格: {agent.get_current_style().value}")
        print("\n天依上线啦！(输入 'help' 查看命令)")

    except Exception as e:
        print(f"启动失败: {e}")
        return

    while True:
        try:
            # 显示提示符
            mode_indicator = "情感" if emotional_mode else "普通"
            user_input = input(f"{mode_indicator} You: ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.lower() in ["exit", "quit"]:
                print("天依: 下次见哟！期待我们的下次相遇~")
                break
            elif user_input.lower() == "help":
                print_help()
                continue
            elif user_input.lower() == "switch mode":
                new_mode = interactive_mode_switch(agent, emotional_mode)
                if new_mode != emotional_mode:
                    emotional_mode = new_mode
                    # 重新创建代理
                    print("\n正在切换模式...")
                    agent = CompanionAgent(use_emotional_mode=emotional_mode)
                    print_mode_info(emotional_mode)
                continue
            elif user_input.lower() == "status":
                print("\n当前状态:")
                print(f"  模式: {'情感陪伴模式' if emotional_mode else '普通RAG模式'}")
                show_memory_status(agent)
                continue
            elif user_input.lower() == "memory":
                show_memory_status(agent)
                continue
            elif user_input.lower().startswith("set style"):
                parts = user_input.split()
                if len(parts) >= 3:
                    style_name = parts[2]
                    if agent.set_style(style_name):
                        print(f"当前回复风格: {agent.get_current_style().value}")
                else:
                    print("用法: set style [casual|professional|concise]")
                    print("可用风格:")
                    for style_name, style_desc in agent.get_available_styles().items():
                        print(f"  {style_name}: {style_desc}")
                continue

            # 普通对话
            response = agent.chat(user_input)
            
            # 显示回复
            if emotional_mode:
                print(f"天依: {response}")
            else:
                print(f"天依: {response}")
                
        except KeyboardInterrupt:
            print("\n\n天依: 下次见哟！期待我们的下次相遇~")
            break
        except Exception as e:
            print(f"错误: {e}")
            # 尝试恢复
            try:
                print("尝试重新连接...")
                agent = CompanionAgent(use_emotional_mode=emotional_mode)
            except:
                print("重新连接失败，请重启程序")
                break

if __name__ == "__main__":
    main()