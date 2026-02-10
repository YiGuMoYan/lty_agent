#!/bin/bash

echo "================================"
echo "   洛天依情感陪伴系统启动器"
echo "================================"
echo
echo "选择启动模式:"
echo
echo "1. 情感陪伴模式 (推荐)"
echo "2. 普通RAG模式"
echo "3. 显示帮助"
echo "4. 退出"
echo

read -p "请选择 (1-4): " choice

case $choice in
    1)
        echo
        echo "🌸 启动情感陪伴模式..."
        python main.py --emotional
        ;;
    2)
        echo
        echo "📚 启动普通RAG模式..."
        python main.py --regular
        ;;
    3)
        echo
        python main.py --help-ui
        echo
        read -p "按回车键继续..."
        ;;
    4)
        echo
        echo "👋 再见！"
        exit 0
        ;;
    *)
        echo
        echo "❌ 无效选择，请重新运行"
        read -p "按回车键继续..."
        ;;
esac