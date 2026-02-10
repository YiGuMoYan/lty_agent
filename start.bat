@echo off
chcp 65001 >nul
title 洛天依情感陪伴系统

echo.
echo ================================
echo   洛天依情感陪伴系统
echo ================================
echo.
echo 1. 情感陪伴模式启动
echo 2. 普通RAG模式启动  
echo 3. 快速测试
echo 4. 显示帮助
echo 5. 退出
echo.

:menu
set /p choice=请选择 (1-5): 

if "%choice%"=="1" (
    echo.
    echo 情感陪伴模式启动...
    python main.py --emotional
) else if "%choice%"=="2" (
    echo.
    echo 普通RAG模式启动...
    python main.py --regular
) else if "%choice%"=="3" (
    echo.
    echo 运行快速测试...
    python scripts/test_emotional_simple.py
) else if "%choice%"=="4" (
    echo.
    echo 查看帮助信息...
    python main.py --help-ui
    pause
) else if "%choice%"=="5" (
    echo.
    echo 再见！
    exit
) else (
    echo.
    echo 无效选择，请重新运行
    pause
)

goto menu