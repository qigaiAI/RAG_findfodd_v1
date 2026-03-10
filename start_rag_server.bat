@echo off

echo === 启动后端服务器 ===
echo 当前目录: %cd%
echo Python版本:
python --version

echo 启动rag_server.py...
python rag_server.py

echo 按任意键退出...
pause > nul
