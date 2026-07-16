@echo off
REM obsidian_wiki_skill wrapper for Windows (cmd.exe)
REM
REM 用法：
REM   scripts\wiki.cmd <project_root> "<query>" --k 5 --json
REM   scripts\wiki.cmd build-index <project_root>
REM   scripts\wiki.cmd build-graph <project_root>
REM   scripts\wiki.cmd update <project_root> [--apply]
REM
REM Python 解释器选择优先级：
REM   1. 环境变量 WIKI_VENV_PYTHON（或 .env 中配置）
REM   2. PATH 中的 python
setlocal enabledelayedexpansion

REM 定位 skill_dir（本脚本在 scripts\ 下，上一级即 skill_dir）
set "SCRIPT_DIR=%~dp0"
set "SKILL_DIR=%SCRIPT_DIR%.."

REM 加载 .env 中的 WIKI_VENV_PYTHON（若存在）
set "VENV_PY=python"
if exist "%SKILL_DIR%\.env" (
    for /f "usebackq tokens=1,*=2 delims==" %%a in ("%SKILL_DIR%\.env") do (
        set "key=%%a"
        set "val=%%b"
        if /i "!key!"=="WIKI_VENV_PYTHON" set "VENV_PY=!val!"
    )
)

REM 固化环境变量
set "PYTHONDONTWRITEBYTECODE=1"

REM 子命令路由
set "SUBCMD=%1"
if "%SUBCMD%"=="" goto :help
if "%SUBCMD%"=="-h" goto :help
if "%SUBCMD%"=="--help" goto :help
if "%SUBCMD%"=="help" goto :help

if /i "%SUBCMD%"=="build-index" (
    shift
    "%VENV_PY%" "%SKILL_DIR%\scripts\build_index.py" %2 %3 %4 %5
    goto :eof
)
if /i "%SUBCMD%"=="build-graph" (
    shift
    "%VENV_PY%" "%SKILL_DIR%\scripts\build_graph.py" %2 %3 %4 %5
    goto :eof
)
if /i "%SUBCMD%"=="update" (
    shift
    "%VENV_PY%" "%SKILL_DIR%\scripts\update_wiki.py" %2 %3 %4 %5
    goto :eof
)
if /i "%SUBCMD%"=="caption-list" (
    shift
    "%VENV_PY%" "%SKILL_DIR%\scripts\picture_caption.py" %2 list %3 %4 %5
    goto :eof
)
if /i "%SUBCMD%"=="caption-apply" (
    set "PROJ=%2"
    set "CAPS=%3"
    "%VENV_PY%" "%SKILL_DIR%\scripts\picture_caption.py" "!PROJ!" apply "!CAPS!"
    goto :eof
)

REM 默认：当作 query 调用
"%VENV_PY%" "%SKILL_DIR%\scripts\query.py" %*
goto :eof

:help
echo obsidian_wiki_skill wrapper
echo.
echo 用法:
echo   wiki.cmd ^<project_root^> "^<query^>" [query 选项]    检索知识库
echo   wiki.cmd build-index ^<project_root^>               构建索引
echo   wiki.cmd build-graph ^<project_root^>               构建图谱
echo   wiki.cmd update ^<project_root^> [--apply]          增量更新
echo   wiki.cmd caption-list ^<project_root^> [--limit N]  列出待标注图片
echo   wiki.cmd caption-apply ^<project_root^> ^<json^>      写入 caption
echo.
echo Python 解释器:
echo   优先用环境变量 WIKI_VENV_PYTHON（或 ^<skill_dir^>\.env 中配置），
echo   否则 fallback 到 PATH 中的 python。
echo.
echo 配置:
echo   一次性配置写入 ^<skill_dir^>\.env（复制 .env.example 为 .env 填值）。
echo   scripts\_config.py 会自动加载到所有 Python 脚本。
goto :eof
