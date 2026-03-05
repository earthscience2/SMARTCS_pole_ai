@echo off
REM UTF-8 인코딩으로 Python 스크립트 실행
REM 사용법: run_with_utf8.bat "python script.py"

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
chcp 65001 >nul

%*