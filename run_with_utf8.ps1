# UTF-8 인코딩으로 Python 스크립트 실행
# 사용법: .\run_with_utf8.ps1 "python script.py"

param(
    [Parameter(Mandatory=$true)]
    [string]$Command
)

# UTF-8 환경 변수 설정
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# 콘솔 인코딩을 UTF-8로 설정
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

# 명령 실행
Invoke-Expression $Command