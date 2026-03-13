@echo off
REM OM_analyzer 실행 스크립트

REM 가상환경(venv)이 프로젝트 폴더 내에 있다고 가정합니다.
REM 가상환경을 사용하지 않는 경우 아래 두 줄은 삭제하거나 주석처리 하세요.
echo Activating virtual environment...
CALL .\\venv\\Scripts\\activate.bat

echo Starting OM Analyzer...
REM main.py를 실행합니다.
python main.py

REM 프로그램 종료 후 잠시 대기 (오류 메시지 확인용)
pause