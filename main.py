import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from common.logger import setupLogger
from config import settings
import os

def main():
    """애플리케이션 메인 실행 함수"""
    # 결과 저장 폴더가 없으면 생성
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    
    # 로거 설정 (필요시)
    # setupLogger()
    
    print("OM Analyzer 프로그램을 시작합니다.")
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()