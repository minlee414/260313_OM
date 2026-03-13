import os
import sqlite3
from config import settings

def initializeDatabase():
    """DB 파일과 테이블가 없으면 생성합니다."""
    db_path = settings.DB_FILE
    
    if os.path.exists(db_path):
        print(f"데이터베이스 파일이 이미 존재합니다: {db_path}")
        # 사용자 입력을 받아 기존 DB를 백업하고 새로 만들 수도 있습니다.
        return

    print(f"데이터베이스 파일을 생성합니다: {db_path}")
    
    try:
        # DB 디렉터리가 없으면 생성
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 분석 결과를 저장할 테이블 생성
        cursor.execute('''
        CREATE TABLE analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            alpha_al_fraction REAL,
            si_inter_fraction REAL,
            pore_fraction REAL,
            analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        print("테이블 'analysis_results' 생성이 완료되었습니다.")
        
    except Exception as e:
        print(f"DB 생성 중 오류 발생: {e}")

if __name__ == "__main__":
    initializeDatabase()