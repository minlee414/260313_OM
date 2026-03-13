import sqlite3
from config import settings
import os

def insertResult(filename, analysis_data):
    """
    분석 결과 한 건을 데이터베이스에 삽입합니다.

    :param filename: 분석한 원본 이미지 파일명
    :param analysis_data: 분율 정보가 담긴 딕셔너리
    :return: 성공 시 True, 실패 시 False
    """
    conn = None
    try:
        conn = sqlite3.connect(settings.DB_FILE)
        cursor = conn.cursor()

        # SQL INSERT 문
        # 테이블 스키마와 순서가 일치해야 합니다.
        sql = '''
        INSERT INTO analysis_results (filename, alpha_al_fraction, si_inter_fraction, pore_fraction)
        VALUES (?, ?, ?, ?)
        '''
        
        # 데이터 튜플 생성
        data_tuple = (
            os.path.basename(filename),
            analysis_data.get('alpha_al_fraction', 0),
            analysis_data.get('si_inter_fraction', 0),
            analysis_data.get('pore_fraction', 0)
        )
        
        cursor.execute(sql, data_tuple)
        conn.commit()
        # print(f"DB에 결과 저장 완료: {os.path.basename(filename)}")
        return True

    except sqlite3.Error as e:
        print(f"데이터베이스 오류 발생: {e}")
        return False
    finally:
        if conn:
            conn.close()