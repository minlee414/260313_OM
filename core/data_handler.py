import os
import csv
from config import settings
from db import database

# --- Helper Functions ---

def _saveSummaryCsv(image_filename, analysis_data):
    """요약 CSV 파일에 분석 결과를 한 줄 추가합니다."""
    summary_file_path = os.path.join(settings.OUTPUT_DIR, 'summary_analysis.csv')
    
    # 파일이 존재하지 않으면 헤더를 씁니다.
    write_header = not os.path.exists(summary_file_path)
    
    with open(summary_file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['filename', 'alpha_al_fraction', 'si_inter_fraction', 'pore_fraction'])
        
        writer.writerow([
            os.path.basename(image_filename),
            f"{analysis_data.get('alpha_al_fraction', 0):.4f}",
            f"{analysis_data.get('si_inter_fraction', 0):.4f}",
            f"{analysis_data.get('pore_fraction', 0):.4f}"
        ])

def _saveDetailedCsv(image_filename, analysis_data):
    """
    이미지별 상세 분석 데이터를 별도의 CSV 파일로 저장합니다.
    (이 기능을 위해서는 image_processor가 상세 데이터를 반환해야 합니다)
    """
    # 상세 데이터가 없는 경우, 이 함수는 실행되지 않습니다.
    if 'details' not in analysis_data or not analysis_data['details']:
        return

    # 출력 파일명 생성 (예: 'image_01.jpg' -> 'image_01_details.csv')
    base_name = os.path.splitext(os.path.basename(image_filename))[0]
    output_filename = f"{base_name}_details.csv"
    detailed_file_path = os.path.join(settings.OUTPUT_DIR, output_filename)
    
    # 상세 데이터 (예: [{'id': 1, 'phase': 'pore', 'area': 50.2}, ...])
    detailed_results = analysis_data['details']
    
    with open(detailed_file_path, 'w', newline='', encoding='utf-8') as f:
        # 헤더는 상세 데이터의 첫 번째 항목의 key들을 사용
        if detailed_results:
            writer = csv.DictWriter(f, fieldnames=detailed_results[0].keys())
            writer.writeheader()
            writer.writerows(detailed_results)


# --- Main Public Function ---

def saveData(image_path, analysis_data):
    """
    분석 데이터를 모든 형식(요약CSV, 상세CSV, DB)으로 저장하는 메인 함수

    :param image_path: 분석한 원본 이미지의 전체 경로
    :param analysis_data: image_processor로부터 받은 분석 결과 딕셔너리
    """
    if not analysis_data:
        print("저장할 분석 데이터가 없습니다.")
        return

    try:
        # 1. 요약 CSV 파일에 결과 추가
        _saveSummaryCsv(image_path, analysis_data)
        
        # 2. 이미지별 상세 CSV 파일 생성
        # 참고: 이 기능이 동작하려면 image_processor.py가 'details' 키를 포함한
        # 상세 입자 정보를 analysis_data에 담아 반환해야 합니다.
        _saveDetailedCsv(image_path, analysis_data)

        # 3. 데이터베이스에 결과 삽입
        database.insertResult(image_path, analysis_data)
        
        print(f"'{os.path.basename(image_path)}'의 분석 결과 저장이 완료되었습니다.")

    except Exception as e:
        print(f"데이터 저장 중 오류 발생: {e}")