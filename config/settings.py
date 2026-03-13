import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(BASE_DIR, 'db', 'analysis_results.db')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_outputs')

# 밝기 임계값 (Si가 IM보다 어두우므로 범위 순서를 변경함)
PHASE_THRESHOLDS = {
    'pore': (0, 30),
    'si': (31, 70),             # Si가 더 어두움
    'intermetallic': (71, 120), # IM이 Si보다 밝음
    'alpha_al': (180, 255)
}

MIN_AREAS = {
    'pore': 20,
    'alpha_al': 100,
    'intermetallic': 50
}

SI_RULES = {
    'min_area': 50,
    'max_circularity': 0.4,
    'min_aspect_ratio': 3.0
}

MICRON_PER_PIXEL = 0.14