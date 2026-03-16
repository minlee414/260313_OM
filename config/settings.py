import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(BASE_DIR, 'db', 'analysis_results.db')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_outputs')

# 밝기 임계값 (Si가 IM보다 어두우므로 범위 순서를 변경함)
PHASE_THRESHOLDS = {
    'pore': (0, 100),
    'si': (101, 140),             # Si가 더 어두움
    'intermetallic': (141, 180), # IM이 Si보다 밝음
    'alpha_al': (181, 255)
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

MICRON_PER_PIXEL = 0.14  # μm/pixel (Axioskop 2 Mat, Axiocam 820)

# CLAHE 파라미터: 사진 간 조명/에칭 차이 보정용 (False로 끄면 기존 방식)
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)