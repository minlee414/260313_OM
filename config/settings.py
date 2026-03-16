import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(BASE_DIR, 'db', 'analysis_results.db')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_outputs')

# 밝기 임계값 (Si가 IM보다 어두우므로 범위 순서를 변경함)
PHASE_THRESHOLDS = {
    'pore': (0, 40),
    'si': (41, 120),             # Si가 더 어두움
    'intermetallic': (121, 180), # IM이 Si보다 밝음
    'alpha_al': (181, 255)
}

MIN_AREAS = {
    'pore': 20,
    'alpha_al': 100,
    'intermetallic': 50
}

SI_RULES = {
    'min_area': 50,
}

MICRON_PER_PIXEL = 0.14  # μm/pixel (Axioskop 2 Mat, Axiocam 820)

# 전처리 블러: 분석 전 노이즈/잔점 제거 (홀수, 클수록 더 많이 뭉갬)
MEDIAN_BLUR_KERNEL = 3

# 로컬 대비 필터: Si/IM이 주변 알파-Al보다 얼마나 어두운지 조건
# kernel_size: 주변 평균을 구하는 영역 크기(px, 홀수), 클수록 넓은 범위와 비교
# min_diff: 주변 평균보다 이 값 이상 어두워야 Si/IM으로 인정
LOCAL_CONTRAST = {
    'kernel_size': 51,
    'min_diff': 15,
}

# IM 최대 면적: 이보다 큰 2차 상 후보 → Al 덴드라이트 내부로 판단하여 Alpha-Al 재분류
# 0 = 끔 (제한 없음)
MAX_AREAS = {
    'intermetallic': 0,  # 0 = 끔. 필요시 큰 값(예: 50000)으로 설정
}

# Si 경계 근처 IM 제거 반경 (px): Si-Al 그레이스케일 그래디언트를 IM으로 오분류하는 것을 방지
# Si 최종 마스크를 이 반경만큼 팽창시켜 근접 IM 픽셀을 Alpha-Al로 재분류. 0 = 끔
SI_EXCLUSION_RADIUS = 3
AL_EXCLUSION_RADIUS = 3  # Al 경계 근처 IM 제거 반경 (Al-IM 그래디언트 방지), 0=끔

# Watershed 입자 분리: 서로 붙어있는 입자를 경계선으로 나눔 (상별 개별 설정)
# DIST_RATIO: 거리 변환 최댓값 대비 전경 마커 임계 비율 (0.1~0.9)
#   낮을수록 더 많이 나눔(과분할 위험), 높을수록 덜 나눔(분리 부족 위험)
WATERSHED_PHASES = {
    'si':            False,
    'intermetallic': False,
    'alpha_al':      False,
    'pore':          False,
}
WATERSHED_DIST_RATIO = 0.4

# 침식 분리: radius×2 px 미만의 좁은 목(neck)을 끊어 입자를 분리
# 길쭉한 입자는 일정하게 두꺼우면 끊어지지 않음
EROSION_PHASES = {
    'si':            False,
    'intermetallic': False,
    'alpha_al':      False,
    'pore':          False,
}
EROSION_RADIUS = 5  # px (목 너비 < radius×2 px 이면 분리)

# CLAHE 파라미터: 사진 간 조명/에칭 차이 보정용 (False로 끄면 기존 방식)
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 1.0
CLAHE_TILE_SIZE = (8, 8)