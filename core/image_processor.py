import cv2
import numpy as np
import math
from config import settings


def normalizeImage(gray_image, blur_kernel=None):
    """
    CLAHE(선택)로 사진 간 조명/에칭 차이를 보정한 뒤 노이즈 제거.
    settings.USE_CLAHE = False 로 끄면 기존 방식(medianBlur만) 사용.

    blur_kernel: medianBlur 커널 크기 (홀수). 클수록 작은 잔점/노이즈 제거.
    """
    if settings.USE_CLAHE:
        clahe = cv2.createCLAHE(
            clipLimit=settings.CLAHE_CLIP_LIMIT,
            tileGridSize=settings.CLAHE_TILE_SIZE
        )
        img = clahe.apply(gray_image)
    else:
        img = gray_image

    k = blur_kernel if blur_kernel is not None else settings.MEDIAN_BLUR_KERNEL
    k = k if k % 2 == 1 else k + 1  # 홀수 강제
    return cv2.medianBlur(img, k)


def analyzeFeatures(mask, min_area_filter, phase_label):
    """
    마스크에서 개별 입자를 분리하고, 각 입자의 형상 및 크기 데이터를 반환.

    Returns:
        features (list of dict): 입자별 상세 데이터
        valid_mask (ndarray): 노이즈 제거된 최종 마스크
    """
    features = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    valid_mask = np.zeros_like(mask)
    pixel_area = settings.MICRON_PER_PIXEL ** 2  # μm²/pixel

    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]

        if area_px < min_area_filter:
            continue

        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        roi_labels = labels[y:y+h, x:x+w]
        single_obj_mask_roi = (roi_labels == i).astype(np.uint8)

        contours, _ = cv2.findContours(single_obj_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = contours[0]
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * math.pi * area_px) / (perimeter ** 2) if perimeter > 0 else 0

        if len(cnt) >= 5:
            rect = cv2.minAreaRect(cnt)
            (rw, rh) = rect[1]
            aspect_ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 1
        else:
            aspect_ratio = 1.0

        valid_mask[y:y+h, x:x+w] = cv2.bitwise_or(
            valid_mask[y:y+h, x:x+w], single_obj_mask_roi * 255
        )

        # ECD(등가원직경): 면적이 같은 원의 지름 (μm)
        ecd_um = 2 * math.sqrt(area_px / math.pi) * settings.MICRON_PER_PIXEL

        features.append({
            'phase': phase_label,
            'area_pixels': int(area_px),
            'area_um2': round(area_px * pixel_area, 4),
            'ecd_um': round(ecd_um, 4),
            'circularity': round(circularity, 4),
            'aspect_ratio': round(aspect_ratio, 4),
            'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
            'roi_mask': single_obj_mask_roi
        })

    return features, valid_mask


def filterSiByShape(si_candidate_mask, si_rules):
    """
    형상 필터(원형도/종횡비)로 Si 후보 마스크에서 최종 Si만 추출.
    윤곽점이 5개 미만인 작은 입자는 종횡비 측정 불가 → Si로 분류
    (측정 불가 입자를 IM으로 쌓이게 하지 않음)
    """
    final_si_mask = np.zeros_like(si_candidate_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(si_candidate_mask, connectivity=8)

    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]

        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        roi_labels = labels[y:y+h, x:x+w]
        single_obj_mask_roi = (roi_labels == i).astype(np.uint8)

        contours, _ = cv2.findContours(single_obj_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = contours[0]
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * math.pi * area_px) / (perimeter ** 2) if perimeter > 0 else 0

        if len(cnt) >= 5:
            rect = cv2.minAreaRect(cnt)
            (rw, rh) = rect[1]
            aspect_ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 1
        else:
            # 윤곽점 부족 → 종횡비 측정 불가: Si 기준값으로 설정해 Si로 분류
            aspect_ratio = si_rules['min_aspect_ratio']

        if (area_px >= si_rules['min_area'] and
                circularity <= si_rules['max_circularity'] and
                aspect_ratio >= si_rules['min_aspect_ratio']):
            final_si_mask[y:y+h, x:x+w] = cv2.bitwise_or(
                final_si_mask[y:y+h, x:x+w], single_obj_mask_roi * 255
            )

    return final_si_mask


def _locallyDarkMask(gray_image, kernel_size, min_diff):
    """
    주변 평균보다 min_diff 이상 어두운 픽셀만 남기는 마스크.
    Si/IM처럼 알파-Al 매트릭스 안에 있는 어두운 입자를 검출하는 데 사용.

    kernel_size: 주변 평균을 구할 영역 크기(홀수 강제). 클수록 넓은 범위와 비교.
    min_diff:    픽셀이 주변보다 이 값 이상 어두워야 마스크에 포함.
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    local_mean = cv2.GaussianBlur(gray_image.astype(np.float32), (k, k), 0)
    diff = local_mean - gray_image.astype(np.float32)  # 양수 = 주변보다 어두움
    return (diff >= min_diff).astype(np.uint8) * 255


def detectScaleBarMask(gray_image):
    """
    이미지 하단의 흰색 스케일 바 박스를 자동 감지하여 마스크 반환.
    감지된 영역은 분석에서 제외됨.
    CLAHE 전 원본 gray 이미지에서 실행해야 신뢰도 높음.
    """
    h, w = gray_image.shape
    search_top = int(h * 0.7)
    roi = gray_image[search_top:, :]

    # 매우 밝은 픽셀(흰색 박스 배경) 추출
    _, bright = cv2.threshold(roi, 245, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    scale_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, num_labels):
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if bh == 0:
            continue
        aspect = bw / bh
        fill   = area / (bw * bh)
        # 스케일 바 박스: 가로가 세로보다 길고, 내부가 밝게 채워진 직사각형
        if aspect > 1.5 and fill > 0.65 and area > 800:
            scale_mask[search_top + by: search_top + by + bh, bx: bx + bw] = 255

    return scale_mask


def _applyWatershed(mask, dist_ratio=0.4):
    """
    Distance transform + Watershed으로 서로 붙어있는 입자를 분리.

    원리:
      1. 각 픽셀에서 가장 가까운 배경(0)까지의 거리를 계산 (거리 변환)
      2. 거리가 충분히 큰 픽셀 = 입자 내부 중심 → 전경 마커로 지정
      3. 여러 마커 사이의 경계를 물이 흘러내려 만나는 지점처럼 계산 → 경계선

    dist_ratio: 거리 최댓값의 몇 배 이상이어야 전경 마커로 인정할지 (0.1~0.9)
      낮으면 많이 나눔(과분할 위험), 높으면 덜 나눔

    Returns:
        separated_mask: 경계가 제거된 분리 마스크 (analyzeFeatures 입력용)
        boundary_mask:  경계선만 담은 마스크 (시각화 오버레이용)
    """
    if not np.any(mask):
        return mask.copy(), np.zeros_like(mask)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return mask.copy(), np.zeros_like(mask)

    # 확실한 전경: 거리가 충분히 큰 픽셀 (각 입자의 중심 부근)
    _, sure_fg = cv2.threshold(dist, dist_ratio * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    if not np.any(sure_fg):
        return mask.copy(), np.zeros_like(mask)

    # 확실한 배경: 마스크를 약간 팽창
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(mask, kernel, iterations=2)

    # 불확실 영역 (배경도 전경도 아님)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 마커 생성: 전경 라벨 1~N, 배경 1, 불확실 0
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers.astype(np.int32) + 1
    markers[unknown == 255] = 0

    # Watershed 실행 (3채널 이미지 필요)
    img_3ch = cv2.merge([mask, mask, mask])
    cv2.watershed(img_3ch, markers)

    # 경계선: watershed가 -1로 표시한 픽셀
    boundary_mask = np.zeros_like(mask)
    boundary_mask[markers == -1] = 255

    # 분리 마스크: 원본에서 경계선 제거
    separated_mask = mask.copy()
    separated_mask[markers == -1] = 0

    return separated_mask, boundary_mask


def _applyErosionSeparation(mask, radius):
    """
    침식(Erosion) 기반 입자 분리.

    1. 마스크를 radius px 침식 → 좁은 목(neck)이 끊어짐
    2. 끊어진 각 덩어리를 씨앗으로 사용
    3. Watershed로 원래 마스크 크기까지 복원
       → 두 씨앗 사이 경계가 분리선이 됨

    Watershed와의 차이:
      - Watershed: 거리 봉우리 개수로 나눔 → 길쭉한 입자도 잘릴 수 있음
      - 침식 분리: 목(연결부) 너비로 나눔 → 길쭉해도 두꺼우면 안 잘림
        (radius × 2 px 보다 좁은 연결부만 끊어짐)
    """
    if not np.any(mask):
        return mask.copy(), np.zeros_like(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    eroded = cv2.erode(mask, kernel)

    if not np.any(eroded):
        return mask.copy(), np.zeros_like(mask)

    # 침식 후 연결 성분 → 각각 독립 씨앗
    _, markers = cv2.connectedComponents(eroded)
    markers = markers.astype(np.int32) + 1  # 배경=1, 각 성분=2+

    # 침식으로 깎인 영역(unknown): 씨앗도 배경도 아닌 영역 → 0으로 표시
    unknown = cv2.subtract(mask, eroded)
    markers[unknown == 255] = 0
    markers[mask == 0] = 1   # 마스크 밖 = 확실한 배경

    # Watershed로 씨앗을 원래 마스크 크기까지 팽창 → 경계선 생성
    img_3ch = cv2.merge([mask, mask, mask])
    cv2.watershed(img_3ch, markers)

    boundary_mask = np.zeros_like(mask)
    boundary_mask[markers == -1] = 255

    separated_mask = mask.copy()
    separated_mask[markers == -1] = 0

    return separated_mask, boundary_mask


def _segmentByGMM(normalized_image, scale_mask=None):
    """
    4-component GMM으로 픽셀을 4상(기공/Si/IM/Alpha-Al)으로 분류.
    평균 밝기 기준 오름차순: 0=pore, 1=si, 2=intermetallic, 3=alpha_al

    Returns:
        mask_pore, mask_si, mask_im, mask_alpha  (각 255/0 이진 마스크)
    """
    from sklearn.mixture import GaussianMixture

    h, w = normalized_image.shape

    # 유효 픽셀 추출 (스케일 바 제외)
    if scale_mask is not None:
        valid_flat = normalized_image[scale_mask == 0].reshape(-1, 1).astype(np.float32)
    else:
        valid_flat = normalized_image.reshape(-1, 1).astype(np.float32)

    # 속도 위해 최대 200k 픽셀 무작위 샘플링
    rng = np.random.RandomState(42)
    if len(valid_flat) > 200000:
        idx = rng.choice(len(valid_flat), 200000, replace=False)
        sample = valid_flat[idx]
    else:
        sample = valid_flat

    gmm = GaussianMixture(n_components=4, covariance_type='full',
                          n_init=5, random_state=42, max_iter=300)
    gmm.fit(sample)

    # 전체 이미지 예측
    labels = gmm.predict(normalized_image.reshape(-1, 1).astype(np.float32))
    labels = labels.reshape(h, w)

    # 평균 밝기 오름차순으로 phase 할당: 어두운 순 → pore, si, im, alpha_al
    order = np.argsort(gmm.means_.flatten())  # order[i] = i번째로 어두운 component 번호
    remap = np.empty(4, dtype=np.uint8)
    for phase_idx, comp_idx in enumerate(order):
        remap[comp_idx] = phase_idx
    phase_img = remap[labels]  # 0=pore, 1=si, 2=im, 3=alpha_al

    means_sorted = gmm.means_.flatten()[order]
    print(f"[GMM] pore={means_sorted[0]:.1f}  si={means_sorted[1]:.1f}  "
          f"im={means_sorted[2]:.1f}  alpha_al={means_sorted[3]:.1f}")

    def _make(idx):
        m = np.where(phase_img == idx, np.uint8(255), np.uint8(0))
        if scale_mask is not None:
            m[scale_mask > 0] = 0
        return m

    return _make(0), _make(1), _make(2), _make(3)


def segmentAndClassify(normalized_image, thresholds, params, original_image, scale_mask=None):
    total_pixels = normalized_image.size
    use_gmm = params.get('use_gmm', False)

    if use_gmm:
        # --- GMM 자동 분류: 임계값 대신 GMM이 초기 마스크 생성 ---
        mask_pore, mask_lower, mask_upper, mask_alpha = _segmentByGMM(normalized_image, scale_mask)
    else:
        # --- 수동 임계값 분류 ---
        # 1. 기공 / 알파-Al: 절대 밝기로 분리
        mask_pore  = cv2.inRange(normalized_image, 0, thresholds['pore'][1])
        mask_alpha = cv2.inRange(normalized_image, thresholds['alpha_al'][0], 255)

        if scale_mask is not None:
            mask_pore [scale_mask > 0] = 0
            mask_alpha[scale_mask > 0] = 0

        # 2. 2차 상 분리: Si 상한을 기준으로 하단/상단 두 구간으로 나눔
        dark_lo  = thresholds['si'][0]
        si_upper = thresholds['si'][1]
        im_lower = thresholds['intermetallic'][0]
        dark_hi  = thresholds['intermetallic'][1]

        def _make_mask(lo, hi):
            m = cv2.inRange(normalized_image, lo, hi)
            m[mask_pore > 0] = 0
            if scale_mask is not None:
                m[scale_mask > 0] = 0
            return m

        mask_lower = _make_mask(dark_lo, si_upper)
        mask_upper = _make_mask(im_lower, dark_hi)

    # 3. 하단 구간(Si 후보)에만 로컬 대비 필터 적용
    lc = params.get('local_contrast', settings.LOCAL_CONTRAST)
    if lc['min_diff'] > 0:
        locally_dark = _locallyDarkMask(normalized_image, lc['kernel_size'], lc['min_diff'])
        # 탈락 픽셀(주변과 밝기 유사) → Al 내부로 재분류
        mask_rejected = cv2.bitwise_and(mask_lower, cv2.bitwise_not(locally_dark))
        mask_lower    = cv2.bitwise_and(mask_lower, locally_dark)
        mask_alpha    = cv2.bitwise_or(mask_alpha, mask_rejected)

    # 4. 알파-Al에서 기공 / Si / IM 픽셀 제거
    mask_alpha[mask_pore  > 0] = 0
    mask_alpha[mask_lower > 0] = 0
    mask_alpha[mask_upper > 0] = 0

    # 5. 밝기 구간으로 직접 분리
    #    하단 구간 → Eutectic Si
    #    상단 구간 → Intermetallic
    ws_phases  = params.get('watershed_phases',  settings.WATERSHED_PHASES)
    ws_ratio   = params.get('watershed_dist_ratio', settings.WATERSHED_DIST_RATIO)
    er_phases  = params.get('erosion_phases',    settings.EROSION_PHASES)
    er_radius  = params.get('erosion_radius',    settings.EROSION_RADIUS)

    # 상별 경계 마스크 (watershed + 침식 합산, 시각화용)
    all_sep_boundaries = np.zeros_like(mask_lower)

    def _separate(mask, phase):
        """watershed / 침식 분리를 순서대로 적용하고 경계 누적"""
        nonlocal all_sep_boundaries
        if ws_phases.get(phase):
            mask, b = _applyWatershed(mask, ws_ratio)
            all_sep_boundaries = cv2.bitwise_or(all_sep_boundaries, b)
        if er_phases.get(phase):
            mask, b = _applyErosionSeparation(mask, er_radius)
            all_sep_boundaries = cv2.bitwise_or(all_sep_boundaries, b)
        return mask

    mask_lower = _separate(mask_lower, 'si')

    si_feats, final_si_mask = analyzeFeatures(mask_lower, params['si_rules']['min_area'], 'eutectic_si')

    # Si 경계 근처 IM 후보 제거: Si-Al 그레이스케일 그래디언트가 IM 밝기 구간에 걸리는 현상 방지
    excl_r = params.get('si_exclusion_radius', settings.SI_EXCLUSION_RADIUS)
    if excl_r > 0 and np.any(final_si_mask > 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * excl_r + 1, 2 * excl_r + 1))
        si_dilated = cv2.dilate(final_si_mask, kernel)
        mask_upper = cv2.bitwise_and(mask_upper, cv2.bitwise_not(si_dilated))

    # Al 경계 근처 IM 후보 제거: Al-IM 그레이스케일 그래디언트가 IM 밝기 구간에 걸리는 현상 방지
    # 원시 Alpha-Al 마스크(밝기 기준만)를 팽창 → 근접 IM 픽셀 제거
    al_excl_r = params.get('al_exclusion_radius', settings.AL_EXCLUSION_RADIUS)
    if al_excl_r > 0:
        raw_alpha = cv2.inRange(normalized_image, thresholds['alpha_al'][0], 255)
        if scale_mask is not None:
            raw_alpha[scale_mask > 0] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * al_excl_r + 1, 2 * al_excl_r + 1))
        al_dilated = cv2.dilate(raw_alpha, kernel)
        mask_upper = cv2.bitwise_and(mask_upper, cv2.bitwise_not(al_dilated))

    mask_upper = _separate(mask_upper, 'intermetallic')

    inter_feats, final_inter_mask = analyzeFeatures(mask_upper, params['min_areas']['intermetallic'], 'intermetallic')

    # IM 최대 면적 필터: 너무 큰 덩어리 → Al 덴드라이트로 판단, Alpha-Al 재분류
    max_im_area = params.get('max_areas', settings.MAX_AREAS).get('intermetallic', 0)
    if max_im_area > 0:
        kept, large_mask = [], np.zeros_like(final_inter_mask)
        for feat in inter_feats:
            if feat['area_pixels'] > max_im_area:
                x, y, w, h = feat['bbox_x'], feat['bbox_y'], feat['bbox_w'], feat['bbox_h']
                large_mask[y:y+h, x:x+w] = cv2.bitwise_or(
                    large_mask[y:y+h, x:x+w], feat['roi_mask'] * 255
                )
                final_inter_mask[y:y+h, x:x+w] = cv2.bitwise_and(
                    final_inter_mask[y:y+h, x:x+w],
                    cv2.bitwise_not(feat['roi_mask'] * 255)
                )
            else:
                kept.append(feat)
        inter_feats = kept
        mask_alpha = cv2.bitwise_or(mask_alpha, large_mask)

    mask_pore  = _separate(mask_pore,  'pore')
    mask_alpha = _separate(mask_alpha, 'alpha_al')

    pore_feats,  final_pore_mask  = analyzeFeatures(mask_pore,  params['min_areas']['pore'],    'pore')
    alpha_feats, final_alpha_mask = analyzeFeatures(mask_alpha, params['min_areas']['alpha_al'], 'alpha_al')

    # 4. 분율 계산
    data = {
        'pore_fraction':          (np.count_nonzero(final_pore_mask)  / total_pixels) * 100,
        'eutectic_si_fraction':   (np.count_nonzero(final_si_mask)    / total_pixels) * 100,
        'intermetallic_fraction': (np.count_nonzero(final_inter_mask) / total_pixels) * 100,
        'alpha_al_fraction':      (np.count_nonzero(final_alpha_mask) / total_pixels) * 100,
    }

    # 5. 입자별 상세 데이터 (roi_mask 제외하고 저장용으로 전달)
    def _strip_roi(feats):
        return [{k: v for k, v in f.items() if k != 'roi_mask'} for f in feats]

    data['details'] = (
        _strip_roi(pore_feats) +
        _strip_roi(si_feats) +
        _strip_roi(inter_feats) +
        _strip_roi(alpha_feats)
    )

    # 6. 시각화 오버레이
    seg_map = np.zeros(normalized_image.shape, dtype=np.uint8)
    seg_map[final_pore_mask  > 0] = 1
    seg_map[final_si_mask    > 0] = 2
    seg_map[final_inter_mask > 0] = 3
    seg_map[final_alpha_mask > 0] = 4

    color_map = np.zeros((*normalized_image.shape, 3), dtype=np.uint8)
    color_map[seg_map == 1] = [255, 0,   0  ]  # 기공       = 파랑 (BGR)
    color_map[seg_map == 2] = [0,   255, 0  ]  # Eutectic Si = 초록
    color_map[seg_map == 3] = [0,   255, 255]  # Intermetallic = 노랑
    color_map[seg_map == 4] = [0,   0,   255]  # Alpha-Al   = 빨강

    result_image = cv2.addWeighted(color_map, 0.5, original_image, 0.5, 0)

    # 분리 경계선 흰색으로 오버레이 (watershed + 침식 분리 모두 포함)
    if np.any(all_sep_boundaries > 0):
        result_image[all_sep_boundaries > 0] = [255, 255, 255]

    return result_image, data


def loadGrayImage(image_path, blur_kernel=None):
    """원본 이미지를 정규화된 그레이스케일로 반환 (GUI 호버용)"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return normalizeImage(gray, blur_kernel)


def analyzeImage(image_path, thresholds, all_params):
    original_image = cv2.imread(image_path)
    if original_image is None:
        return None, None
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    scale_mask = detectScaleBarMask(gray)
    blur_k = all_params.get('median_blur_kernel', settings.MEDIAN_BLUR_KERNEL)
    norm = normalizeImage(gray, blur_k)
    return segmentAndClassify(norm, thresholds, all_params, original_image, scale_mask)
