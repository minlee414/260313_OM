import cv2
import numpy as np
import math
from config import settings


def normalizeImage(gray_image):
    """
    CLAHE(선택)로 사진 간 조명/에칭 차이를 보정한 뒤 노이즈 제거.
    settings.USE_CLAHE = False 로 끄면 기존 방식(medianBlur만) 사용.

    CLAHE(Contrast Limited Adaptive Histogram Equalization):
    - 전체 히스토그램이 아닌 작은 타일 단위로 대비를 높여,
      한 장 안에서 밝기가 불균일해도 어두운/밝은 영역 모두 균일하게 처리.
    - 사진마다 전체 밝기가 달라도 임계값의 상대적 의미가 유지됨.
    - 에칭 정도 차이로 인한 전체 밝기 이동(shift)에 어느 정도 강인해짐.
    """
    if settings.USE_CLAHE:
        clahe = cv2.createCLAHE(
            clipLimit=settings.CLAHE_CLIP_LIMIT,
            tileGridSize=settings.CLAHE_TILE_SIZE
        )
        img = clahe.apply(gray_image)
    else:
        img = gray_image
    return cv2.medianBlur(img, 3)


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
            aspect_ratio = 1.0

        if (area_px >= si_rules['min_area'] and
                circularity <= si_rules['max_circularity'] and
                aspect_ratio >= si_rules['min_aspect_ratio']):
            final_si_mask[y:y+h, x:x+w] = cv2.bitwise_or(
                final_si_mask[y:y+h, x:x+w], single_obj_mask_roi * 255
            )

    return final_si_mask


def segmentAndClassify(normalized_image, thresholds, params, original_image):
    total_pixels = normalized_image.size

    # 1. 밝기 기반 마스크 생성 (우선순위: pore > si > intermetallic > alpha_al)
    mask_pore  = cv2.inRange(normalized_image, thresholds['pore'][0],         thresholds['pore'][1])
    mask_si    = cv2.inRange(normalized_image, thresholds['si'][0],           thresholds['si'][1])
    mask_inter = cv2.inRange(normalized_image, thresholds['intermetallic'][0], thresholds['intermetallic'][1])
    mask_alpha = cv2.inRange(normalized_image, thresholds['alpha_al'][0],      255)

    mask_si[mask_pore > 0] = 0
    mask_inter[mask_pore > 0] = 0
    mask_inter[mask_si > 0] = 0
    mask_alpha[mask_pore > 0] = 0
    mask_alpha[mask_si > 0] = 0
    mask_alpha[mask_inter > 0] = 0

    # 2. 입자 분석 (ECD 포함)
    pore_feats,  final_pore_mask  = analyzeFeatures(mask_pore,  params['min_areas']['pore'],         'pore')
    inter_feats, final_inter_mask = analyzeFeatures(mask_inter, params['min_areas']['intermetallic'], 'intermetallic')
    alpha_feats, final_alpha_mask = analyzeFeatures(mask_alpha, params['min_areas']['alpha_al'],      'alpha_al')

    # 3. Si: 면적 필터 후 형상 2차 필터 (eutectic Si 특징: 저원형도 + 고종횡비)
    si_feats_cand, si_cand_mask = analyzeFeatures(mask_si, params['si_rules']['min_area'], 'eutectic_si')
    final_si_mask = filterSiByShape(si_cand_mask, params['si_rules'])

    # filterSiByShape 통과한 입자만 si_feats로 남김
    # (valid_mask에 남은 픽셀과 대조해 재필터)
    si_feats = []
    for feat in si_feats_cand:
        x, y, w, h = feat['bbox_x'], feat['bbox_y'], feat['bbox_w'], feat['bbox_h']
        roi_si_final = final_si_mask[y:y+h, x:x+w]
        if np.any(roi_si_final > 0):
            si_feats.append(feat)

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
    return result_image, data


def analyzeImage(image_path, thresholds, all_params):
    original_image = cv2.imread(image_path)
    if original_image is None:
        return None, None
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    norm = normalizeImage(gray)
    return segmentAndClassify(norm, thresholds, all_params, original_image)
