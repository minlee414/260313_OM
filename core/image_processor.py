import cv2
import numpy as np
from config import settings

def normalizeImage(gray_image):
    # 원본 명암비를 보존하면서 노이즈만 제거
    return cv2.medianBlur(gray_image, 3)

def analyzeFeatures(mask, min_area_filter):
    features = []
    # 픽셀 단위로 정확하게 영역(Blob) 분리
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 노이즈가 제거된 '순수 픽셀 마스크'를 담을 빈 캔버스
    valid_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels): # 0번은 배경이므로 제외
        area = stats[i, cv2.CC_STAT_AREA]
        
        # 최소 면적(노이즈) 필터링
        if area < min_area_filter:
            continue
            
        # ★ [속도 최적화 1] 전체 이미지 대신, 입자가 포함된 '최소 사각 영역(ROI)'만 잘라냄
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        roi_labels = labels[y:y+h, x:x+w]
        
        # 잘라낸 작은 ROI 안에서만 단일 입자 마스크 생성 (연산 속도 수십 배 향상)
        single_obj_mask_roi = (roi_labels == i).astype(np.uint8)
        
        # 잘라낸 작은 ROI 안에서만 윤곽선 탐색
        contours, _ = cv2.findContours(single_obj_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue

        cnt = contours[0]
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        if len(cnt) >= 5:
            rect = cv2.minAreaRect(cnt)
            (rw, rh) = rect[1]
            aspect_ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 1
        else:
            aspect_ratio = 1

        # ★ [속도 최적화 2] 필터를 통과한 입자의 '작은 마스크'만 최종 캔버스의 해당 위치에 합침
        valid_mask[y:y+h, x:x+w] = cv2.bitwise_or(valid_mask[y:y+h, x:x+w], single_obj_mask_roi * 255)

        features.append({
            'area_pixels': int(area),
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'bbox': (x, y, w, h),
            'roi_mask': single_obj_mask_roi # roi 마스크 전달
        })
    return features, valid_mask

def filterSiByShape(si_candidate_mask, si_rules):
    final_si_mask = np.zeros_like(si_candidate_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(si_candidate_mask, connectivity=8)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        roi_labels = labels[y:y+h, x:x+w]
        single_obj_mask_roi = (roi_labels == i).astype(np.uint8)

        contours, _ = cv2.findContours(single_obj_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
            
        cnt = contours[0]
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        if len(cnt) >= 5:
            rect = cv2.minAreaRect(cnt)
            (rw, rh) = rect[1]
            aspect_ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 1
        else:
            aspect_ratio = 1

        if (area >= si_rules['min_area'] and
            circularity <= si_rules['max_circularity'] and
            aspect_ratio >= si_rules['min_aspect_ratio']):
            final_si_mask[y:y+h, x:x+w] = cv2.bitwise_or(final_si_mask[y:y+h, x:x+w], single_obj_mask_roi * 255)
            
    return final_si_mask

def segmentAndClassify(normalized_image, thresholds, params, original_image):
    # 1. 밝기 기반 마스크 생성 및 강제 배타 처리
    mask_pore = cv2.inRange(normalized_image, thresholds['pore'][0], thresholds['pore'][1])
    mask_si = cv2.inRange(normalized_image, thresholds['si'][0], thresholds['si'][1])
    mask_inter = cv2.inRange(normalized_image, thresholds['intermetallic'][0], thresholds['intermetallic'][1])
    mask_alpha = cv2.inRange(normalized_image, thresholds['alpha_al'][0], thresholds['alpha_al'][1])
    
    mask_si[mask_pore > 0] = 0
    mask_inter[mask_pore > 0] = 0; mask_inter[mask_si > 0] = 0
    mask_alpha[mask_pore > 0] = 0; mask_alpha[mask_si > 0] = 0; mask_alpha[mask_inter > 0] = 0
    
    # 2. 최소 면적 필터링
    _, final_pore_mask = analyzeFeatures(mask_pore, params['min_areas']['pore'])
    _, final_inter_mask = analyzeFeatures(mask_inter, params['min_areas']['intermetallic'])
    _, final_alpha_mask = analyzeFeatures(mask_alpha, params['min_areas']['alpha_al'])
    
    # 3. Si는 최소 면적 필터링 후, 형상으로 2차 정밀 필터링
    _, si_cand_mask = analyzeFeatures(mask_si, params['si_rules']['min_area'])
    final_si_mask = filterSiByShape(si_cand_mask, params['si_rules'])

    # 4. 분율 계산
    total = normalized_image.size
    data = {
        'pore_fraction': (np.count_nonzero(final_pore_mask) / total) * 100,
        'eutectic_si_fraction': (np.count_nonzero(final_si_mask) / total) * 100,
        'intermetallic_fraction': (np.count_nonzero(final_inter_mask) / total) * 100,
        'alpha_al_fraction': (np.count_nonzero(final_alpha_mask) / total) * 100
    }

    # 5. 통합 분할 지도를 이용한 시각화
    seg_map = np.zeros(normalized_image.shape, dtype=np.uint8)
    seg_map[final_pore_mask > 0] = 1
    seg_map[final_si_mask > 0] = 2
    seg_map[final_inter_mask > 0] = 3
    seg_map[final_alpha_mask > 0] = 4
    
    color_map = np.zeros((normalized_image.shape[0], normalized_image.shape[1], 3), dtype=np.uint8)
    color_map[seg_map == 1] = [255, 0, 0]    # 기공 = 파랑
    color_map[seg_map == 2] = [0, 255, 0]    # Si = 초록
    color_map[seg_map == 3] = [0, 255, 255]  # Inter = 노랑
    color_map[seg_map == 4] = [0, 0, 255]    # Alpha = 빨강

    result_image = cv2.addWeighted(color_map, 0.5, original_image, 0.5, 0)
    
    return result_image, data

def analyzeImage(image_path, thresholds, all_params):
    original_image = cv2.imread(image_path)
    if original_image is None: return None, None
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    norm = normalizeImage(gray)
    return segmentAndClassify(norm, thresholds, all_params, original_image)