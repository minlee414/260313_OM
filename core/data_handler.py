import os
import csv
from config import settings
from db import database


def _saveSummaryCsv(image_filename, analysis_data):
    """요약 CSV: 이미지 1장당 1행 (면적 분율 + 입자 수 + ECD 통계)"""
    summary_file_path = os.path.join(settings.OUTPUT_DIR, 'summary_analysis.csv')
    write_header = not os.path.exists(summary_file_path)

    # 상별 ECD 통계 계산
    details = analysis_data.get('details', [])
    stats = _calcEcdStats(details)

    with open(summary_file_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'filename',
                'alpha_al_fraction_%', 'eutectic_si_fraction_%',
                'intermetallic_fraction_%', 'pore_fraction_%',
                'alpha_al_count',
                'alpha_al_ecd_mean_um', 'alpha_al_ecd_std_um',
                'alpha_al_ecd_min_um',  'alpha_al_ecd_max_um',
                'si_count', 'si_ecd_mean_um', 'si_ecd_std_um', 'si_ecd_min_um', 'si_ecd_max_um',
                'im_count', 'im_ecd_mean_um', 'im_ecd_std_um', 'im_ecd_min_um', 'im_ecd_max_um',
                'pore_count', 'pore_ecd_mean_um', 'pore_ecd_std_um', 'pore_ecd_min_um', 'pore_ecd_max_um',
            ])

        row = [
            os.path.basename(image_filename),
            f"{analysis_data.get('alpha_al_fraction', 0):.4f}",
            f"{analysis_data.get('eutectic_si_fraction', 0):.4f}",
            f"{analysis_data.get('intermetallic_fraction', 0):.4f}",
            f"{analysis_data.get('pore_fraction', 0):.4f}",
        ]
        for phase in ('alpha_al', 'eutectic_si', 'intermetallic', 'pore'):
            s = stats.get(phase, {})
            row += [
                s.get('count', 0),
                f"{s.get('mean', 0):.4f}",
                f"{s.get('std', 0):.4f}",
                f"{s.get('min', 0):.4f}",
                f"{s.get('max', 0):.4f}",
            ]
        writer.writerow(row)


def _saveDetailedCsv(image_filename, analysis_data):
    """상세 CSV: 이미지별 입자 1개당 1행 (phase, area, ECD, 원형도, 종횡비)"""
    details = analysis_data.get('details')
    if not details:
        return

    base_name = os.path.splitext(os.path.basename(image_filename))[0]
    output_path = os.path.join(settings.OUTPUT_DIR, f"{base_name}_particles.csv")

    fieldnames = ['phase', 'area_pixels', 'area_um2', 'ecd_um',
                  'circularity', 'aspect_ratio',
                  'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(details)


def _calcEcdStats(details):
    """phase별 ECD 기술통계 계산"""
    from collections import defaultdict
    import math

    grouped = defaultdict(list)
    for p in details:
        grouped[p['phase']].append(p['ecd_um'])

    result = {}
    for phase, ecds in grouped.items():
        n = len(ecds)
        if n == 0:
            continue
        mean = sum(ecds) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in ecds) / n) if n > 1 else 0
        result[phase] = {
            'count': n,
            'mean': mean,
            'std': std,
            'min': min(ecds),
            'max': max(ecds),
        }
    return result


def saveData(image_path, analysis_data):
    """분석 결과를 요약 CSV, 상세 CSV, DB에 저장"""
    if not analysis_data:
        print("저장할 분석 데이터가 없습니다.")
        return

    try:
        _saveSummaryCsv(image_path, analysis_data)
        _saveDetailedCsv(image_path, analysis_data)
        database.insertResult(image_path, analysis_data)
        print(f"'{os.path.basename(image_path)}' 저장 완료.")
    except Exception as e:
        print(f"데이터 저장 중 오류 발생: {e}")


def getEcdStats(analysis_data):
    """GUI 결과 표시용: 상별 ECD 통계 반환"""
    return _calcEcdStats(analysis_data.get('details', []))
