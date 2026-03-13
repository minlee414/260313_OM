# etched_om_microstructure_analyzer

에칭된 OM 이미지에서 알루미늄 주조재 미세조직을 자동 분할하고 정량화하는 Python 프로그램입니다.

## 대상 클래스
- alpha_al_particle
- eutectic_si
- porosity
- unclassified

## 개발 환경
- Windows
- Python 3.10+

## 설치

### 1. 가상환경 생성
```bash
python -m venv .venv

etched_om_microstructure_analyzer/
├─ main.py
├─ requirements.txt
├─ README.md
├─ run_batch.bat
├─ config/
│  ├─ settings.py
│  └─ default_config.yaml
├─ core/
│  ├─ __init__.py
│  ├─ batch_runner.py
│  ├─ file_parser.py
│  ├─ image_loader.py
│  ├─ preprocess.py
│  ├─ quality_metrics.py
│  ├─ segmentation.py
│  ├─ quantification.py
│  ├─ overlay.py
│  └─ tile_processor.py
└─ common/
   ├─ __init__.py
   ├─ logger.py
   ├─ io_utils.py
   └─ utils.py

   