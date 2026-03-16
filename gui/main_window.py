import sys, os, cv2, numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from core.image_processor import analyzeImage
from core.data_handler import saveData, getEcdStats
from config import settings


class ZoomableViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.zoom_factor = 1.15

    def setPixmap(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        zoom = self.zoom_factor if event.angleDelta().y() > 0 else 1 / self.zoom_factor
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(zoom, zoom)


class BatchWorker(QThread):
    """배치 분석을 별도 스레드에서 실행 (UI 멈춤 방지)"""
    progress = pyqtSignal(int, int, str)   # (현재, 전체, 파일명)
    finished = pyqtSignal(int)             # 완료된 파일 수

    def __init__(self, image_files, thresholds, params):
        super().__init__()
        self.image_files = image_files
        self.thresholds = thresholds
        self.params = params

    def run(self):
        total = len(self.image_files)
        done = 0
        for path in self.image_files:
            self.progress.emit(done, total, os.path.basename(path))
            res, data = analyzeImage(path, self.thresholds, self.params)
            if res is not None:
                saveData(path, data)
                done += 1
        self.finished.emit(done)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('OM Analyzer')
        self.setGeometry(50, 50, 1600, 900)
        self.showMaximized()

        self.current_image_path = None
        self.image_files = []

        self.thresh = settings.PHASE_THRESHOLDS.copy()
        self.min_areas = settings.MIN_AREAS.copy()
        self.si_rules = settings.SI_RULES.copy()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        panel = QWidget()
        panel.setFixedWidth(680)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        # --- 파일 선택 ---
        file_grp = QGroupBox("파일 선택")
        file_lay = QVBoxLayout(file_grp)
        self.btn_load = QPushButton('폴더 열기')
        self.file_selector = QComboBox()
        file_lay.addWidget(self.btn_load)
        file_lay.addWidget(self.file_selector)
        layout.addWidget(file_grp)

        # --- 파라미터 ---
        p_grp = QGroupBox("분석 파라미터")
        p_lay = QGridLayout(p_grp)

        self.sp_p_u   = QSpinBox();  self.sp_p_u.setRange(0, 255);   self.sp_p_u.setValue(self.thresh['pore'][1])
        self.sp_si_l  = QSpinBox();  self.sp_si_l.setRange(0, 255);  self.sp_si_l.setValue(self.thresh['si'][0])
        self.sp_si_u  = QSpinBox();  self.sp_si_u.setRange(0, 255);  self.sp_si_u.setValue(self.thresh['si'][1])
        self.sp_im_l  = QSpinBox();  self.sp_im_l.setRange(0, 255);  self.sp_im_l.setValue(self.thresh['intermetallic'][0])
        self.sp_im_u  = QSpinBox();  self.sp_im_u.setRange(0, 255);  self.sp_im_u.setValue(self.thresh['intermetallic'][1])
        self.sp_a_l   = QSpinBox();  self.sp_a_l.setRange(0, 255);   self.sp_a_l.setValue(self.thresh['alpha_al'][0])

        self.sp_a_pore  = QSpinBox(); self.sp_a_pore.setRange(0, 10000);  self.sp_a_pore.setValue(self.min_areas['pore'])
        self.sp_a_si    = QSpinBox(); self.sp_a_si.setRange(0, 10000);    self.sp_a_si.setValue(self.si_rules['min_area'])
        self.sp_a_inter = QSpinBox(); self.sp_a_inter.setRange(0, 10000); self.sp_a_inter.setValue(self.min_areas['intermetallic'])
        self.sp_a_alpha = QSpinBox(); self.sp_a_alpha.setRange(0, 10000); self.sp_a_alpha.setValue(self.min_areas['alpha_al'])

        self.sp_si_circ = QDoubleSpinBox(); self.sp_si_circ.setRange(0, 1.0);  self.sp_si_circ.setSingleStep(0.05); self.sp_si_circ.setValue(self.si_rules['max_circularity'])
        self.sp_si_ar   = QDoubleSpinBox(); self.sp_si_ar.setRange(1.0, 50.0); self.sp_si_ar.setSingleStep(0.5);   self.sp_si_ar.setValue(self.si_rules['min_aspect_ratio'])

        # CLAHE 토글
        self.chk_clahe = QCheckBox("CLAHE 정규화 사용 (조명/에칭 차이 보정)")
        self.chk_clahe.setChecked(settings.USE_CLAHE)
        self.chk_clahe.stateChanged.connect(self._onClaheToggle)

        for sp in [self.sp_p_u, self.sp_si_l, self.sp_si_u, self.sp_im_l, self.sp_im_u,
                   self.sp_a_l, self.sp_a_pore, self.sp_a_si, self.sp_a_inter, self.sp_a_alpha,
                   self.sp_si_circ, self.sp_si_ar]:
            sp.setMinimumWidth(80)
            sp.setStyleSheet("font-size: 14px; padding: 2px;")

        headers = ["상(Phase)", "밝기 범위", "최소 면적(px)", "Si 형상 규칙"]
        for col, h in enumerate(headers):
            p_lay.addWidget(QLabel(f"<b>{h}</b>"), 0, col, Qt.AlignCenter)

        p_lay.addWidget(QLabel("기공 (Pore)"),       1, 0)
        p_lay.addWidget(self.sp_p_u,                 1, 1)
        p_lay.addWidget(self.sp_a_pore,              1, 2)

        p_lay.addWidget(QLabel("Eutectic Si"),       2, 0)
        w_si = QWidget(); si_lay = QHBoxLayout(w_si); si_lay.setContentsMargins(0,0,0,0)
        si_lay.addWidget(self.sp_si_l); si_lay.addWidget(QLabel("~")); si_lay.addWidget(self.sp_si_u)
        p_lay.addWidget(w_si,                        2, 1)
        p_lay.addWidget(self.sp_a_si,                2, 2)
        w_rule = QWidget(); rule_lay = QVBoxLayout(w_rule); rule_lay.setContentsMargins(0,0,0,0)
        rule_lay.addWidget(QLabel("최대 원형도:")); rule_lay.addWidget(self.sp_si_circ)
        rule_lay.addWidget(QLabel("최소 종횡비:")); rule_lay.addWidget(self.sp_si_ar)
        p_lay.addWidget(w_rule,                      2, 3)

        p_lay.addWidget(QLabel("Intermetallics"),    3, 0)
        w_im = QWidget(); im_lay = QHBoxLayout(w_im); im_lay.setContentsMargins(0,0,0,0)
        im_lay.addWidget(self.sp_im_l); im_lay.addWidget(QLabel("~")); im_lay.addWidget(self.sp_im_u)
        p_lay.addWidget(w_im,                        3, 1)
        p_lay.addWidget(self.sp_a_inter,             3, 2)

        p_lay.addWidget(QLabel("Alpha-Al 매트릭스"), 4, 0)
        p_lay.addWidget(self.sp_a_l,                 4, 1)
        p_lay.addWidget(self.sp_a_alpha,             4, 2)

        p_lay.addWidget(self.chk_clahe,              5, 0, 1, 4)
        layout.addWidget(p_grp)

        # --- 범례 ---
        legend = QGroupBox("결과 범례")
        l_lay = QHBoxLayout(legend)
        l_lay.addWidget(QLabel("<span style='color:red;'>■</span> 알파-Al"))
        l_lay.addWidget(QLabel("<span style='color:yellow;'>■</span> Intermetallics"))
        l_lay.addWidget(QLabel("<span style='color:green;'>■</span> Eutectic Si"))
        l_lay.addWidget(QLabel("<span style='color:blue;'>■</span> 기공"))
        layout.addWidget(legend)

        # --- 버튼 ---
        btn_row = QHBoxLayout()
        self.btn_run   = QPushButton('▶ 단일 이미지 분석')
        self.btn_batch = QPushButton('⚡ 배치 분석 (전체 폴더)')
        self.btn_run.setMinimumHeight(40)
        self.btn_batch.setMinimumHeight(40)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_batch)
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.txt = QTextEdit()
        self.txt.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.txt)
        main_layout.addWidget(panel)

        # --- 이미지 뷰어 ---
        splitter = QSplitter(Qt.Vertical)
        self.viewer_orig   = ZoomableViewer()
        self.viewer_result = ZoomableViewer()
        grp_orig = QGroupBox("원본 이미지 (휠: 줌, 드래그: 이동)")
        QVBoxLayout(grp_orig).addWidget(self.viewer_orig)
        grp_res = QGroupBox("분석 결과 이미지 (휠: 줌, 드래그: 이동)")
        QVBoxLayout(grp_res).addWidget(self.viewer_result)
        splitter.addWidget(grp_orig)
        splitter.addWidget(grp_res)
        main_layout.addWidget(splitter, 1)

        self.btn_load.clicked.connect(self.loadFolder)
        self.btn_run.clicked.connect(self.runAnalysis)
        self.btn_batch.clicked.connect(self.runBatch)
        self.file_selector.currentIndexChanged.connect(self.updateImage)

    # ------------------------------------------------------------------
    def _onClaheToggle(self, state):
        settings.USE_CLAHE = bool(state)

    def _currentParams(self):
        thresh = {
            'pore':          (0, self.sp_p_u.value()),
            'si':            (self.sp_si_l.value(), self.sp_si_u.value()),
            'intermetallic': (self.sp_im_l.value(), self.sp_im_u.value()),
            'alpha_al':      (self.sp_a_l.value(), 255),
        }
        min_areas = {
            'pore':          self.sp_a_pore.value(),
            'alpha_al':      self.sp_a_alpha.value(),
            'intermetallic': self.sp_a_inter.value(),
        }
        si_rules = {
            'min_area':        self.sp_a_si.value(),
            'max_circularity': self.sp_si_circ.value(),
            'min_aspect_ratio': self.sp_si_ar.value(),
        }
        return thresh, {'min_areas': min_areas, 'si_rules': si_rules}

    def loadFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if not folder:
            return
        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        self.image_files = [
            os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(exts)
        ]
        self.file_selector.clear()
        self.file_selector.addItem(f"--- {len(self.image_files)}개 파일 로드됨 ---")
        for f in self.image_files:
            self.file_selector.addItem(os.path.basename(f))

    def updateImage(self, idx):
        if 0 < idx <= len(self.image_files):
            self.current_image_path = self.image_files[idx - 1]
            self.displayImage(self.viewer_orig, self.current_image_path)
            self._clearResultViewer()

    def _clearResultViewer(self):
        self.viewer_result.scene.clear()
        self.viewer_result.pixmap_item = QGraphicsPixmapItem()
        self.viewer_result.scene.addItem(self.viewer_result.pixmap_item)

    def runAnalysis(self):
        if not self.current_image_path:
            return
        thresh, params = self._currentParams()
        res, data = analyzeImage(self.current_image_path, thresh, params)
        if res is not None:
            self.displayImage(self.viewer_result, res)
            self.txt.setText(self._formatResult(data))
            saveData(self.current_image_path, data)

    def runBatch(self):
        if not self.image_files:
            QMessageBox.warning(self, "경고", "먼저 폴더를 열어 이미지를 로드하세요.")
            return
        thresh, params = self._currentParams()
        self.btn_run.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.image_files))
        self.progress_bar.setValue(0)
        self.txt.setText("배치 분석 시작...\n")

        self._batch_worker = BatchWorker(self.image_files, thresh, params)
        self._batch_worker.progress.connect(self._onBatchProgress)
        self._batch_worker.finished.connect(self._onBatchFinished)
        self._batch_worker.start()

    def _onBatchProgress(self, done, total, filename):
        self.progress_bar.setValue(done)
        self.txt.append(f"[{done+1}/{total}] {filename}")

    def _onBatchFinished(self, done):
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.txt.append(f"\n완료: {done}개 이미지 분석 저장 → {settings.OUTPUT_DIR}")
        self.btn_run.setEnabled(True)
        self.btn_batch.setEnabled(True)

    def _formatResult(self, data):
        lines = ["--- 면적 분율 ---"]
        phase_names = {
            'alpha_al_fraction':      'Alpha-Al',
            'eutectic_si_fraction':   'Eutectic Si',
            'intermetallic_fraction': 'Intermetallics',
            'pore_fraction':          '기공 (Pore)',
        }
        for key, name in phase_names.items():
            lines.append(f"  {name}: {data.get(key, 0):.2f}%")

        lines.append("\n--- ECD 분포 (등가원직경, μm) ---")
        stats = getEcdStats(data)
        label_map = {
            'eutectic_si':   'Eutectic Si',
            'intermetallic': 'Intermetallics',
            'pore':          '기공',
            'alpha_al':      'Alpha-Al',
        }
        for phase, label in label_map.items():
            s = stats.get(phase)
            if s:
                lines.append(
                    f"  {label}: n={s['count']}  "
                    f"평균={s['mean']:.2f}  std={s['std']:.2f}  "
                    f"min={s['min']:.2f}  max={s['max']:.2f} μm"
                )
        return "\n".join(lines)

    def displayImage(self, viewer, img_src):
        img = cv2.imread(img_src) if isinstance(img_src, str) else img_src
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                      rgb.shape[1] * 3, QImage.Format_RGB888)
        viewer.setPixmap(QPixmap.fromImage(qimg))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
