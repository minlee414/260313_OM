import sys, os, cv2, numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from core.image_processor import analyzeImage, loadGrayImage
from core.data_handler import saveData, getEcdStats
from config import settings


class ZoomableViewer(QGraphicsView):
    # 마우스가 이미지 위를 움직일 때 (x, y, gray_value) 전달
    pixelHovered = pyqtSignal(int, int, int)
    # 뷰 변경(줌·패닝) 시 transform + 스크롤 위치 동기화용
    viewSynced = pyqtSignal(object, int, int)   # (QTransform, hval, vval)

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
        self.setMouseTracking(True)

        self._gray_image = None
        self._syncing = False

        # 패닝(드래그)으로 스크롤 바 값이 바뀔 때도 동기화
        self.horizontalScrollBar().valueChanged.connect(self._onScrollChanged)
        self.verticalScrollBar().valueChanged.connect(self._onScrollChanged)

    def setGrayImage(self, gray_np):
        self._gray_image = gray_np

    def setPixmap(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _onScrollChanged(self):
        if not self._syncing:
            self.viewSynced.emit(
                self.transform(),
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value(),
            )

    def applySync(self, transform, hval, vval):
        """반대쪽 뷰어에서 받은 transform + 스크롤 위치를 그대로 적용"""
        if self._syncing:
            return
        self._syncing = True
        self.setTransform(transform)
        self.horizontalScrollBar().setValue(hval)
        self.verticalScrollBar().setValue(vval)
        self._syncing = False

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._gray_image is None:
            return
        scene_pos = self.mapToScene(event.pos())
        x, y = int(scene_pos.x()), int(scene_pos.y())
        h, w = self._gray_image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            val = int(self._gray_image[y, x])
            self.pixelHovered.emit(x, y, val)

    def wheelEvent(self, event):
        zoom = self.zoom_factor if event.angleDelta().y() > 0 else 1 / self.zoom_factor
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(zoom, zoom)
        # wheelEvent 후 스크롤 바도 같이 바뀌므로 _onScrollChanged 에서 자동 emit


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

        self.sp_a_pore   = QSpinBox(); self.sp_a_pore.setRange(0, 10000);   self.sp_a_pore.setValue(self.min_areas['pore'])
        self.sp_a_si     = QSpinBox(); self.sp_a_si.setRange(0, 10000);     self.sp_a_si.setValue(self.si_rules['min_area'])
        self.sp_a_inter  = QSpinBox(); self.sp_a_inter.setRange(0, 10000);  self.sp_a_inter.setValue(self.min_areas['intermetallic'])
        self.sp_a_alpha  = QSpinBox(); self.sp_a_alpha.setRange(0, 10000);  self.sp_a_alpha.setValue(self.min_areas['alpha_al'])
        self.sp_ma_inter = QSpinBox(); self.sp_ma_inter.setRange(0, 100000); self.sp_ma_inter.setValue(settings.MAX_AREAS['intermetallic'])
        self.sp_ma_inter.setSpecialValueText("0 (끔)")

        # 로컬 대비 파라미터
        self.sp_lc_kernel = QSpinBox()
        self.sp_lc_kernel.setRange(11, 301)
        self.sp_lc_kernel.setSingleStep(10)
        self.sp_lc_kernel.setValue(settings.LOCAL_CONTRAST['kernel_size'])

        self.sp_lc_diff = QSpinBox()
        self.sp_lc_diff.setRange(0, 100)
        self.sp_lc_diff.setValue(settings.LOCAL_CONTRAST['min_diff'])

        # Si 경계 제외 반경
        self.sp_si_excl = QSpinBox()
        self.sp_si_excl.setRange(0, 30)
        self.sp_si_excl.setValue(settings.SI_EXCLUSION_RADIUS)
        self.sp_si_excl.setSpecialValueText("0 (끔)")

        self.chk_clahe = QCheckBox("CLAHE 정규화 사용 (조명/에칭 차이 보정)")
        self.chk_clahe.setChecked(settings.USE_CLAHE)
        self.chk_clahe.stateChanged.connect(self._onClaheToggle)

        # Watershed 입자 분리 (상별 개별 체크박스)
        ws = settings.WATERSHED_PHASES
        self.chk_ws_si    = QCheckBox("Si")
        self.chk_ws_im    = QCheckBox("IM")
        self.chk_ws_al    = QCheckBox("Al")
        self.chk_ws_pore  = QCheckBox("기공")
        self.chk_ws_si.setChecked(ws.get('si', False))
        self.chk_ws_im.setChecked(ws.get('intermetallic', False))
        self.chk_ws_al.setChecked(ws.get('alpha_al', False))
        self.chk_ws_pore.setChecked(ws.get('pore', False))

        self.sp_ws_ratio = QDoubleSpinBox()
        self.sp_ws_ratio.setRange(0.01, 0.9)
        self.sp_ws_ratio.setSingleStep(0.05)
        self.sp_ws_ratio.setDecimals(2)
        self.sp_ws_ratio.setValue(settings.WATERSHED_DIST_RATIO)
        self.sp_ws_ratio.setMinimumWidth(70)
        self.sp_ws_ratio.setStyleSheet("font-size: 14px; padding: 2px;")

        for sp in [self.sp_p_u, self.sp_si_l, self.sp_si_u, self.sp_im_l, self.sp_im_u,
                   self.sp_a_l, self.sp_a_pore, self.sp_a_si, self.sp_a_inter, self.sp_a_alpha,
                   self.sp_lc_kernel, self.sp_lc_diff, self.sp_ma_inter, self.sp_si_excl]:
            sp.setMinimumWidth(80)
            sp.setStyleSheet("font-size: 14px; padding: 2px;")

        headers = ["상(Phase)", "밝기 범위", "면적 필터(px)"]
        for col, h in enumerate(headers):
            p_lay.addWidget(QLabel(f"<b>{h}</b>"), 0, col, Qt.AlignCenter)

        # 1행: 기공
        p_lay.addWidget(QLabel("기공 (Pore)"),        1, 0)
        p_lay.addWidget(self.sp_p_u,                  1, 1)
        p_lay.addWidget(self.sp_a_pore,               1, 2)

        # 2행: Eutectic Si  (밝기 하단 구간 → 직접 Si)
        p_lay.addWidget(QLabel("Eutectic Si"),         2, 0)
        w_si = QWidget(); si_lay = QHBoxLayout(w_si); si_lay.setContentsMargins(0,0,0,0)
        si_lay.addWidget(self.sp_si_l); si_lay.addWidget(QLabel("~")); si_lay.addWidget(self.sp_si_u)
        p_lay.addWidget(w_si,                         2, 1)
        p_lay.addWidget(self.sp_a_si,                 2, 2)

        # 3행: Intermetallics  (밝기 상단 구간 → 직접 IM)
        p_lay.addWidget(QLabel("Intermetallics"),      3, 0)
        w_im = QWidget(); im_lay = QHBoxLayout(w_im); im_lay.setContentsMargins(0,0,0,0)
        im_lay.addWidget(self.sp_im_l); im_lay.addWidget(QLabel("~")); im_lay.addWidget(self.sp_im_u)
        p_lay.addWidget(w_im,                         3, 1)
        w_im_area = QWidget(); im_area_lay = QHBoxLayout(w_im_area); im_area_lay.setContentsMargins(0,0,0,0)
        im_area_lay.addWidget(QLabel("최소:")); im_area_lay.addWidget(self.sp_a_inter)
        im_area_lay.addWidget(QLabel("  최대:")); im_area_lay.addWidget(self.sp_ma_inter)
        p_lay.addWidget(w_im_area,                    3, 2)

        # 4행: Alpha-Al
        p_lay.addWidget(QLabel("Alpha-Al 매트릭스"),   4, 0)
        p_lay.addWidget(self.sp_a_l,                  4, 1)
        p_lay.addWidget(self.sp_a_alpha,              4, 2)

        # 5행: 로컬 대비 필터
        lc_grp_w = QWidget(); lc_lay = QHBoxLayout(lc_grp_w); lc_lay.setContentsMargins(0,0,0,0)
        lc_lay.addWidget(QLabel("로컬 대비 (Si/IM):"))
        lc_lay.addWidget(QLabel("커널:"))
        lc_lay.addWidget(self.sp_lc_kernel)
        lc_lay.addWidget(QLabel("px   최소 차이:"))
        lc_lay.addWidget(self.sp_lc_diff)
        lc_lay.addWidget(QLabel("(0=끔)"))
        lc_lay.addStretch()
        p_lay.addWidget(lc_grp_w,                    5, 0, 1, 4)

        # 6행: Si 경계 제외 반경
        excl_w = QWidget(); excl_lay = QHBoxLayout(excl_w); excl_lay.setContentsMargins(0,0,0,0)
        excl_lay.addWidget(QLabel("Si 경계 제외 반경:"))
        excl_lay.addWidget(self.sp_si_excl)
        excl_lay.addWidget(QLabel("px  (Si-Al 그래디언트 → IM 오분류 방지, 0=끔)"))
        excl_lay.addStretch()
        p_lay.addWidget(excl_w,                      6, 0, 1, 4)
        p_lay.addWidget(self.chk_clahe,              7, 0, 1, 4)

        # 8행: Watershed (상별 개별)
        ws_w = QWidget(); ws_lay = QHBoxLayout(ws_w); ws_lay.setContentsMargins(0,0,0,0)
        ws_lay.addWidget(QLabel("Watershed:"))
        ws_lay.addWidget(self.chk_ws_si)
        ws_lay.addWidget(self.chk_ws_im)
        ws_lay.addWidget(self.chk_ws_al)
        ws_lay.addWidget(self.chk_ws_pore)
        ws_lay.addWidget(QLabel("  민감도:"))
        ws_lay.addWidget(self.sp_ws_ratio)
        ws_lay.addWidget(QLabel("(낮을수록 많이 나눔)"))
        ws_lay.addStretch()
        p_lay.addWidget(ws_w,                        8, 0, 1, 4)
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
        orig_layout = QVBoxLayout(grp_orig)
        orig_layout.addWidget(self.viewer_orig)
        # 호버 시 밝기 값 표시 레이블
        self.lbl_hover = QLabel("커서를 원본 이미지 위에 올리면 밝기 값이 표시됩니다.")
        self.lbl_hover.setStyleSheet("font-size: 13px; color: #333; padding: 2px;")
        orig_layout.addWidget(self.lbl_hover)

        grp_res = QGroupBox("분석 결과 이미지 (휠: 줌, 드래그: 이동)")
        QVBoxLayout(grp_res).addWidget(self.viewer_result)

        splitter.addWidget(grp_orig)
        splitter.addWidget(grp_res)
        main_layout.addWidget(splitter, 1)

        self.btn_load.clicked.connect(self.loadFolder)
        self.btn_run.clicked.connect(self.runAnalysis)
        self.btn_batch.clicked.connect(self.runBatch)
        self.file_selector.currentIndexChanged.connect(self.updateImage)
        self.viewer_orig.pixelHovered.connect(self._onPixelHovered)

        # 원본 ↔ 결과 뷰어 줌·패닝 동기화 (같은 위치를 같이 봄)
        self.viewer_orig.viewSynced.connect(self.viewer_result.applySync)
        self.viewer_result.viewSynced.connect(self.viewer_orig.applySync)

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
            'min_area': self.sp_a_si.value(),
        }
        k = self.sp_lc_kernel.value()
        local_contrast = {
            'kernel_size': k if k % 2 == 1 else k + 1,
            'min_diff':    self.sp_lc_diff.value(),
        }
        max_areas = {
            'intermetallic': self.sp_ma_inter.value(),
        }
        return thresh, {'min_areas': min_areas, 'si_rules': si_rules,
                        'local_contrast': local_contrast, 'max_areas': max_areas,
                        'si_exclusion_radius': self.sp_si_excl.value(),
                        'watershed_phases': {
                            'si':            self.chk_ws_si.isChecked(),
                            'intermetallic': self.chk_ws_im.isChecked(),
                            'alpha_al':      self.chk_ws_al.isChecked(),
                            'pore':          self.chk_ws_pore.isChecked(),
                        },
                        'watershed_dist_ratio': self.sp_ws_ratio.value()}

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

    def _onPixelHovered(self, x, y, val):
        phase = self._grayValueToPhase(val)
        self.lbl_hover.setText(
            f"  위치: ({x}, {y})   밝기(gray): {val}   "
            f"→ 현재 파라미터 기준 추정 상: {phase}"
        )

    def _grayValueToPhase(self, val):
        """현재 GUI 임계값 기준으로 픽셀 밝기가 어느 상에 해당하는지 추정"""
        if val <= self.sp_p_u.value():
            return "기공 (Pore)"
        if self.sp_si_l.value() <= val <= self.sp_si_u.value():
            return "Eutectic Si"
        if self.sp_im_l.value() <= val <= self.sp_im_u.value():
            return "Intermetallics"
        if val >= self.sp_a_l.value():
            return "Alpha-Al"
        return "미분류"

    def updateImage(self, idx):
        if 0 < idx <= len(self.image_files):
            self.current_image_path = self.image_files[idx - 1]
            self.displayImage(self.viewer_orig, self.current_image_path)
            # 호버용 그레이스케일 이미지 로드 및 뷰어에 전달
            gray = loadGrayImage(self.current_image_path)
            self.viewer_orig.setGrayImage(gray)
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
