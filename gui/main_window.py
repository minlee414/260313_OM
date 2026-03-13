import sys, os, cv2, numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt

from core.image_processor import analyzeImage
from core.data_handler import saveData 
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
        if event.angleDelta().y() > 0:
            zoom = self.zoom_factor
        else:
            zoom = 1 / self.zoom_factor
        
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(zoom, zoom)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('OM Analyzer - 정밀 줌/분류 도구')
        self.setGeometry(50, 50, 1600, 900)
        self.showMaximized() 
        
        self.thresh = settings.PHASE_THRESHOLDS.copy()
        self.min_areas = settings.MIN_AREAS.copy()
        self.si_rules = settings.SI_RULES.copy()
        
        central = QWidget(); self.setCentralWidget(central); main_layout = QHBoxLayout(central)
        
        # ★ 왼쪽 패널 너비를 550에서 650으로 확장
        panel = QWidget(); panel.setFixedWidth(650) 
        layout = QVBoxLayout(panel); layout.setAlignment(Qt.AlignTop)
        
        file_grp = QGroupBox("파일 선택"); file_lay = QVBoxLayout(file_grp)
        self.btn_load = QPushButton('폴더 열기'); self.file_selector = QComboBox()
        file_lay.addWidget(self.btn_load); file_lay.addWidget(self.file_selector); layout.addWidget(file_grp)

        p_grp = QGroupBox("분석 파라미터 상세 제어"); p_lay = QGridLayout(p_grp)
        
        # 스핀박스 인스턴스 생성
        self.sp_p_u = QSpinBox(); self.sp_p_u.setRange(0, 255); self.sp_p_u.setValue(self.thresh['pore'][1])
        self.sp_si_l = QSpinBox(); self.sp_si_l.setRange(0, 255); self.sp_si_l.setValue(self.thresh['si'][0])
        self.sp_si_u = QSpinBox(); self.sp_si_u.setRange(0, 255); self.sp_si_u.setValue(self.thresh['si'][1])
        self.sp_im_l = QSpinBox(); self.sp_im_l.setRange(0, 255); self.sp_im_l.setValue(self.thresh['intermetallic'][0])
        self.sp_im_u = QSpinBox(); self.sp_im_u.setRange(0, 255); self.sp_im_u.setValue(self.thresh['intermetallic'][1])
        self.sp_a_l = QSpinBox(); self.sp_a_l.setRange(0, 255); self.sp_a_l.setValue(self.thresh['alpha_al'][0])
        
        self.sp_a_pore = QSpinBox(); self.sp_a_pore.setRange(0, 10000); self.sp_a_pore.setValue(self.min_areas['pore'])
        self.sp_a_si = QSpinBox(); self.sp_a_si.setRange(0, 10000); self.sp_a_si.setValue(self.si_rules['min_area'])
        self.sp_a_inter = QSpinBox(); self.sp_a_inter.setRange(0, 10000); self.sp_a_inter.setValue(self.min_areas['intermetallic'])
        self.sp_a_alpha = QSpinBox(); self.sp_a_alpha.setRange(0, 10000); self.sp_a_alpha.setValue(self.min_areas['alpha_al'])
        
        self.sp_si_circ = QDoubleSpinBox(); self.sp_si_circ.setRange(0, 1.0); self.sp_si_circ.setSingleStep(0.05); self.sp_si_circ.setValue(self.si_rules['max_circularity'])
        self.sp_si_ar = QDoubleSpinBox(); self.sp_si_ar.setRange(1.0, 50.0); self.sp_si_ar.setSingleStep(0.5); self.sp_si_ar.setValue(self.si_rules['min_aspect_ratio'])

        # ★ 모든 입력칸 너비 2배 확장 및 폰트 크기 증가
        spinboxes =[self.sp_p_u, self.sp_si_l, self.sp_si_u, self.sp_im_l, self.sp_im_u, self.sp_a_l,
                     self.sp_a_pore, self.sp_a_si, self.sp_a_inter, self.sp_a_alpha,
                     self.sp_si_circ, self.sp_si_ar]
        for sp in spinboxes:
            sp.setMinimumWidth(80) # 너비 강제 확장
            sp.setStyleSheet("font-size: 14px; padding: 2px;") # 글씨 크기 확장

        # 표 헤더
        headers =["상(Phase)", "밝기 범위", "최소 면적", "Si 형상 규칙"]
        for col, h in enumerate(headers): p_lay.addWidget(QLabel(f"<b>{h}</b>"), 0, col, Qt.AlignCenter)

        # 1행: 기공
        p_lay.addWidget(QLabel("기공 (Pore)"), 1, 0)
        p_lay.addWidget(self.sp_p_u, 1, 1); p_lay.addWidget(self.sp_a_pore, 1, 2)

        # ★ 2행: Eutectic Si (IM 위로 순서 변경)
        p_lay.addWidget(QLabel("Eutectic Si"), 2, 0)
        si_lay = QHBoxLayout(); si_lay.addWidget(self.sp_si_l); si_lay.addWidget(QLabel("~")); si_lay.addWidget(self.sp_si_u)
        w_si = QWidget(); w_si.setLayout(si_lay); p_lay.addWidget(w_si, 2, 1)
        p_lay.addWidget(self.sp_a_si, 2, 2)
        si_rule_lay = QVBoxLayout()
        si_rule_lay.addWidget(QLabel("최대 원형도:")); si_rule_lay.addWidget(self.sp_si_circ)
        si_rule_lay.addWidget(QLabel("최소 종횡비:")); si_rule_lay.addWidget(self.sp_si_ar)
        w_rule = QWidget(); w_rule.setLayout(si_rule_lay); p_lay.addWidget(w_rule, 2, 3)

        # ★ 3행: Intermetallics (Si 아래로 순서 변경)
        p_lay.addWidget(QLabel("Intermetallics"), 3, 0)
        im_lay = QHBoxLayout(); im_lay.addWidget(self.sp_im_l); im_lay.addWidget(QLabel("~")); im_lay.addWidget(self.sp_im_u)
        w_im = QWidget(); w_im.setLayout(im_lay); p_lay.addWidget(w_im, 3, 1)
        p_lay.addWidget(self.sp_a_inter, 3, 2)

        # 4행: Alpha-Al
        p_lay.addWidget(QLabel("Alpha-Al 매트릭스"), 4, 0)
        p_lay.addWidget(self.sp_a_l, 4, 1); p_lay.addWidget(self.sp_a_alpha, 4, 2)
        layout.addWidget(p_grp)
        
        legend = QGroupBox("결과 범례"); l_lay = QHBoxLayout(legend)
        l_lay.addWidget(QLabel("<span style='color:red;'>■</span> 알파-Al"))
        l_lay.addWidget(QLabel("<span style='color:yellow;'>■</span> Intermetallics"))
        l_lay.addWidget(QLabel("<span style='color:green;'>■</span> Eutectic Si"))
        l_lay.addWidget(QLabel("<span style='color:blue;'>■</span> 기공"))
        layout.addWidget(legend)

        self.btn_run = QPushButton('▶ 파라미터 적용 및 분석 실행'); self.btn_run.setMinimumHeight(40)
        layout.addWidget(self.btn_run)
        
        self.txt = QTextEdit()
        self.txt.setStyleSheet("font-size: 14px;") # 결과 창 글씨도 약간 키움
        layout.addWidget(self.txt)
        main_layout.addWidget(panel)

        # 오른쪽 뷰어 패널 (QSplitter)
        splitter = QSplitter(Qt.Vertical)
        self.viewer_orig = ZoomableViewer()
        self.viewer_result = ZoomableViewer()
        
        grp_orig = QGroupBox("원본 이미지 (마우스 휠: 줌, 클릭&드래그: 이동)"); l_orig = QVBoxLayout(grp_orig); l_orig.addWidget(self.viewer_orig)
        grp_res = QGroupBox("분석 결과 이미지 (마우스 휠: 줌, 클릭&드래그: 이동)"); l_res = QVBoxLayout(grp_res); l_res.addWidget(self.viewer_result)

        splitter.addWidget(grp_orig)
        splitter.addWidget(grp_res)
        main_layout.addWidget(splitter, 1)

        self.btn_load.clicked.connect(self.loadFolder)
        self.btn_run.clicked.connect(self.runAnalysis)
        self.file_selector.currentIndexChanged.connect(self.updateImage)

    def loadFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "선택")
        if not folder: return
        self.image_files =[os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
        self.file_selector.clear(); self.file_selector.addItem("--- 파일 선택 ---")
        for f in self.image_files: self.file_selector.addItem(os.path.basename(f))

    def updateImage(self, idx):
        if idx > 0 and idx <= len(self.image_files):
            self.current_image_path = self.image_files[idx-1]
            self.displayImage(self.viewer_orig, self.current_image_path)
            self.viewer_result.scene.clear()
            self.viewer_result.pixmap_item = QGraphicsPixmapItem()
            self.viewer_result.scene.addItem(self.viewer_result.pixmap_item)

    def runAnalysis(self):
        if not self.current_image_path: return
        
        self.thresh['pore'] = (0, self.sp_p_u.value())
        self.thresh['si'] = (self.sp_si_l.value(), self.sp_si_u.value())
        self.thresh['intermetallic'] = (self.sp_im_l.value(), self.sp_im_u.value())
        self.thresh['alpha_al'] = (self.sp_a_l.value(), 255)
        
        self.min_areas = {
            'pore': self.sp_a_pore.value(),
            'alpha_al': self.sp_a_alpha.value(),
            'intermetallic': self.sp_a_inter.value()
        }
        self.si_rules = {
            'min_area': self.sp_a_si.value(),
            'max_circularity': self.sp_si_circ.value(),
            'min_aspect_ratio': self.sp_si_ar.value()
        }
        
        all_params = {'min_areas': self.min_areas, 'si_rules': self.si_rules}
        
        res, data = analyzeImage(self.current_image_path, self.thresh, all_params)
        if res is not None:
            self.displayImage(self.viewer_result, res)
            self.txt.setText("--- 분석 결과 ---\n" + "\n".join([f"{k}: {v:.2f}%" for k,v in data.items() if 'fraction' in k]))
            saveData(self.current_image_path, data)

    def displayImage(self, viewer, img_src):
        img = cv2.imread(img_src) if isinstance(img_src, str) else img_src
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format_RGB888)
        viewer.setPixmap(QPixmap.fromImage(qimg))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())