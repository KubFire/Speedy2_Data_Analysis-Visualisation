#TO DO
# button for screenshot
# updatovat aby ruzova cara ve 3D attitude se menila pokud je selected baro nebo accel.
# 2D attitude tak abych mohl zvetsovat zmensovat verrtikalni osu nezavisle na horizontalni.
# zmenit napis position na 3D position
# zmenit nazev Altitude vs time na Altitude
# 2D attitude sirka neni stejna jako box 3D attitude nad ni
# timestamp touchdown ve 3D position nedoshauje k lajne barometer altitude.
# dat vpravo nahoru png "Kubfire" logo 
# dat do leveho horniho rohu logo Speedy 2
# dat deepresearchi at uklidi ten kod.


import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import traceback
import ctypes

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QPushButton, QLabel, QFileDialog, 
                             QSplitter, QComboBox, QCheckBox, QGroupBox, QStyleOptionSlider, QStyle)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QPainter, QColor, QFont, QVector3D, QPen, QShortcut, QKeySequence
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# ==========================================
# CONFIGURATION
# ==========================================
LOGO_FILENAME = r'C:\Users\kubfi\OneDrive\Documents\GitHub\Speedy2_Data_Analysis&Visualisation\visualisation\SPEEDY_LOGO_V2_red_lowres.png'

# Fix taskbar icon on Windows
try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('speedy2.visualiser.1.0')
except Exception:
    pass

class MarkerSlider(QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.markers = []

    def set_markers(self, markers):
        self.markers = list(markers.keys())
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.markers or self.maximum() <= 0:
            return
        painter = QPainter(self)
        painter.setPen(QPen(QColor("red"), 2))
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        rect = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, self)
        
        for m in self.markers:
            frac = m / self.maximum()
            x = rect.left() + int(frac * rect.width())
            painter.drawLine(x, rect.top() + 2, x, rect.bottom() - 2)

class TimelineLabels(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.markers = {}
        self.total_frames = 1
        self.setFixedHeight(20)

    def set_markers(self, markers, total_frames):
        self.markers = markers
        self.total_frames = max(1, total_frames)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor("red"))
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        
        w = self.width()
        margin = 10 
        track_w = w - (margin * 2)
        
        for frame, text in self.markers.items():
            x = margin + int((frame / self.total_frames) * track_w)
            painter.drawText(x - 15, 15, text)


class RocketVisualizerGUI(QMainWindow):
    def __init__(self, df, file_path):
        super().__init__()
        self.setWindowTitle(f"Speedy 2 Flight Visualisation - {os.path.basename(file_path)}")
        self.setWindowIcon(QIcon(LOGO_FILENAME))
        self.resize(1800, 1000)

        # State Variables
        self.is_playing = True
        self.current_frame = 0
        self.speed_multiplier = 1.0
        self.base_fps = 60
        self.mode = 'quat' 
        
        self.inv_r = 1
        self.inv_p = 1
        self.inv_y = 1
        self.euler_seq = 'YXZ'

        self.middle_mode = 'baro'
        self.right_mode = 'baro'
        self.unwrap_angles = False
        
        self.event_markers_items = []

        # 1. Process Data
        self.process_data(df)
        
        # 2. Precompute 3D Rotations
        self.build_rocket_geometry()
        self.calculate_rotations()

        # 3. Setup UI
        self.init_ui()
        self.update_ui_state()

        # Global Spacebar Hotkey
        self.shortcut_space = QShortcut(QKeySequence("Space"), self)
        self.shortcut_space.activated.connect(self.toggle_play)

        # 4. Animation State
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / self.base_fps))

    def get_col_data(self, df, possible_names, default_val=0.0):
        df_cols_clean = [str(c).lower().replace(' ', '').replace('_', '') for c in df.columns]
        for name in possible_names:
            clean_name = str(name).lower().replace(' ', '').replace('_', '')
            if clean_name in df_cols_clean:
                idx = df_cols_clean.index(clean_name)
                return df.iloc[:, idx].values, True
        return np.full(len(df), float(default_val)), False

    def process_data(self, df):
        df = df.interpolate(method='linear').fillna(0)

        self.time_data, _ = self.get_col_data(df, ['time_s', 'Time_s', 'time'])
        self.frames_total = len(self.time_data)
        
        if 'q_w' in df.columns and 'q_x' in df.columns:
            self.mode = 'quat'
            self.q_data = df[['q_x', 'q_y', 'q_z', 'q_w']].values
            
            # Generate eulers for the 2D plot
            eulers = R.from_quat(self.q_data).as_euler('YXZ', degrees=True)
            self.raw_yaw = np.radians(eulers[:, 0])
            self.raw_pitch = np.radians(eulers[:, 1])
            self.raw_roll = np.radians(eulers[:, 2])
        else:
            self.mode = 'euler'
            r, _ = self.get_col_data(df, ['euler_x_deg', 'Attitude_Roll_deg', 'roll'])
            p, _ = self.get_col_data(df, ['euler_y_deg', 'Attitude_Pitch_deg', 'pitch'])
            y, _ = self.get_col_data(df, ['euler_z_deg', 'Attitude_Yaw_deg', 'yaw'])
            
            self.raw_roll = np.radians(r)
            self.raw_pitch = np.radians(p)
            self.raw_yaw = np.radians(y)

        self.calculate_2d_attitude()

        self.state_data, _ = self.get_col_data(df, ['State', 'state'])
        self.state_markers = {}
        state_labels = {4: "Launch", 5: "Apogee", 6: "Chute", 7: "Touchdown"}
        for s_val, s_name in state_labels.items():
            idx = np.where(self.state_data == s_val)[0]
            if len(idx) > 0:
                self.state_markers[idx[0]] = s_name

        px, _ = self.get_col_data(df, ['pos_x', 'Pos_X_m', 'x'])
        py, _ = self.get_col_data(df, ['pos_y', 'Pos_Y_m', 'y'])
        
        pz_baro, found_baro = self.get_col_data(df, ['True Alt', 'Baro_Alt', 'Baro', 'baro_z'])
        pz_fused, found_fused = self.get_col_data(df, ['Pos_Z_m_Fused', 'pos_z', 'fused_z', 'Accel_Alt'])

        if not found_baro and found_fused: pz_baro = pz_fused.copy()
        if not found_fused and found_baro: pz_fused = pz_baro.copy()

        self.pos_data_baro = np.column_stack((px, py, pz_baro))
        self.pos_data_fused = np.column_stack((px, py, pz_fused))

        vx, found_vx = self.get_col_data(df, ['vel_x'])
        vy, _ = self.get_col_data(df, ['vel_y'])
        vz, _ = self.get_col_data(df, ['vel_z'])

        if found_vx:
            self.vel_data = np.column_stack((vx, vy, vz))
        else:
            self.vel_data = np.gradient(self.pos_data_baro, self.time_data, axis=0)

        sb = np.linalg.norm(self.vel_data, axis=1, keepdims=True)
        self.vel_dirs = np.divide(self.vel_data, sb, out=np.zeros_like(self.vel_data), where=sb!=0) * 2.0

        self.mids = [np.nanmean([np.nanmin(self.pos_data_baro[:, i]), np.nanmax(self.pos_data_baro[:, i])]) for i in range(3)]
        self.max_range = max([np.nanmax(self.pos_data_baro[:, i]) - np.nanmin(self.pos_data_baro[:, i]) for i in range(3)]) / 2.0
        if self.max_range == 0: self.max_range = 1
        
        self.min_z_baro = np.nanmin(self.pos_data_baro[:, 2])
        self.min_z_fused = np.nanmin(self.pos_data_fused[:, 2])

    def calculate_2d_attitude(self):
        self.att_r_raw = np.degrees(self.raw_roll * self.inv_r)
        self.att_p_raw = np.degrees(self.raw_pitch * self.inv_p)
        self.att_y_raw = np.degrees(self.raw_yaw * self.inv_y)

        self.att_r_unwrap = np.degrees(np.unwrap(np.radians(self.att_r_raw)))
        self.att_p_unwrap = np.degrees(np.unwrap(np.radians(self.att_p_raw)))
        self.att_y_unwrap = np.degrees(np.unwrap(np.radians(self.att_y_raw)))

    def build_rocket_geometry(self):
        lines = []
        r = 0.2
        for i in range(8):
            t1 = i * np.pi / 4
            t2 = (i+1) * np.pi / 4
            x1, y1 = r * np.cos(t1), r * np.sin(t1)
            x2, y2 = r * np.cos(t2), r * np.sin(t2)
            lines.extend([[x1, y1, 0.8], [x2, y2, 0.8]])   
            lines.extend([[x1, y1, -1.0], [x2, y2, -1.0]]) 
            lines.extend([[x1, y1, -1.0], [x1, y1, 0.8]])  
            lines.extend([[x1, y1, 0.8], [0, 0, 1.4]])     
        for i in range(4):
            t = i * np.pi / 2
            x, y = r * np.cos(t), r * np.sin(t)
            fx, fy = 0.6 * np.cos(t), 0.6 * np.sin(t)
            lines.extend([[x, y, -0.6], [fx, fy, -1.0]])
            lines.extend([[fx, fy, -1.0], [x, y, -1.0]])
            
        self.rocket_verts_local = np.array(lines)

        self.vec_roll_local = np.array([[0,0,0], [0,0,2.0]])   
        self.vec_pitch_local = np.array([[0,0,0], [0,2.0,0]])  
        self.vec_yaw_local = np.array([[0,0,0], [2.0,0,0]])    

        pos_scale = self.max_range * 0.15 
        self.pos_vec_roll_local = np.array([[0,0,0], [0,0,pos_scale]])
        self.pos_vec_pitch_local = np.array([[0,0,0], [0,pos_scale,0]])
        self.pos_vec_yaw_local = np.array([[0,0,0], [pos_scale,0,0]])

    def calculate_rotations(self):
        if self.mode == 'quat':
            self.rot_matrices = R.from_quat(self.q_data).as_matrix()
        else:
            angles = np.column_stack((self.raw_yaw * self.inv_y, self.raw_pitch * self.inv_p, self.raw_roll * self.inv_r))
            self.rot_matrices = R.from_euler(self.euler_seq, angles).as_matrix()

        self.pre_rocket = np.einsum('vm,fcm->fvc', self.rocket_verts_local, self.rot_matrices)
        self.pre_roll = np.einsum('vm,fcm->fvc', self.vec_roll_local, self.rot_matrices)
        self.pre_pitch = np.einsum('vm,fcm->fvc', self.vec_pitch_local, self.rot_matrices)
        self.pre_yaw = np.einsum('vm,fcm->fvc', self.vec_yaw_local, self.rot_matrices)
        
        self.pre_pos_roll_b = self.pos_data_baro[:, None, :] + np.einsum('vm,fcm->fvc', self.pos_vec_roll_local, self.rot_matrices)
        self.pre_pos_pitch_b = self.pos_data_baro[:, None, :] + np.einsum('vm,fcm->fvc', self.pos_vec_pitch_local, self.rot_matrices)
        self.pre_pos_yaw_b = self.pos_data_baro[:, None, :] + np.einsum('vm,fcm->fvc', self.pos_vec_yaw_local, self.rot_matrices)

        self.pre_pos_roll_f = self.pos_data_fused[:, None, :] + np.einsum('vm,fcm->fvc', self.pos_vec_roll_local, self.rot_matrices)
        self.pre_pos_pitch_f = self.pos_data_fused[:, None, :] + np.einsum('vm,fcm->fvc', self.pos_vec_pitch_local, self.rot_matrices)
        self.pre_pos_yaw_f = self.pos_data_fused[:, None, :] + np.einsum('vm,fcm->fvc', self.pos_vec_yaw_local, self.rot_matrices)

    def create_view_panel(self, title, widget, btn_home_callback, toggle_callback=None):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5)

        header_layout = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: bold; font-size: 16px; color: #EEEEEE;")
        header_layout.addWidget(lbl)

        if toggle_callback:
            btn_toggle = QPushButton(f"Mode: {self.middle_mode.upper()}" if 'Position' in title else f"Mode: {self.right_mode.upper()}")
            btn_toggle.setStyleSheet("color: red; font-weight: bold;")
            btn_toggle.clicked.connect(lambda: toggle_callback(btn_toggle))
            header_layout.addWidget(btn_toggle)

            if 'Position' in title:
                self.btn_middle_toggle = btn_toggle
            elif 'Altitude' in title:
                self.btn_right_toggle = btn_toggle

        btn_home = QPushButton("Home")
        btn_home.clicked.connect(btn_home_callback)
        header_layout.addWidget(btn_home)
        
        layout.addLayout(header_layout)
        layout.addWidget(widget, stretch=1)
        return container

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        title_lbl = QLabel("SPEEDY 2 FLIGHT VISUALISATION")
        title_lbl.setStyleSheet("color: red; font-size: 24px; font-weight: bold;")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_lbl)

        self.view_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.view_splitter, stretch=1)

        # =========================================
        # 1. Left Panel (3D Attitude + 2D Attitude)
        # =========================================
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 3D Attitude
        self.view_local = gl.GLViewWidget()
        self.view_local.opts['center'] = QVector3D(0, 0, 0)
        self.view_local.opts['distance'] = 5
        
        self.gl_rocket = gl.GLLinePlotItem(color=(1, 1, 1, 1), mode='lines', width=2, antialias=True)
        self.gl_roll = gl.GLLinePlotItem(color=(1, 0, 0, 1), mode='lines', width=3)
        self.gl_pitch = gl.GLLinePlotItem(color=(0, 1, 0, 1), mode='lines', width=3)
        self.gl_yaw = gl.GLLinePlotItem(color=(0, 0, 1, 1), mode='lines', width=3)
        self.gl_vel_dir = gl.GLLinePlotItem(color=(1, 0, 1, 1), mode='lines', width=2)
        
        for item in [self.gl_rocket, self.gl_roll, self.gl_pitch, self.gl_yaw, self.gl_vel_dir]:
            self.view_local.addItem(item)
            
        grid_local = gl.GLGridItem()
        grid_local.scale(0.2, 0.2, 0.2)
        self.view_local.addItem(grid_local)

        panel_local = self.create_view_panel("3D Attitude", self.view_local, self.home_local)
        left_splitter.addWidget(panel_local)

        # 2D Attitude
        pg.setConfigOptions(antialias=True)
        self.plot_att = pg.PlotWidget()
        self.plot_att.setLabel('left', 'Angle (°)')
        self.plot_att.setLabel('bottom', 'Time (s)')
        self.plot_att.showGrid(x=True, y=True)
        self.plot_att.addLegend(offset=(-10, 10))

        self.curve_att_r = self.plot_att.plot(self.time_data, self.att_r_raw, pen=pg.mkPen('red', width=2), name="Roll")
        self.curve_att_p = self.plot_att.plot(self.time_data, self.att_p_raw, pen=pg.mkPen('green', width=2), name="Pitch")
        self.curve_att_y = self.plot_att.plot(self.time_data, self.att_y_raw, pen=pg.mkPen('blue', width=2), name="Yaw")
        self.att_vline = pg.PlotDataItem(pen=pg.mkPen('red', width=2))
        self.plot_att.addItem(self.att_vline)

        # Attitude 2D Controls
        att_control_widget = QWidget()
        att_clayout = QHBoxLayout(att_control_widget)
        att_clayout.setContentsMargins(0,0,0,0)
        
        self.chk_att_r = QCheckBox("Roll (R)")
        self.chk_att_p = QCheckBox("Pitch (G)")
        self.chk_att_y = QCheckBox("Yaw (B)")
        self.chk_att_unwrap = QCheckBox("Unwrap Angles")
        
        self.chk_att_r.setChecked(True)
        self.chk_att_p.setChecked(True)
        self.chk_att_y.setChecked(True)

        self.chk_att_r.stateChanged.connect(self.update_att_graph)
        self.chk_att_p.stateChanged.connect(self.update_att_graph)
        self.chk_att_y.stateChanged.connect(self.update_att_graph)
        self.chk_att_unwrap.stateChanged.connect(self.update_att_graph)

        for chk in [self.chk_att_r, self.chk_att_p, self.chk_att_y, self.chk_att_unwrap]:
            chk.setStyleSheet("color: #EEEEEE;")
            att_clayout.addWidget(chk)

        att_container = QVBoxLayout()
        att_container.addWidget(self.plot_att, stretch=1)
        att_container.addWidget(att_control_widget)
        
        att_panel_widget = QWidget()
        att_panel_widget.setLayout(att_container)
        panel_att_2d = self.create_view_panel("2D Attitude", att_panel_widget, lambda: self.plot_att.autoRange())
        left_splitter.addWidget(panel_att_2d)

        self.view_splitter.addWidget(left_splitter)


        # =========================================
        # 2. Middle Panel (Global Position)
        # =========================================
        self.view_global = gl.GLViewWidget()
        self.view_global.opts['center'] = QVector3D(self.mids[0], self.mids[1], self.mids[2])
        self.view_global.opts['distance'] = self.max_range * 2.5

        self.gl_path_baro = gl.GLLinePlotItem(pos=self.pos_data_baro, color=(1, 0.5, 0, 1.0), mode='line_strip', width=1.5)
        self.gl_path_fused = gl.GLLinePlotItem(pos=self.pos_data_fused, color=(0, 1, 1, 0.3), mode='line_strip', width=1.5)
        
        self.gl_pos_scatter = gl.GLScatterPlotItem(color=(1, 1, 1, 1), size=10)
        self.gl_pos_roll = gl.GLLinePlotItem(color=(1, 0, 0, 1), mode='lines', width=2)
        self.gl_pos_pitch = gl.GLLinePlotItem(color=(0, 1, 0, 1), mode='lines', width=2)
        self.gl_pos_yaw = gl.GLLinePlotItem(color=(0, 0, 1, 1), mode='lines', width=2)
        
        for item in [self.gl_path_baro, self.gl_path_fused, self.gl_pos_scatter, self.gl_pos_roll, self.gl_pos_pitch, self.gl_pos_yaw]:
            self.view_global.addItem(item)
            
        grid_global = gl.GLGridItem()
        grid_global.scale(self.max_range/10, self.max_range/10, self.max_range/10)
        grid_global.translate(self.mids[0], self.mids[1], 0)
        self.view_global.addItem(grid_global)

        panel_global = self.create_view_panel("Position", self.view_global, self.home_global, self.toggle_middle_alt)
        self.view_splitter.addWidget(panel_global)


        # =========================================
        # 3. Right Panel (Altitude 2D)
        # =========================================
        self.plot_alt = pg.PlotWidget()
        self.plot_alt.setLabel('left', 'Altitude (m)')
        self.plot_alt.setLabel('bottom', 'Time (s)')
        self.plot_alt.showGrid(x=True, y=True)
        self.plot_alt.addLegend(offset=(-10, 10))
        
        self.curve_baro = self.plot_alt.plot(self.time_data, self.pos_data_baro[:, 2], pen=pg.mkPen('orange', width=2), name="Baro")
        self.curve_fused = self.plot_alt.plot(self.time_data, self.pos_data_fused[:, 2], pen=pg.mkPen('cyan', width=1, style=Qt.PenStyle.DashLine), name="Fused")
        
        self.alt_vline = pg.PlotDataItem(pen=pg.mkPen('red', width=2))
        self.plot_alt.addItem(self.alt_vline)

        panel_alt = self.create_view_panel("Altitude vs Time", self.plot_alt, self.home_alt, self.toggle_right_alt)
        self.view_splitter.addWidget(panel_alt)

        self.view_splitter.setSizes([500, 600, 500])

        # Event Markers Initialization
        self.setup_event_markers()

        # =========================================
        # -- Control Panel Bottom --
        # =========================================
        control_container = QWidget()
        control_layout = QVBoxLayout(control_container)
        control_layout.setContentsMargins(0, 5, 0, 0)
        main_layout.addWidget(control_container)

        settings_layout = QHBoxLayout()
        
        cam_group = QGroupBox("Camera Position")
        cam_group.setStyleSheet("color: red; font-weight: bold;")
        cam_layout = QHBoxLayout(cam_group)
        self.btn_top = QPushButton("Top")
        self.btn_front = QPushButton("Front")
        self.btn_side = QPushButton("Side")
        self.btn_top.clicked.connect(lambda: self.set_camera(90, 0))
        self.btn_front.clicked.connect(lambda: self.set_camera(0, 90))
        self.btn_side.clicked.connect(lambda: self.set_camera(0, 0))
        for btn in [self.btn_top, self.btn_front, self.btn_side]:
            cam_layout.addWidget(btn)
        settings_layout.addWidget(cam_group)

        self.euler_group = QGroupBox("Euler Corrections (Disabled if Quat)")
        self.euler_group.setStyleSheet("color: red; font-weight: bold;")
        euler_layout = QHBoxLayout(self.euler_group)
        
        self.combo_seq = QComboBox()
        self.combo_seq.addItems(["ZYX", "XYZ", "YXZ", "ZXY", "XZY", "YZX"])
        self.combo_seq.setCurrentText(self.euler_seq)
        self.combo_seq.currentTextChanged.connect(self.on_euler_changed)
        
        lbl_seq = QLabel("Sequence:")
        lbl_seq.setStyleSheet("color: #EEEEEE;")
        euler_layout.addWidget(lbl_seq)
        euler_layout.addWidget(self.combo_seq)
        
        self.chk_r = QCheckBox("Invert Roll")
        self.chk_p = QCheckBox("Invert Pitch")
        self.chk_y = QCheckBox("Invert Yaw")
        self.chk_r.stateChanged.connect(self.on_euler_changed)
        self.chk_p.stateChanged.connect(self.on_euler_changed)
        self.chk_y.stateChanged.connect(self.on_euler_changed)
        
        for chk in [self.chk_r, self.chk_p, self.chk_y]:
            chk.setStyleSheet("color: #EEEEEE;")
            euler_layout.addWidget(chk)
            
        settings_layout.addWidget(self.euler_group)
        
        self.chk_markers = QCheckBox("Show Graph Event Markers")
        self.chk_markers.setStyleSheet("color: #EEEEEE; font-weight: bold;")
        self.chk_markers.setChecked(True)
        self.chk_markers.stateChanged.connect(self.toggle_event_markers)
        settings_layout.addWidget(self.chk_markers)

        settings_layout.addStretch()
        control_layout.addLayout(settings_layout)

        slider_wrapper = QVBoxLayout()
        slider_layout = QHBoxLayout()
        
        lbl_tl = QLabel("Timeline:")
        lbl_tl.setStyleSheet("color: red; font-weight: bold;")
        slider_layout.addWidget(lbl_tl)
        
        self.slider = MarkerSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.frames_total - 1)
        self.slider.setValue(0)
        self.slider.setStyleSheet("""
            QSlider::handle:horizontal { background: red; width: 12px; margin: -5px 0; border-radius: 6px; }
            QSlider::sub-page:horizontal { background: #ff7777; }
            QSlider::add-page:horizontal { background: #cccccc; }
        """)
        self.slider.valueChanged.connect(self.on_slider_moved)
        slider_layout.addWidget(self.slider)
        slider_wrapper.addLayout(slider_layout)

        self.timeline_labels = TimelineLabels()
        self.timeline_labels.set_markers(self.state_markers, self.frames_total)
        self.slider.set_markers(self.state_markers)
        
        lbl_spacer = QHBoxLayout()
        lbl_spacer.addSpacing(70) 
        lbl_spacer.addWidget(self.timeline_labels)
        slider_wrapper.addLayout(lbl_spacer)
        
        control_layout.addLayout(slider_wrapper)

        btn_layout = QHBoxLayout()

        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        for b in [self.btn_prev, self.btn_next]:
            b.setAutoRepeat(True)
            b.setAutoRepeatDelay(400)
            b.setAutoRepeatInterval(int(1000 / (self.base_fps * 0.25)))
        self.btn_prev.clicked.connect(lambda: self.step_frame(-1))
        self.btn_next.clicked.connect(lambda: self.step_frame(1))

        self.btn_play = QPushButton("Pause")
        self.btn_play.setStyleSheet("color: red; font-weight: bold;")
        self.btn_play.clicked.connect(self.toggle_play)
        
        btn_layout.addWidget(self.btn_prev)
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_next)
        
        self.btn_load = QPushButton("Load New File")
        self.btn_load.clicked.connect(self.load_new_file)
        btn_layout.addWidget(self.btn_load)

        btn_layout.addStretch()

        self.lbl_speed = QLabel("Speed: 1.0x")
        self.lbl_speed.setStyleSheet("color: red; font-weight: bold;")
        btn_layout.addWidget(self.lbl_speed)
        
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setMinimum(1)
        self.slider_speed.setMaximum(50)
        self.slider_speed.setValue(10)
        self.slider_speed.setStyleSheet("""
            QSlider::handle:horizontal { background: red; width: 10px; margin: -4px 0; border-radius: 5px;}
            QSlider::sub-page:horizontal { background: #ffaaaa; }
        """)
        self.slider_speed.valueChanged.connect(self.on_speed_changed)
        self.slider_speed.setFixedWidth(150)
        btn_layout.addWidget(self.slider_speed)

        self.btn_reset_speed = QPushButton("1x")
        self.btn_reset_speed.clicked.connect(lambda: self.slider_speed.setValue(10))
        btn_layout.addWidget(self.btn_reset_speed)
        
        btn_layout.addStretch()
        
        self.lbl_status = QLabel("Time: 0.00s | Frame: 0/0")
        self.lbl_status.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
        btn_layout.addWidget(self.lbl_status)

        control_layout.addLayout(btn_layout)

    def setup_event_markers(self):
        self.event_markers_items = []
        for idx in self.state_markers.keys():
            t = self.time_data[idx]
            
            # Altitude 2D
            la = pg.InfiniteLine(pos=t, angle=90, movable=False, pen=pg.mkPen(color='gray', width=1, style=Qt.PenStyle.DashLine))
            self.plot_alt.addItem(la)
            self.event_markers_items.append(la)
            
            # Attitude 2D
            lt = pg.InfiniteLine(pos=t, angle=90, movable=False, pen=pg.mkPen(color='gray', width=1, style=Qt.PenStyle.DashLine))
            self.plot_att.addItem(lt)
            self.event_markers_items.append(lt)

            # Global 3D Position
            px, py, pz = self.pos_data_fused[idx]
            line_3d = gl.GLLinePlotItem(pos=np.array([[px, py, self.min_z_fused], [px, py, pz + 100]]), color=(0.7, 0.7, 0.7, 0.8), width=1)
            self.view_global.addItem(line_3d)
            self.event_markers_items.append(line_3d)

    def toggle_event_markers(self, state):
        visible = bool(state)
        for item in self.event_markers_items:
            item.setVisible(visible)

    def update_att_graph(self):
        self.unwrap_angles = self.chk_att_unwrap.isChecked()
        
        r_data = self.att_r_unwrap if self.unwrap_angles else self.att_r_raw
        p_data = self.att_p_unwrap if self.unwrap_angles else self.att_p_raw
        y_data = self.att_y_unwrap if self.unwrap_angles else self.att_y_raw

        self.curve_att_r.setData(self.time_data, r_data)
        self.curve_att_p.setData(self.time_data, p_data)
        self.curve_att_y.setData(self.time_data, y_data)
        
        self.curve_att_r.setVisible(self.chk_att_r.isChecked())
        self.curve_att_p.setVisible(self.chk_att_p.isChecked())
        self.curve_att_y.setVisible(self.chk_att_y.isChecked())
        
        self.plot_att.autoRange()

    def step_frame(self, amount):
        if self.is_playing:
            self.toggle_play()
        
        self.current_frame = np.clip(self.current_frame + amount, 0, self.frames_total - 1)
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)
        self.render_frame()

    def home_local(self):
        self.view_local.opts['center'] = QVector3D(0, 0, 0)
        self.view_local.setCameraPosition(distance=5, elevation=30, azimuth=45)

    def home_global(self):
        self.view_global.opts['center'] = QVector3D(self.mids[0], self.mids[1], self.mids[2])
        self.view_global.setCameraPosition(distance=self.max_range * 2.5, elevation=30, azimuth=45)

    def home_alt(self):
        self.plot_alt.autoRange()

    def toggle_middle_alt(self, btn):
        self.middle_mode = 'fused' if self.middle_mode == 'baro' else 'baro'
        btn.setText(f"Mode: {self.middle_mode.upper()}")
        
        if self.middle_mode == 'baro':
            self.gl_path_baro.setData(color=(1, 0.5, 0, 1.0))
            self.gl_path_fused.setData(color=(0, 1, 1, 0.3))
        else:
            self.gl_path_baro.setData(color=(1, 0.5, 0, 0.3))
            self.gl_path_fused.setData(color=(0, 1, 1, 1.0))
        self.render_frame()

    def toggle_right_alt(self, btn):
        self.right_mode = 'fused' if self.right_mode == 'baro' else 'baro'
        btn.setText(f"Mode: {self.right_mode.upper()}")
        
        if self.right_mode == 'baro':
            self.curve_baro.setPen(pg.mkPen('orange', width=2))
            self.curve_fused.setPen(pg.mkPen('cyan', width=1, style=Qt.PenStyle.DashLine))
            self.curve_fused.setZValue(-1)
            self.curve_baro.setZValue(1)
        else:
            self.curve_fused.setPen(pg.mkPen('cyan', width=2))
            self.curve_baro.setPen(pg.mkPen('orange', width=1, style=Qt.PenStyle.DashLine))
            self.curve_baro.setZValue(-1)
            self.curve_fused.setZValue(1)
        self.render_frame()

    def update_ui_state(self):
        self.euler_group.setEnabled(self.mode == 'euler')

    def set_camera(self, elevation, azimuth):
        dist_local = self.view_local.opts['distance']
        self.view_local.setCameraPosition(distance=dist_local, elevation=elevation, azimuth=azimuth)
        
        dist_global = self.view_global.opts['distance']
        self.view_global.setCameraPosition(distance=dist_global, elevation=elevation, azimuth=azimuth)

    def on_euler_changed(self):
        if self.mode != 'euler': return
        self.euler_seq = self.combo_seq.currentText()
        self.inv_r = -1 if self.chk_r.isChecked() else 1
        self.inv_p = -1 if self.chk_p.isChecked() else 1
        self.inv_y = -1 if self.chk_y.isChecked() else 1
        self.calculate_2d_attitude()
        self.calculate_rotations()
        self.update_att_graph()
        self.render_frame()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.setText("Pause" if self.is_playing else "Play")

    def on_slider_moved(self, value):
        self.current_frame = value
        self.render_frame()

    def on_speed_changed(self, value):
        self.speed_multiplier = value / 10.0
        self.lbl_speed.setText(f"Speed: {self.speed_multiplier:.1f}x")
        new_interval = int(1000 / (self.base_fps * self.speed_multiplier))
        self.timer.setInterval(max(1, new_interval))

    def update_frame(self):
        if not self.is_playing: return
        self.current_frame += 1
        if self.current_frame >= self.frames_total:
            self.current_frame = 0
            
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)
        self.render_frame()

    def render_frame(self):
        frame = self.current_frame

        self.gl_rocket.setData(pos=self.pre_rocket[frame])
        self.gl_roll.setData(pos=self.pre_roll[frame])
        self.gl_pitch.setData(pos=self.pre_pitch[frame])
        self.gl_yaw.setData(pos=self.pre_yaw[frame])
        self.gl_vel_dir.setData(pos=np.array([[0, 0, 0], self.vel_dirs[frame]]))

        if self.middle_mode == 'baro':
            curr_pos = self.pos_data_baro[frame:frame+1]
            self.gl_pos_scatter.setData(pos=curr_pos)
            self.gl_pos_roll.setData(pos=self.pre_pos_roll_b[frame])
            self.gl_pos_pitch.setData(pos=self.pre_pos_pitch_b[frame])
            self.gl_pos_yaw.setData(pos=self.pre_pos_yaw_b[frame])
        else:
            curr_pos = self.pos_data_fused[frame:frame+1]
            self.gl_pos_scatter.setData(pos=curr_pos)
            self.gl_pos_roll.setData(pos=self.pre_pos_roll_f[frame])
            self.gl_pos_pitch.setData(pos=self.pre_pos_pitch_f[frame])
            self.gl_pos_yaw.setData(pos=self.pre_pos_yaw_f[frame])

        t = self.time_data[frame]
        
        # Altitude 2D Marker Update
        if self.right_mode == 'baro':
            active_z = self.pos_data_baro[frame, 2]
            base_z = self.min_z_baro
        else:
            active_z = self.pos_data_fused[frame, 2]
            base_z = self.min_z_fused
            
        self.alt_vline.setData(x=[t, t], y=[base_z, active_z])

        # Attitude 2D Marker Update
        y_min, y_max = self.plot_att.viewRange()[1]
        self.att_vline.setData(x=[t, t], y=[y_min, y_max])

        self.lbl_status.setText(f"Flight Time: {t:.2f}s | Frame: {frame}/{self.frames_total}")

    def load_new_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Trajectory File", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if file_path:
            self.timer.stop()
            try:
                # Remove old markers
                for item in self.event_markers_items:
                    try:
                        self.plot_alt.removeItem(item)
                        self.plot_att.removeItem(item)
                        self.view_global.removeItem(item)
                    except:
                        pass
                self.event_markers_items.clear()

                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                df.columns = df.columns.str.strip()
                
                self.process_data(df)
                self.build_rocket_geometry()
                self.calculate_2d_attitude()
                self.calculate_rotations()
                self.update_att_graph()
                
                self.update_ui_state()
                self.slider.setMaximum(self.frames_total - 1)
                self.timeline_labels.set_markers(self.state_markers, self.frames_total)
                self.slider.set_markers(self.state_markers)

                self.middle_mode = 'baro'
                self.right_mode = 'baro'
                if hasattr(self, 'btn_middle_toggle'): self.btn_middle_toggle.setText("Mode: BARO")
                if hasattr(self, 'btn_right_toggle'): self.btn_right_toggle.setText("Mode: BARO")

                self.gl_path_baro.setData(pos=self.pos_data_baro, color=(1, 0.5, 0, 1.0))
                self.gl_path_fused.setData(pos=self.pos_data_fused, color=(0, 1, 1, 0.3))
                self.view_global.opts['center'] = QVector3D(self.mids[0], self.mids[1], self.mids[2])
                self.home_global()
                
                self.curve_baro.setData(self.time_data, self.pos_data_baro[:, 2], pen=pg.mkPen('orange', width=2))
                self.curve_fused.setData(self.time_data, self.pos_data_fused[:, 2], pen=pg.mkPen('cyan', width=1, style=Qt.PenStyle.DashLine))
                self.home_alt()
                
                self.setup_event_markers()
                self.toggle_event_markers(self.chk_markers.isChecked())

                self.setWindowTitle(f"Speedy 2 Flight Visualisation - {os.path.basename(file_path)}")
                self.current_frame = 0
                self.timer.start()

            except Exception as e:
                print(f"[ERROR] Failed to load new file: {e}")
                traceback.print_exc()

def run():
    app = QApplication(sys.argv)
    
    target_file, _ = QFileDialog.getOpenFileName(
        None, "Select Initial Trajectory File", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
    )
    if not target_file:
        sys.exit(0)

    try:
        if target_file.endswith('.csv'):
            df = pd.read_csv(target_file)
        else:
            df = pd.read_excel(target_file)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load data: {e}")
        traceback.print_exc()
        sys.exit(1)

    window = RocketVisualizerGUI(df, target_file)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    run()