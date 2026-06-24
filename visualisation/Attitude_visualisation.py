import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from tkinter import filedialog

# ==========================================
# CONFIGURATION
# ==========================================
DATA_FILE = r'C:\Users\kubfi\Downloads\Launch\processed_trajectory_final.xlsx'
LOGO_FILENAME = 'SPEEDY_LOGO_V2_red_transparent (1).png'
FRAME_SKIP = 5 

# Event indices (Excel Row Number)
LIFTOFF_ROW = 267
APOGEE_ROW = 516
TOUCHDOWN_ROW = 1197

# Column configuration
TIME_COL = 'Time_s'
ROLL_COL = 'Attitude_Roll_deg'
PITCH_COL = 'Attitude_Pitch_deg'
YAW_COL = 'Attitude_Yaw_deg'
POS_X_COL = 'Pos_X_m'
POS_Y_COL = 'Pos_Y_m'
ACCEL_ALT_COL = 'Pos_Z_m_Fused' 
BARO_ALT_COL = 'True Alt' 
STATE_COL = 'State' 

class RocketVisualizer:
    def __init__(self, df, file_path, icon_path):
        print("[DEBUG] Initializing visualizer...")
        self.df = df
        self.file_path = file_path
        self.icon_path = icon_path
        self.new_file_to_load = None
        
        required_cols = [TIME_COL, ROLL_COL, PITCH_COL, YAW_COL, POS_X_COL, POS_Y_COL, ACCEL_ALT_COL, BARO_ALT_COL]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"\n[ERROR] Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            print("Please fix the column names in the CONFIGURATION block.")
            sys.exit()

        self.time_data = df[TIME_COL].values
        self.frames_total = len(self.time_data)
        
        # Determine Dynamic Event Indices
        self.idx_liftoff = None
        self.idx_apogee = None
        self.idx_ch_deploy = None
        self.idx_touchdown = None

        if STATE_COL in df.columns:
            df[STATE_COL] = pd.to_numeric(df[STATE_COL], errors='coerce')
            def get_idx(val):
                idxs = df.index[df[STATE_COL] == val].tolist()
                return idxs[0] if len(idxs) > 0 else None
            self.idx_liftoff = get_idx(4)
            self.idx_apogee = get_idx(5)
            self.idx_ch_deploy = get_idx(6)
            self.idx_touchdown = get_idx(7)

        # Data extraction
        self.raw_roll = np.radians(df[ROLL_COL].values.copy())
        self.raw_pitch = np.radians(df[PITCH_COL].values.copy())
        self.raw_yaw = np.radians(df[YAW_COL].values.copy())
        self.pos_x = df[POS_X_COL].values.copy()
        self.pos_y = df[POS_Y_COL].values.copy()
        self.alt_fused = df[ACCEL_ALT_COL].values.copy()
        self.alt_baro = df[BARO_ALT_COL].values.copy()

        # FREEZE DATA AFTER TOUCHDOWN
        self.freeze_limit = self.idx_touchdown if self.idx_touchdown is not None else (self.frames_total - 1)

        if self.freeze_limit < self.frames_total:
            self.raw_roll[self.freeze_limit:] = self.raw_roll[self.freeze_limit]
            self.raw_pitch[self.freeze_limit:] = self.raw_pitch[self.freeze_limit]
            self.raw_yaw[self.freeze_limit:] = self.raw_yaw[self.freeze_limit]
            self.pos_x[self.freeze_limit:] = self.pos_x[self.freeze_limit]
            self.pos_y[self.freeze_limit:] = self.pos_y[self.freeze_limit]
            self.alt_fused[self.freeze_limit:] = self.alt_fused[self.freeze_limit]
            self.alt_baro[self.freeze_limit:] = self.alt_baro[self.freeze_limit]

        self.pos_data_fused = np.column_stack((self.pos_x, self.pos_y, self.alt_fused))
        self.pos_data_baro = np.column_stack((self.pos_x, self.pos_y, self.alt_baro))

        # Calculate Velocity Vectors for the Flight Path indicator
        self.vel_data_baro = np.gradient(self.pos_data_baro, self.time_data, axis=0)
        self.vel_data_fused = np.gradient(self.pos_data_fused, self.time_data, axis=0)
        
        sb = np.linalg.norm(self.vel_data_baro, axis=1)
        sf = np.linalg.norm(self.vel_data_fused, axis=1)
        sb[sb == 0] = 1e-6
        sf[sf == 0] = 1e-6
        
        # Scale direction vectors to match attitude vectors (length 2.0)
        self.vel_dirs_baro = (self.vel_data_baro.T / sb).T * 2.0
        self.vel_dirs_fused = (self.vel_data_fused.T / sf).T * 2.0

        # Zero out velocity before liftoff and after touchdown
        if self.idx_liftoff is not None:
            self.vel_dirs_baro[:self.idx_liftoff] = 0
            self.vel_dirs_fused[:self.idx_liftoff] = 0
        if self.freeze_limit < self.frames_total:
            self.vel_dirs_baro[self.freeze_limit:] = 0
            self.vel_dirs_fused[self.freeze_limit:] = 0

        self.inv_r = 1
        self.inv_p = 1
        self.inv_y = 1
        self.euler_seq = 'YXZ'
        self.is_playing = True
        self.active_path = 'baro' 
        
        self.speed_multiplier = 1.0
        self.current_frame_float = 0.0
        
        self.mappings = [
            ('Roll-Pitch-Yaw', self.raw_roll, self.raw_pitch, self.raw_yaw),
            ('Pitch-Roll-Yaw', self.raw_pitch, self.raw_roll, self.raw_yaw),
            ('Yaw-Pitch-Roll', self.raw_yaw, self.raw_pitch, self.raw_roll),
            ('Roll-Yaw-Pitch', self.raw_roll, self.raw_yaw, self.raw_pitch)
        ]
        self.map_idx = 0

        self.build_rocket_geometry()
        
        self.vec_roll_local = np.array([[0,0,0], [0,0,2.0]])   
        self.vec_pitch_local = np.array([[0,0,0], [0,2.0,0]])  
        self.vec_yaw_local = np.array([[0,0,0], [2.0,0,0]])    

        max_range = max([np.ptp(self.pos_data_baro[:, i]) for i in range(3)]) / 2.0
        if max_range < 1: max_range = 1.0
        pos_scale = max_range * 0.15 
        
        self.pos_vec_roll_local = np.array([[0,0,0], [0,0,pos_scale]])
        self.pos_vec_pitch_local = np.array([[0,0,0], [0,pos_scale,0]])
        self.pos_vec_yaw_local = np.array([[0,0,0], [pos_scale,0,0]])

        self.setup_plot()
        self.recalculate_rotations()
        
        print("[DEBUG] Starting animation loop...")
        self.anim = animation.FuncAnimation(
            self.fig, self.update_anim, interval=30, blit=False, cache_frame_data=False
        )
        plt.show()

    def build_rocket_geometry(self):
        lines = []
        r = 0.2
        for i in range(8):
            t1 = i * np.pi / 4
            t2 = (i+1) * np.pi / 4
            x1, y1 = r * np.cos(t1), r * np.sin(t1)
            x2, y2 = r * np.cos(t2), r * np.sin(t2)
            lines.append([[x1, y1, 0.8], [x2, y2, 0.8]])   
            lines.append([[x1, y1, -1.0], [x2, y2, -1.0]]) 
            lines.append([[x1, y1, -1.0], [x1, y1, 0.8]])  
            lines.append([[x1, y1, 0.8], [0, 0, 1.4]])     
        for i in range(4):
            t = i * np.pi / 2
            x, y = r * np.cos(t), r * np.sin(t)
            fx, fy = 0.6 * np.cos(t), 0.6 * np.sin(t)
            lines.append([[x, y, -0.6], [fx, fy, -1.0]])
            lines.append([[fx, fy, -1.0], [x, y, -1.0]])
        self.rocket_lines_local = np.array(lines)

    def recalculate_rotations(self):
        _, r_data, p_data, y_data = self.mappings[self.map_idx]
        angles = np.column_stack((y_data * self.inv_y, p_data * self.inv_p, r_data * self.inv_r))
        self.rot_matrices = R.from_euler(self.euler_seq, angles).as_matrix()

        self.pre_rocket = np.zeros((self.frames_total, len(self.rocket_lines_local), 2, 3))
        self.pre_roll = np.zeros((self.frames_total, 2, 3))
        self.pre_pitch = np.zeros((self.frames_total, 2, 3))
        self.pre_yaw = np.zeros((self.frames_total, 2, 3))

        self.pre_pos_baro_roll = np.zeros((self.frames_total, 2, 3))
        self.pre_pos_baro_pitch = np.zeros((self.frames_total, 2, 3))
        self.pre_pos_baro_yaw = np.zeros((self.frames_total, 2, 3))
        
        self.pre_pos_fused_roll = np.zeros((self.frames_total, 2, 3))
        self.pre_pos_fused_pitch = np.zeros((self.frames_total, 2, 3))
        self.pre_pos_fused_yaw = np.zeros((self.frames_total, 2, 3))

        for i in range(self.frames_total):
            rot = self.rot_matrices[i]
            
            self.pre_rocket[i] = np.dot(self.rocket_lines_local, rot.T)
            self.pre_roll[i] = np.dot(self.vec_roll_local, rot.T)
            self.pre_pitch[i] = np.dot(self.vec_pitch_local, rot.T)
            self.pre_yaw[i] = np.dot(self.vec_yaw_local, rot.T)
            
            b_pos = self.pos_data_baro[i]
            self.pre_pos_baro_roll[i] = b_pos + np.dot(self.pos_vec_roll_local, rot.T)
            self.pre_pos_baro_pitch[i] = b_pos + np.dot(self.pos_vec_pitch_local, rot.T)
            self.pre_pos_baro_yaw[i] = b_pos + np.dot(self.pos_vec_yaw_local, rot.T)

            f_pos = self.pos_data_fused[i]
            self.pre_pos_fused_roll[i] = f_pos + np.dot(self.pos_vec_roll_local, rot.T)
            self.pre_pos_fused_pitch[i] = f_pos + np.dot(self.pos_vec_pitch_local, rot.T)
            self.pre_pos_fused_yaw[i] = f_pos + np.dot(self.pos_vec_yaw_local, rot.T)

        if hasattr(self, 'slider'):
            self.draw_frame(int(self.slider.val))
            self.fig.canvas.draw_idle()

    def setup_plot(self):
        self.fig = plt.figure(figsize=(18, 9))
        
        try:
            self.fig.canvas.manager.set_window_title("Speedy 2 Flight Visualisation")
            if os.path.exists(self.icon_path):
                icon_img = tk.PhotoImage(file=self.icon_path)
                self.fig.canvas.manager.window.iconphoto(True, icon_img)
        except Exception as e:
            print(f"[DEBUG] Window manager settings failed: {e}")
            
        # Global Texts (Bottom Bar)
        self.brand_text = self.fig.text(0.5, 0.03, "Speedy 2 Flight Visualisation", ha='center', fontsize=14, color='red', fontweight='bold')
        self.time_text = self.fig.text(0.98, 0.03, "", ha='right', fontsize=12, fontweight='bold', color='black')
        self.file_text = self.fig.text(0.02, 0.03, f"Loaded: {os.path.basename(self.file_path)}", ha='left', fontsize=10, color='gray')

        self.ax_3d = self.fig.add_axes([0.01, 0.35, 0.31, 0.6], projection='3d')
        self.ax_3d.set_title('Local Attitude')
        self.rocket_collection = Line3DCollection(self.rocket_lines_local, colors='black', linewidths=1.5)
        self.ax_3d.add_collection3d(self.rocket_collection)

        self.line_roll, = self.ax_3d.plot([], [], [], color='red', lw=3, label='Roll')
        self.line_pitch, = self.ax_3d.plot([], [], [], color='green', lw=3, label='Pitch')
        self.line_yaw, = self.ax_3d.plot([], [], [], color='blue', lw=3, label='Yaw')
        
        # New Flight Path Vector
        self.line_path, = self.ax_3d.plot([], [], [], color='magenta', lw=2, linestyle='--', label='Path Dir')

        d = 1.8
        for ax_obj in [self.ax_3d]:
            ax_obj.text(0, d, 0, "Front", color='gray', ha='center')
            ax_obj.text(0, -d, 0, "Back", color='gray', ha='center')
            ax_obj.text(-d, 0, 0, "Left", color='gray', ha='center')
            ax_obj.text(d, 0, 0, "Right", color='gray', ha='center')
            ax_obj.text(0, 0, d, "Top", color='gray', ha='center')
            ax_obj.text(0, 0, -d, "Bottom", color='gray', ha='center')
            ax_obj.set_xlim([-1.5, 1.5])
            ax_obj.set_ylim([-1.5, 1.5])
            ax_obj.set_zlim([-1.5, 1.5])
            ax_obj.legend(loc='upper left', fontsize=8)

        self.ax_pos = self.fig.add_axes([0.34, 0.35, 0.31, 0.6], projection='3d')
        self.ax_pos.set_title('Global Position & Orientation')
        
        limit = self.freeze_limit + 1
        self.path_baro, = self.ax_pos.plot(self.pos_data_baro[:limit, 0], self.pos_data_baro[:limit, 1], self.pos_data_baro[:limit, 2], 
                                           color='orange', alpha=1.0, lw=1.5, label='Baro Path')
        self.path_fused, = self.ax_pos.plot(self.pos_data_fused[:limit, 0], self.pos_data_fused[:limit, 1], self.pos_data_fused[:limit, 2], 
                                            color='cyan', alpha=0.2, lw=1.5, label='Accel Path')
        
        self.pos_scatter, = self.ax_pos.plot([], [], [], marker='o', color='black', markersize=6)
        self.pos_line_roll, = self.ax_pos.plot([], [], [], color='red', lw=2)
        self.pos_line_pitch, = self.ax_pos.plot([], [], [], color='green', lw=2)
        self.pos_line_yaw, = self.ax_pos.plot([], [], [], color='blue', lw=2)
        
        mids = [np.mean([np.min(self.pos_data_baro[:, i]), np.max(self.pos_data_baro[:, i])]) for i in range(3)]
        max_range = max([np.ptp(self.pos_data_baro[:, i]) for i in range(3)]) / 2.0
        if max_range == 0: max_range = 1
        self.ax_pos.set_xlim(mids[0] - max_range, mids[0] + max_range)
        self.ax_pos.set_ylim(mids[1] - max_range, mids[1] + max_range)
        self.ax_pos.set_zlim(mids[2] - max_range, mids[2] + max_range)
        self.ax_pos.legend(loc='upper right', fontsize=8)

        ax_btn_path = self.fig.add_axes([0.43, 0.31, 0.12, 0.04])
        self.btn_path = Button(ax_btn_path, 'Toggle Alt Mode (Baro)')
        self.btn_path.on_clicked(self.toggle_path)

        self.ax_alt = self.fig.add_axes([0.67, 0.35, 0.31, 0.55])
        self.ax_alt.plot(self.time_data[:limit], self.alt_baro[:limit], color='black', lw=1.5, label='Baro Altitude')
        self.ax_alt.set_title('Barometric Altitude vs Time')
        self.ax_alt.set_xlabel('Time (s)')
        self.ax_alt.set_ylabel('Altitude (m)')
        self.ax_alt.grid(True)
        
        self.alt_vline, = self.ax_alt.plot([], [], color='red', lw=2)

        # Timeline Slider & Faint 1-Second Timestamps
        self.ax_slider = self.fig.add_axes([0.1, 0.22, 0.8, 0.03])
        self.slider = Slider(self.ax_slider, 'Time', 0, self.frames_total - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)

        self.ax_slider.xaxis.set_visible(True)
        max_sec = int(np.ceil(self.time_data[-1]))
        tick_frames = []
        tick_labels = []
        for s in range(0, max_sec + 1, 1):
            idx = (np.abs(self.time_data - s)).argmin()
            tick_frames.append(idx)
            tick_labels.append(str(s))
        self.ax_slider.set_xticks(tick_frames)
        self.ax_slider.set_xticklabels(tick_labels, color='gray', fontsize=7)
        self.ax_slider.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, colors='gray')
        for spine in self.ax_slider.spines.values():
            spine.set_visible(False)

        # Draw Dynamic Event Markers 
        events = [
            (self.idx_liftoff, 'Liftoff', 'red'),
            (self.idx_apogee, 'Apogee', 'red'),
            (self.idx_ch_deploy, 'CH Deploy', 'red'),
            (self.idx_touchdown, 'Touchdown', 'red')
        ]

        for idx, label, color in events:
            if idx is not None and idx < self.frames_total:
                # Slider
                self.ax_slider.axvline(idx, color=color, lw=2)
                self.ax_slider.text(idx, 1.05, label, color=color, fontsize=9, transform=self.ax_slider.get_xaxis_transform(), ha='center')
                # Altitude Graph
                t_val = self.time_data[idx]
                alt_val = self.alt_baro[idx]
                self.ax_alt.scatter([t_val], [alt_val], color=color, zorder=5)
                self.ax_alt.text(t_val, alt_val + 5, f' {label}', color=color, fontsize=9, zorder=5)
                # Position Graph
                x_b, y_b, z_b = self.pos_data_baro[idx]
                self.ax_pos.scatter([x_b], [y_b], [z_b], color=color, s=30, zorder=5)
                self.ax_pos.text(x_b, y_b, z_b, f' {label}', color=color, fontsize=8, zorder=5)

        self.ax_speed = self.fig.add_axes([0.1, 0.15, 0.15, 0.03])
        self.slider_speed = Slider(self.ax_speed, 'Speed', 0.1, 5.0, valinit=1.0, valstep=0.1)
        self.slider_speed.on_changed(self.on_speed_change)

        ax_btn_1x = self.fig.add_axes([0.27, 0.15, 0.05, 0.03])
        self.btn_1x = Button(ax_btn_1x, '1x')
        self.btn_1x.on_clicked(self.reset_speed)

        ax_btn_m5 = self.fig.add_axes([0.34, 0.15, 0.04, 0.03])
        self.btn_m5 = Button(ax_btn_m5, '-5 f')
        self.btn_m5.on_clicked(lambda e: self.step_frame(-5))
        ax_btn_m1 = self.fig.add_axes([0.39, 0.15, 0.04, 0.03])
        self.btn_m1 = Button(ax_btn_m1, '-1 f')
        self.btn_m1.on_clicked(lambda e: self.step_frame(-1))
        ax_btn_p1 = self.fig.add_axes([0.44, 0.15, 0.04, 0.03])
        self.btn_p1 = Button(ax_btn_p1, '+1 f')
        self.btn_p1.on_clicked(lambda e: self.step_frame(1))
        ax_btn_p5 = self.fig.add_axes([0.49, 0.15, 0.04, 0.03])
        self.btn_p5 = Button(ax_btn_p5, '+5 f')
        self.btn_p5.on_clicked(lambda e: self.step_frame(5))

        ax_btn_load = self.fig.add_axes([0.80, 0.15, 0.10, 0.03])
        self.btn_load = Button(ax_btn_load, 'Load New File')
        self.btn_load.on_clicked(self.load_new_file)

        ax_view_top = self.fig.add_axes([0.01, 0.8, 0.04, 0.04])
        self.btn_top = Button(ax_view_top, 'Top')
        self.btn_top.on_clicked(lambda e: self.set_camera(90, 0))
        ax_view_front = self.fig.add_axes([0.01, 0.75, 0.04, 0.04])
        self.btn_front = Button(ax_view_front, 'Front')
        self.btn_front.on_clicked(lambda e: self.set_camera(0, 90))
        ax_view_side = self.fig.add_axes([0.01, 0.70, 0.04, 0.04])
        self.btn_side = Button(ax_view_side, 'Side')
        self.btn_side.on_clicked(lambda e: self.set_camera(0, 0))

        ax_play = self.fig.add_axes([0.1, 0.08, 0.1, 0.05])
        self.btn_play = Button(ax_play, 'Pause')
        self.btn_play.on_clicked(self.toggle_play)

        ax_map = self.fig.add_axes([0.25, 0.08, 0.2, 0.05])
        self.btn_map = Button(ax_map, f'Map: {self.mappings[self.map_idx][0]}')
        self.btn_map.on_clicked(self.cycle_mapping)

        ax_inv = self.fig.add_axes([0.55, 0.05, 0.15, 0.1])
        self.chk_inv = CheckButtons(ax_inv, ['Invert Roll', 'Invert Pitch', 'Invert Yaw'], [False, False, False])
        self.chk_inv.on_clicked(self.toggle_inversion)

        ax_euler = self.fig.add_axes([0.75, 0.05, 0.15, 0.1])
        self.rad_euler = RadioButtons(ax_euler, ['ZYX', 'XYZ', 'YXZ'], active=2)
        self.rad_euler.on_clicked(self.change_euler)

    def load_new_file(self, event):
        root = tk.Tk()
        root.withdraw()
        new_file = filedialog.askopenfilename(
            title="Select the Trajectory File",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if new_file:
            self.new_file_to_load = new_file
            plt.close(self.fig)

    def set_camera(self, elev, azim):
        self.ax_3d.view_init(elev=elev, azim=azim)
        self.ax_pos.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()

    def toggle_path(self, event):
        if self.active_path == 'baro':
            self.active_path = 'fused'
            self.path_baro.set_alpha(0.2)
            self.path_fused.set_alpha(1.0)
            self.btn_path.label.set_text('Toggle Alt Mode (Accel)')
        else:
            self.active_path = 'baro'
            self.path_baro.set_alpha(1.0)
            self.path_fused.set_alpha(0.2)
            self.btn_path.label.set_text('Toggle Alt Mode (Baro)')
        
        self.draw_frame(int(self.slider.val))
        self.fig.canvas.draw_idle()

    def step_frame(self, amount):
        frame = int(self.slider.val) + amount
        frame = max(0, min(self.frames_total - 1, frame))
        self.current_frame_float = float(frame)
        self.slider.set_val(frame)

    def toggle_play(self, event):
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Pause' if self.is_playing else 'Play')
        self.fig.canvas.draw_idle()

    def on_speed_change(self, val):
        self.speed_multiplier = float(val)

    def reset_speed(self, event):
        self.slider_speed.set_val(1.0)

    def on_slider_change(self, val):
        frame = int(val)
        self.current_frame_float = float(val)
        self.draw_frame(frame)
        if not self.is_playing:
            self.fig.canvas.draw_idle()

    def cycle_mapping(self, event):
        self.map_idx = (self.map_idx + 1) % len(self.mappings)
        self.btn_map.label.set_text(f'Map: {self.mappings[self.map_idx][0]}')
        self.recalculate_rotations()

    def toggle_inversion(self, label):
        states = self.chk_inv.get_status()
        self.inv_r = -1 if states[0] else 1
        self.inv_p = -1 if states[1] else 1
        self.inv_y = -1 if states[2] else 1
        self.recalculate_rotations()

    def change_euler(self, label):
        self.euler_seq = label
        self.recalculate_rotations()

    def update_anim(self, _):
        if not self.is_playing:
            return
            
        self.current_frame_float += FRAME_SKIP * self.speed_multiplier
        if self.current_frame_float >= self.frames_total:
            self.current_frame_float = 0
            
        frame = int(self.current_frame_float)
        
        self.slider.eventson = False
        self.slider.set_val(frame)
        self.slider.eventson = True
        
        self.draw_frame(frame)

    def draw_frame(self, frame):
        self.rocket_collection.set_segments(self.pre_rocket[frame])

        self.line_roll.set_data(self.pre_roll[frame, :, 0], self.pre_roll[frame, :, 1])
        self.line_roll.set_3d_properties(self.pre_roll[frame, :, 2])

        self.line_pitch.set_data(self.pre_pitch[frame, :, 0], self.pre_pitch[frame, :, 1])
        self.line_pitch.set_3d_properties(self.pre_pitch[frame, :, 2])

        self.line_yaw.set_data(self.pre_yaw[frame, :, 0], self.pre_yaw[frame, :, 1])
        self.line_yaw.set_3d_properties(self.pre_yaw[frame, :, 2])
        
        if self.active_path == 'baro':
            data_arr = self.pos_data_baro
            roll_arr = self.pre_pos_baro_roll
            pitch_arr = self.pre_pos_baro_pitch
            yaw_arr = self.pre_pos_baro_yaw
            v_dir = self.vel_dirs_baro[frame]
        else:
            data_arr = self.pos_data_fused
            roll_arr = self.pre_pos_fused_roll
            pitch_arr = self.pre_pos_fused_pitch
            yaw_arr = self.pre_pos_fused_yaw
            v_dir = self.vel_dirs_fused[frame]

        # Draw Flight Path Vector in Local Attitude 
        self.line_path.set_data([0, v_dir[0]], [0, v_dir[1]])
        self.line_path.set_3d_properties([0, v_dir[2]])

        self.pos_scatter.set_data(data_arr[frame:frame+1, 0], data_arr[frame:frame+1, 1])
        self.pos_scatter.set_3d_properties(data_arr[frame:frame+1, 2])

        self.pos_line_roll.set_data(roll_arr[frame, :, 0], roll_arr[frame, :, 1])
        self.pos_line_roll.set_3d_properties(roll_arr[frame, :, 2])

        self.pos_line_pitch.set_data(pitch_arr[frame, :, 0], pitch_arr[frame, :, 1])
        self.pos_line_pitch.set_3d_properties(pitch_arr[frame, :, 2])

        self.pos_line_yaw.set_data(yaw_arr[frame, :, 0], yaw_arr[frame, :, 1])
        self.pos_line_yaw.set_3d_properties(yaw_arr[frame, :, 2])

        t = self.time_data[frame]
        alt = self.alt_baro[frame]
        ymin, _ = self.ax_alt.get_ylim()
        
        self.alt_vline.set_data([t, t], [ymin, alt])
        
        # Updating the text in the bottom right corner
        self.time_text.set_text(f"Flight Time: {t:.2f}s | Frame: {frame}/{self.frames_total}")

def run():
    print("[DEBUG] Script started.")
    
    target_file = DATA_FILE
    if not os.path.isfile(target_file):
        print(f"[DEBUG] File NOT found at {DATA_FILE}. Opening file dialog...")
        root = tk.Tk()
        root.withdraw() 
        target_file = filedialog.askopenfilename(
            title="Select the Trajectory File",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not target_file:
            print("[DEBUG] No file selected. Exiting.")
            sys.exit()

    while target_file:
        data_dir = os.path.dirname(target_file)
        icon_path = os.path.join(data_dir, LOGO_FILENAME)

        print(f"[DEBUG] Loading data from {target_file} into Pandas...")
        try:
            if target_file.endswith('.csv'):
                df = pd.read_csv(target_file)
            else:
                df = pd.read_excel(target_file)
            df.columns = df.columns.str.strip()
            print(f"[DEBUG] Successfully loaded {len(df)} rows.")
        except Exception as e:
            print(f"[DEBUG] Error loading file: {e}")
            sys.exit()

        app = RocketVisualizer(df, target_file, icon_path)
        target_file = app.new_file_to_load

if __name__ == "__main__":
    run()