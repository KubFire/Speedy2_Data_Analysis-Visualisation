import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from tkinter import filedialog
import traceback

# ==========================================
# CONFIGURATION
# ==========================================
DATA_FILE = r'C:\Users\kubfi\OneDrive\Documents\GitHub\Speedy2_Data_Analysis&Visualisation\data\23_6_26_launch\processed_trajectory.xlsx'
LOGO_FILENAME = r'C:\Users\kubfi\OneDrive\Documents\GitHub\Speedy2_Data_Analysis&Visualisation\visualisation\SPEEDY_LOGO_V2_red_transparent (1).png'
FRAME_SKIP = 5 

# Column configuration
TIME_COL = 'time_s'
QW_COL = 'q_w'
QX_COL = 'q_x'
QY_COL = 'q_y'
QZ_COL = 'q_z'
POS_X_COL = 'pos_x'
POS_Y_COL = 'pos_y'
POS_Z_COL = 'pos_z'
VEL_X_COL = 'vel_x'
VEL_Y_COL = 'vel_y'
VEL_Z_COL = 'vel_z'

class RocketVisualizer:
    def __init__(self, df, file_path, icon_path):
        print("[DEBUG] Initializing RocketVisualizer object...")
        self.df = df
        self.file_path = file_path
        self.icon_path = icon_path
        self.new_file_to_load = None
        
        required_cols = [TIME_COL, QW_COL, QX_COL, QY_COL, QZ_COL, POS_X_COL, POS_Y_COL, POS_Z_COL, VEL_X_COL, VEL_Y_COL, VEL_Z_COL]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"\n[ERROR] Missing columns in the dataset: {missing_cols}")
            print(f"[DEBUG] Available columns are: {df.columns.tolist()}")
            sys.exit(1)

        self.time_data = df[TIME_COL].values
        self.frames_total = len(self.time_data)
        
        # Scipy Rotation.from_quat expects [x, y, z, w] format
        self.q_data = df[[QX_COL, QY_COL, QZ_COL, QW_COL]].values
        
        self.pos_data = np.column_stack((df[POS_X_COL].values, df[POS_Y_COL].values, df[POS_Z_COL].values))
        self.vel_data = np.column_stack((df[VEL_X_COL].values, df[VEL_Y_COL].values, df[VEL_Z_COL].values))

        # Calculate directional velocity unit vectors for the flight path indicator
        sb = np.linalg.norm(self.vel_data, axis=1, keepdims=True)
        self.vel_dirs = np.divide(self.vel_data, sb, out=np.zeros_like(self.vel_data), where=sb!=0) * 2.0

        self.is_playing = True
        self.speed_multiplier = 1.0
        self.current_frame_float = 0.0

        print("[DEBUG] Building rocket geometry...")
        self.build_rocket_geometry()
        
        self.vec_roll_local = np.array([[0,0,0], [0,0,2.0]])   
        self.vec_pitch_local = np.array([[0,0,0], [0,2.0,0]])  
        self.vec_yaw_local = np.array([[0,0,0], [2.0,0,0]])    

        max_range = max([np.ptp(self.pos_data[:, i]) for i in range(3)]) / 2.0
        if max_range < 1: max_range = 1.0
        pos_scale = max_range * 0.15 
        
        self.pos_vec_roll_local = np.array([[0,0,0], [0,0,pos_scale]])
        self.pos_vec_pitch_local = np.array([[0,0,0], [0,pos_scale,0]])
        self.pos_vec_yaw_local = np.array([[0,0,0], [pos_scale,0,0]])

        print("[DEBUG] Precomputing matrix rotations (this might take a second)...")
        self.precompute_rotations()
        
        print("[DEBUG] Setting up Matplotlib UI...")
        self.setup_plot()
        
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

    def precompute_rotations(self):
        self.rot_matrices = R.from_quat(self.q_data).as_matrix()

        self.pre_rocket = np.einsum('lpm,fcm->flpc', self.rocket_lines_local, self.rot_matrices)
        self.pre_roll = np.einsum('pm,fcm->fpc', self.vec_roll_local, self.rot_matrices)
        self.pre_pitch = np.einsum('pm,fcm->fpc', self.vec_pitch_local, self.rot_matrices)
        self.pre_yaw = np.einsum('pm,fcm->fpc', self.vec_yaw_local, self.rot_matrices)
        
        self.pre_pos_roll = self.pos_data[:, None, :] + np.einsum('pm,fcm->fpc', self.pos_vec_roll_local, self.rot_matrices)
        self.pre_pos_pitch = self.pos_data[:, None, :] + np.einsum('pm,fcm->fpc', self.pos_vec_pitch_local, self.rot_matrices)
        self.pre_pos_yaw = self.pos_data[:, None, :] + np.einsum('pm,fcm->fpc', self.pos_vec_yaw_local, self.rot_matrices)

    def setup_plot(self):
        self.fig = plt.figure(figsize=(18, 9))
        
        try:
            self.fig.canvas.manager.set_window_title("Speedy 2 Flight Visualisation")
            if os.path.exists(self.icon_path):
                icon_img = tk.PhotoImage(file=self.icon_path)
                self.fig.canvas.manager.window.iconphoto(True, icon_img)
        except Exception as e:
            print(f"[DEBUG] Window icon/title set failed (Safe to ignore): {e}")
            pass
            
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
        self.line_path, = self.ax_3d.plot([], [], [], color='magenta', lw=2, linestyle='--', label='Velocity Dir')

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
        
        self.path_line, = self.ax_pos.plot(self.pos_data[:, 0], self.pos_data[:, 1], self.pos_data[:, 2], color='orange', alpha=1.0, lw=1.5, label='Flight Path')
        self.pos_scatter, = self.ax_pos.plot([], [], [], marker='o', color='black', markersize=6)
        self.pos_line_roll, = self.ax_pos.plot([], [], [], color='red', lw=2)
        self.pos_line_pitch, = self.ax_pos.plot([], [], [], color='green', lw=2)
        self.pos_line_yaw, = self.ax_pos.plot([], [], [], color='blue', lw=2)
        
        # Use nanmean, nanmin, and nanmax to ignore empty cells in the Excel sheet
        mids = [np.nanmean([np.nanmin(self.pos_data[:, i]), np.nanmax(self.pos_data[:, i])]) for i in range(3)]
        
        # Calculate ptp (peak-to-peak) manually using nanmax and nanmin
        max_range = max([np.nanmax(self.pos_data[:, i]) - np.nanmin(self.pos_data[:, i]) for i in range(3)]) / 2.0
        if max_range == 0: max_range = 1
        self.ax_pos.set_xlim(mids[0] - max_range, mids[0] + max_range)
        self.ax_pos.set_ylim(mids[1] - max_range, mids[1] + max_range)
        self.ax_pos.set_zlim(mids[2] - max_range, mids[2] + max_range)
        self.ax_pos.legend(loc='upper right', fontsize=8)

        self.ax_alt = self.fig.add_axes([0.67, 0.35, 0.31, 0.55])
        self.ax_alt.plot(self.time_data, self.pos_data[:, 2], color='black', lw=1.5, label='Altitude')
        self.ax_alt.set_title('Altitude vs Time')
        self.ax_alt.set_xlabel('Time (s)')
        self.ax_alt.set_ylabel('Altitude (m)')
        self.ax_alt.grid(True)
        self.alt_vline, = self.ax_alt.plot([], [], color='red', lw=2)

        self.ax_slider = self.fig.add_axes([0.1, 0.22, 0.8, 0.03])
        self.slider = Slider(self.ax_slider, 'Time', 0, self.frames_total - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)

        self.ax_slider.xaxis.set_visible(True)
        max_sec = int(np.ceil(self.time_data[-1]))
        tick_frames = []
        tick_labels = []
        for s in range(0, max_sec + 1, max(1, max_sec // 10)):
            idx = (np.abs(self.time_data - s)).argmin()
            tick_frames.append(idx)
            tick_labels.append(str(s))
        self.ax_slider.set_xticks(tick_frames)
        self.ax_slider.set_xticklabels(tick_labels, color='gray', fontsize=7)
        self.ax_slider.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, colors='gray')
        for spine in self.ax_slider.spines.values():
            spine.set_visible(False)

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
        
        v_dir = self.vel_dirs[frame]
        self.line_path.set_data([0, v_dir[0]], [0, v_dir[1]])
        self.line_path.set_3d_properties([0, v_dir[2]])

        self.pos_scatter.set_data(self.pos_data[frame:frame+1, 0], self.pos_data[frame:frame+1, 1])
        self.pos_scatter.set_3d_properties(self.pos_data[frame:frame+1, 2])

        self.pos_line_roll.set_data(self.pre_pos_roll[frame, :, 0], self.pre_pos_roll[frame, :, 1])
        self.pos_line_roll.set_3d_properties(self.pre_pos_roll[frame, :, 2])

        self.pos_line_pitch.set_data(self.pre_pos_pitch[frame, :, 0], self.pre_pos_pitch[frame, :, 1])
        self.pos_line_pitch.set_3d_properties(self.pre_pos_pitch[frame, :, 2])

        self.pos_line_yaw.set_data(self.pre_pos_yaw[frame, :, 0], self.pre_pos_yaw[frame, :, 1])
        self.pos_line_yaw.set_3d_properties(self.pre_pos_yaw[frame, :, 2])

        t = self.time_data[frame]
        alt = self.pos_data[frame, 2]
        ymin, _ = self.ax_alt.get_ylim()
        
        self.alt_vline.set_data([t, t], [ymin, alt])
        
        self.time_text.set_text(f"Flight Time: {t:.2f}s | Frame: {frame}/{self.frames_total}")

def run():
    print("[DEBUG] Script execution started.")
    target_file = DATA_FILE
    
    if not os.path.isfile(target_file):
        print(f"[DEBUG] Default file not found at: {target_file}")
        print("[DEBUG] Opening file dialog...")
        root = tk.Tk()
        root.withdraw() 
        target_file = filedialog.askopenfilename(
            title="Select the Trajectory File",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not target_file:
            print("[DEBUG] No file selected. Exiting.")
            sys.exit(0)

    while target_file:
        icon_path = LOGO_FILENAME
        print(f"[DEBUG] Attempting to load file: {target_file}")

        try:
            if target_file.endswith('.csv'):
                df = pd.read_csv(target_file)
            else:
                df = pd.read_excel(target_file)
            df.columns = df.columns.str.strip()
            print(f"[DEBUG] Data loaded successfully. Shape: {df.shape}")
            
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Pandas failed to read the file: {e}")
            print("[DEBUG] Ensure you have 'openpyxl' installed if loading an Excel file (pip install openpyxl).")
            traceback.print_exc()
            sys.exit(1)

        try:
            app = RocketVisualizer(df, target_file, icon_path)
            target_file = app.new_file_to_load
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Visualizer crashed during execution: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    run()