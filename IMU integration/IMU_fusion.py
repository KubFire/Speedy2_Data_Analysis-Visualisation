import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.integrate import cumulative_trapezoid

def compute_adaptive_orientation(acc_body, gyro_body, dt_array, base_beta=0.15):
    num_samples = len(dt_array)
    q = np.zeros((num_samples, 4))
    
    q[0] = [1.0, 0.0, 0.0, 0.0] 
    
    for i in range(1, num_samples):
        q_prev = q[i-1]
        a_vec = acc_body[i]
        g_vec = gyro_body[i]
        dt = dt_array[i]
        
        qw, qx, qy, qz = q_prev
        gx, gy, gz = g_vec
        
        dot_q = 0.5 * np.array([
            -qx*gx - qy*gy - qz*gz,
             qw*gx + qy*gz - qz*gy,
             qw*gy - qx*gz + qz*gx,
             qw*gz + qx*gy - qy*gx
        ])
        
        a_norm = np.linalg.norm(a_vec)
        g_diff = abs(a_norm - 9.80665)
        
        if a_norm == 0 or g_diff > 0.98:
            beta = 0.0  
        else:
            beta = base_beta  
            
        if beta > 0:
            a_vec = a_vec / a_norm
            ax, ay, az = a_vec
            
            f1 = 2.0 * (qx*qz - qw*qy) - ax
            f2 = 2.0 * (qw*qx + qy*qz) - ay
            f3 = 2.0 * (0.5 - qx**2 - qy**2) - az
            
            j11, j12, j13, j14 = -2.0*qy,  2.0*qz, -2.0*qw,  2.0*qx
            j21, j22, j23, j24 =  2.0*qx,  2.0*qw,  2.0*qz,  2.0*qy
            j31, j32, j33, j34 =  0.0,    -4.0*qx, -4.0*qy,  0.0
            
            step = np.array([
                j11*f1 + j21*f2 + j31*f3,
                j12*f1 + j22*f2 + j32*f3,
                j13*f1 + j23*f2 + j33*f3,
                j14*f1 + j24*f2 + j34*f3
            ])
            
            step_norm = np.linalg.norm(step)
            if step_norm > 0:
                step = step / step_norm
            
            dot_q -= beta * step
            
        q_new = q_prev + dot_q * dt
        
        q_norm = np.linalg.norm(q_new)
        if q_norm == 0 or np.isnan(q_norm):
            q[i] = q_prev  
        else:
            q[i] = q_new / q_norm
        
    return q


def process_flight_data(df, time_col='time', acc_cols=('ax', 'ay', 'az'), gyro_cols=('gx', 'gy', 'gz'), alt_col='True Alt', state_col='flight_state'):
    # Drop only trailing empty Excel rows by checking the time column specifically. 
    # This preserves the structural integrity of the internal time steps.
    df = df.dropna(subset=[time_col]).copy()
    
    time_s_raw = df[time_col].values / 1000.0
    
    dt_raw = np.diff(time_s_raw, prepend=time_s_raw[0]) 
    
    dt_array = np.clip(dt_raw, a_min=0.001, a_max=None)
    time_s = np.cumsum(dt_array)
    
    acc_raw = df[list(acc_cols)].values
    
    resting_mag = np.mean(np.linalg.norm(acc_raw[:50], axis=1))
    if resting_mag == 0 or np.isnan(resting_mag):
        resting_mag = 1.0  
    acc_body = (acc_raw / resting_mag) * 9.80665
    
    gyro_scale = (np.pi / 180.0) / 32.8 
    
    gyro_body = df[list(gyro_cols)].values * gyro_scale
    
    Q = compute_adaptive_orientation(acc_body, gyro_body, dt_array)
    
    rotations = R.from_quat(np.roll(Q, -1, axis=1)) 
    euler_deg = rotations.as_euler('xyz', degrees=True)
    
    acc_earth = rotations.apply(acc_body)
    
    bias_window = slice(0, min(50, len(time_s)))
    bias_vector = np.mean(acc_earth[bias_window], axis=0)
    
    acc_earth_linear = acc_earth - bias_vector
    
    # Restored exact pure integration from the original version
    vel_earth = cumulative_trapezoid(acc_earth_linear, x=time_s, axis=0, initial=0)
    pos_earth = cumulative_trapezoid(vel_earth, x=time_s, axis=0, initial=0)
    
    baro_alt = df[alt_col].values if alt_col in df.columns else np.zeros(len(time_s))
    state_data = df[state_col].values if state_col in df.columns else np.zeros(len(time_s))
    
    return time_s, Q, euler_deg, pos_earth, vel_earth, baro_alt, state_data


if __name__ == "__main__":
    data_dir = r"C:\Users\kubfi\OneDrive\Documents\GitHub\Speedy2_Data_Analysis&Visualisation\data\23_6_26_launch"
    
    input_file = os.path.join(data_dir, "launch IMU data.xlsx")
    state_file = os.path.join(data_dir, "processed_trajectory_final_brokenVersion.xlsx - Sheet1.csv")
    output_file = os.path.join(data_dir, "processed_trajectory.xlsx")
    
    print(f"Loading data from: {input_file}")
    
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
        print(f"Successfully loaded {len(df)} rows. Processing telemetry...")
        
        # Merge State from the attached CSV
        try:
            df_state = pd.read_csv(state_file)
            csv_state_col = next((col for col in df_state.columns if 'state' in col.lower()), None)
            
            if csv_state_col:
                target_length = len(df)
                state_values = df_state[csv_state_col].values
                
                if len(state_values) >= target_length:
                    df['flight_state'] = state_values[:target_length]
                else:
                    padded_states = np.pad(state_values, (0, target_length - len(state_values)), mode='edge')
                    df['flight_state'] = padded_states
                    
                print(f"Successfully merged '{csv_state_col}' from attached CSV.")
            else:
                print("Warning: Could not find a column named 'state' in the attached CSV.")
                df['flight_state'] = 0
        except Exception as e:
            print(f"Warning: Could not load or merge state file: {e}")
            df['flight_state'] = 0
            
    except Exception as e:
        print(f"CRITICAL ERROR loading Excel file: {e}")
        exit(1)
    
    try:
        time_s, Q, euler_deg, pos_earth, vel_earth, baro_alt, state_data = process_flight_data(
            df, 
            time_col='time',          
            acc_cols=('ax', 'ay', 'az'), 
            gyro_cols=('gx', 'gy', 'gz'),
            alt_col='True Alt',  
            state_col='flight_state'
        )
    except Exception as e:
         print(f"CRITICAL ERROR during math integration: {e}")
         exit(1)
    
    output_df = pd.DataFrame({
        'time_s': time_s,
        'state': state_data,
        'baro_alt': baro_alt,
        'q_w': Q[:, 0],
        'q_x': Q[:, 1],
        'q_y': Q[:, 2],
        'q_z': Q[:, 3],
        'euler_x_deg': euler_deg[:, 0],
        'euler_y_deg': euler_deg[:, 1],
        'euler_z_deg': euler_deg[:, 2],
        'pos_x': pos_earth[:, 0],
        'pos_y': pos_earth[:, 1],
        'pos_z': pos_earth[:, 2],
        'vel_x': vel_earth[:, 0],
        'vel_y': vel_earth[:, 1],
        'vel_z': vel_earth[:, 2]
    })
    
    print(f"Exporting {len(output_df)} processed rows to: {output_file}")
    
    output_df.to_excel(output_file, index=False, engine='openpyxl')
    
    print("Integration complete.")