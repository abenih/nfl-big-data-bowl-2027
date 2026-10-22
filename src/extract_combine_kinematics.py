"""
Extract Combine Kinematics for Defensive Linemen & Edge Rushers
from RUN_THE_HOOP_DRILL and PASS_RUSH_DRILL.
"""

import os
import numpy as np
import pandas as pd
import polars as pl
from scipy.signal import savgol_filter

DATA_DIR = '/home/benji/Downloads/y/nfl-big-data-bowl-2027'
OUTPUT_DIR = '/home/benji/Downloads/nfl_bdb_2027_bend/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_hoop_drill():
    print("Loading Combine tracking for RUN_THE_HOOP_DRILL...")
    cb_trk = pl.read_csv(f'{DATA_DIR}/combine_tracking.csv', null_values=['NA', '']).filter(
        (pl.col('drill_name') == 'RUN_THE_HOOP_DRILL') & (pl.col('entity_type') == 'PLAYER')
    ).to_pandas()
    
    # Nominal hoop center fitted across all cohorts: (48.557, 23.378), radius 5.624 yds
    HOOP_X0 = 48.557
    HOOP_Y0 = 23.378
    HOOP_R0 = 5.624

    records = []
    
    for (pid, draft_year, att), grp in cb_trk.groupby(['nfl_id', 'draft_year', 'attempt']):
        grp = grp.sort_values('time').reset_index(drop=True)
        if len(grp) < 25:
            continue
            
        x_raw = grp['x'].values
        y_raw = grp['y'].values
        time_vals = grp['time'].values
        
        # Apply Savitzky-Golay filter to smooth 10 Hz optical/sensor coordinates
        window = min(11, len(grp) if len(grp) % 2 != 0 else len(grp) - 1)
        if window < 5:
            continue
            
        x_smooth = savgol_filter(x_raw, window_length=window, polyorder=2)
        y_smooth = savgol_filter(y_raw, window_length=window, polyorder=2)
        
        # 1st derivatives (velocities) at dt = 0.1s
        dt = 0.1
        vx = np.gradient(x_smooth, dt)
        vy = np.gradient(y_smooth, dt)
        speed = np.sqrt(vx**2 + vy**2)
        
        # 2nd derivatives (accelerations)
        ax = np.gradient(vx, dt)
        ay = np.gradient(vy, dt)
        accel_mag = np.sqrt(ax**2 + ay**2)
        
        # Differential geometry: Curvature kappa(t) = |vx*ay - vy*ax| / (vx^2 + vy^2)^(3/2)
        denom = (vx**2 + vy**2)**(1.5)
        valid_denom = denom > 1e-2
        kappa = np.zeros_like(speed)
        kappa[valid_denom] = np.abs(vx[valid_denom] * ay[valid_denom] - vy[valid_denom] * ax[valid_denom]) / denom[valid_denom]
        
        # Centripetal acceleration a_c = speed^2 * kappa
        a_c = (speed**2) * kappa
        
        # Radial distance from the physical hoop center
        r_dist = np.sqrt((x_smooth - HOOP_X0)**2 + (y_smooth - HOOP_Y0)**2)
        # Radial drift: integrated excess distance beyond hoop boundary
        radial_drift = np.mean(np.maximum(0, r_dist - HOOP_R0))
        
        # Identify the apex of the turn (maximum curvature point)
        # Focus on the middle 60% of the trajectory to avoid start/stop artifacts
        mid_start = int(0.15 * len(grp))
        mid_end = int(0.85 * len(grp))
        
        if mid_end <= mid_start:
            continue
            
        mid_kappa = kappa[mid_start:mid_end]
        apex_sub_idx = np.argmax(mid_kappa)
        apex_idx = mid_start + apex_sub_idx
        
        apex_curvature = kappa[apex_idx]
        apex_radius = 1.0 / max(1e-3, apex_curvature)
        v_at_apex = speed[apex_idx]
        v_entry = np.max(speed[:apex_idx]) if apex_idx > 0 else speed[apex_idx]
        v_peak = np.max(speed)
        
        # Speed Retention Ratio through the bend
        speed_retention = v_at_apex / max(1.0, v_entry)
        
        # Peak centripetal acceleration in the turn
        peak_centripetal = np.max(a_c[mid_start:mid_end]) if len(a_c[mid_start:mid_end]) > 0 else 0.0
        
        # Drill duration (seconds)
        total_time = len(grp) * dt
        
        # Tangential acceleration along the curve
        # a_t = (vx*ax + vy*ay) / speed
        a_t = np.zeros_like(speed)
        valid_s = speed > 1e-2
        a_t[valid_s] = (vx[valid_s] * ax[valid_s] + vy[valid_s] * ay[valid_s]) / speed[valid_s]
        exit_accel = np.mean(a_t[apex_idx:]) if apex_idx < len(a_t) else 0.0
        
        # Edge Bend Index (composite metric):
        # Rewards high centripetal acceleration, high speed retention, low drill time, and minimal radial drift
        ebi = (peak_centripetal * speed_retention) / (total_time * (1.0 + 0.5 * radial_drift))
        
        records.append({
            'nfl_id': pid,
            'draft_year': draft_year,
            'attempt': att,
            'total_time': total_time,
            'v_peak': v_peak,
            'v_at_apex': v_at_apex,
            'v_entry': v_entry,
            'speed_retention': speed_retention,
            'apex_curvature': apex_curvature,
            'apex_radius': apex_radius,
            'peak_centripetal': peak_centripetal,
            'radial_drift': radial_drift,
            'exit_accel': exit_accel,
            'edge_bend_index': ebi
        })
        
    df_hoop = pd.DataFrame(records)
    print(f"Extracted hoop drill features for {len(df_hoop)} attempts across {df_hoop['nfl_id'].nunique()} players.")
    
    # Aggregate per player (best or mean across attempts)
    hoop_player_summary = df_hoop.groupby('nfl_id').agg({
        'draft_year': 'first',
        'total_time': 'min', # best time
        'v_peak': 'max',
        'v_at_apex': 'max',
        'speed_retention': 'max',
        'apex_curvature': 'max',
        'apex_radius': 'min',
        'peak_centripetal': 'max',
        'radial_drift': 'mean',
        'exit_accel': 'max',
        'edge_bend_index': 'max'
    }).reset_index()
    
    return hoop_player_summary

def process_pass_rush_drill():
    print("Loading Combine tracking for PASS_RUSH_DRILL...")
    cb_trk = pl.read_csv(f'{DATA_DIR}/combine_tracking.csv', null_values=['NA', '']).filter(
        (pl.col('drill_name') == 'PASS_RUSH_DRILL') & (pl.col('entity_type') == 'PLAYER')
    ).to_pandas()
    
    records = []
    for (pid, att), grp in cb_trk.groupby(['nfl_id', 'attempt']):
        grp = grp.sort_values('time').reset_index(drop=True)
        if len(grp) < 15:
            continue
        
        s = grp['s'].values
        a = grp['a'].values
        
        # Initial 10 frames (1 second) acceleration burst
        first_step_accel = np.max(a[:min(10, len(a))])
        first_step_speed = s[min(10, len(s)-1)]
        max_rush_accel = np.max(a)
        max_rush_speed = np.max(s)
        
        records.append({
            'nfl_id': pid,
            'attempt': att,
            'pr_first_step_accel': first_step_accel,
            'pr_first_step_speed': first_step_speed,
            'pr_max_accel': max_rush_accel,
            'pr_max_speed': max_rush_speed
        })
        
    df_pr = pd.DataFrame(records).groupby('nfl_id').mean().reset_index()
    print(f"Extracted pass rush drill features for {len(df_pr)} players.")
    return df_pr

def main():
    df_hoop = process_hoop_drill()
    df_pr = process_pass_rush_drill()
    
    # Merge combine tracking features
    combine_kinematics = pd.merge(df_hoop, df_pr, on='nfl_id', how='outer')
    
    # Normalize Edge Bend Index to a 0-100 scale
    ebi_vals = combine_kinematics['edge_bend_index'].dropna()
    min_ebi, max_ebi = ebi_vals.quantile(0.02), ebi_vals.quantile(0.98)
    combine_kinematics['ebi_score'] = 100.0 * (combine_kinematics['edge_bend_index'] - min_ebi) / (max_ebi - min_ebi)
    combine_kinematics['ebi_score'] = combine_kinematics['ebi_score'].clip(0, 100)
    
    out_path = f'{OUTPUT_DIR}/combine_edge_kinematics.csv'
    combine_kinematics.to_csv(out_path, index=False)
    print(f"Saved combine kinematics to {out_path} ({len(combine_kinematics)} rows)")
    print(combine_kinematics.head())

if __name__ == '__main__':
    main()

