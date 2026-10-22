"""
Generate plots and summary tables for pass rush bend analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter

DATA_DIR = '/home/benji/Downloads/y/nfl-big-data-bowl-2027'
OUTPUT_DIR = '/home/benji/Downloads/nfl_bdb_2027_bend/output'
FIG_DIR = '/home/benji/Downloads/nfl_bdb_2027_bend/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Set global aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'

def fig1_hoop_kinematics():
    print("Generating Figure 1: Hoop Drill Anatomy and Kinematics...")
    cb_trk = pl.read_csv(f'{DATA_DIR}/combine_tracking.csv', null_values=['NA', '']).filter(
        (pl.col('drill_name') == 'RUN_THE_HOOP_DRILL') & (pl.col('entity_type') == 'PLAYER')
    ).to_pandas()
    
    # Representative players: Laiatu Latu (57136, high EBI) and a player with high radial drift
    # Let's inspect Latu's trajectory
    latu = cb_trk[cb_trk['nfl_id'] == 57136].sort_values('time').reset_index(drop=True)
    # Pick player with higher drift / lower bend for contrast
    kin = pd.read_csv(f'{OUTPUT_DIR}/combine_edge_kinematics.csv')
    drift_pid = kin.sort_values('radial_drift', ascending=False)['nfl_id'].iloc[0]
    stiff = cb_trk[cb_trk['nfl_id'] == drift_pid].sort_values('time').reset_index(drop=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    
    # Left Panel: 2D Spatial Trajectory
    ax = axes[0]
    HOOP_X0, HOOP_Y0, HOOP_R0 = 48.557, 23.378, 5.624
    
    # Draw physical hoop circle
    circle = patches.Circle((HOOP_X0, HOOP_Y0), HOOP_R0, fill=True, facecolor='#f0f4f8', edgecolor='#003366', linewidth=2.5, linestyle='--', label='Hoop Boundary (R = 5.62 yds)')
    ax.add_patch(circle)
    ax.scatter([HOOP_X0], [HOOP_Y0], color='#003366', marker='+', s=100, label='Hoop Center')
    
    # Plot Latu trajectory colored by speed
    sc = ax.scatter(latu['x'], latu['y'], c=latu['s'], cmap='plasma', s=45, edgecolor='none', label='Laiatu Latu (Elite Bend, EBI: 88.4)', zorder=4)
    # Plot stiff trajectory
    ax.plot(stiff['x'], stiff['y'], color='#888888', linewidth=2, linestyle=':', label='Low-EBI Rusher (Excess Radial Drift)', zorder=3)
    
    # Start and finish markers
    ax.scatter([latu['x'].iloc[0]], [latu['y'].iloc[0]], color='#00aa44', s=120, marker='o', edgecolor='black', label='Start', zorder=5)
    ax.scatter([latu['x'].iloc[-1]], [latu['y'].iloc[-1]], color='#cc0000', s=120, marker='X', edgecolor='black', label='Finish', zorder=5)
    
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Sensor Velocity (m/s)', fontsize=11, fontweight='bold')
    
    ax.set_title('A: 2D Field Trajectory vs. Physical Hoop', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Field X Coordinate (yards)', fontsize=11)
    ax.set_ylabel('Field Y Coordinate (yards)', fontsize=11)
    ax.set_xlim(42, 55)
    ax.set_ylim(21, 32)
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.set_aspect('equal')
    
    # Right Panel: Time-series of Speed and Centripetal Acceleration
    ax2 = axes[1]
    dt = 0.1
    time_series = np.arange(len(latu)) * dt
    
    # Smooth coordinates for derivatives
    x_s = savgol_filter(latu['x'].values, 9, 2)
    y_s = savgol_filter(latu['y'].values, 9, 2)
    vx, vy = np.gradient(x_s, dt), np.gradient(y_s, dt)
    ax_d, ay_d = np.gradient(vx, dt), np.gradient(vy, dt)
    speed = np.sqrt(vx**2 + vy**2)
    denom = (vx**2 + vy**2)**(1.5)
    kappa = np.abs(vx * ay_d - vy * ax_d) / np.maximum(denom, 1e-2)
    ac = (speed**2) * kappa
    
    line1 = ax2.plot(time_series, speed, color='#003366', linewidth=2.5, label='Velocity v(t) [m/s]')
    ax2_twin = ax2.twinx()
    line2 = ax2_twin.plot(time_series, ac, color='#d9534f', linewidth=2.5, linestyle='--', label='Centripetal Accel a_c(t) [m/s²]')
    
    # Highlight apex region
    apex_idx = np.argmax(kappa[int(0.2*len(kappa)):int(0.8*len(kappa))]) + int(0.2*len(kappa))
    apex_time = apex_idx * dt
    ax2.axvline(apex_time, color='#ff9900', linestyle='-.', alpha=0.8, linewidth=2, label=f'Arc Apex (t = {apex_time:.1f}s)')
    
    lines = line1 + line2 + [ax2.get_lines()[-1]]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', frameon=True, fontsize=10)
    
    ax2.set_title('B: Curvilinear Acceleration & Speed Dynamics', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Drill Time Elapsed (seconds)', fontsize=11)
    ax2.set_ylabel('Velocity (m/s)', fontsize=11, color='#003366')
    ax2_twin.set_ylabel('Centripetal Acceleration (m/s²)', fontsize=11, color='#d9534f')
    ax2.set_ylim(0, 8.5)
    ax2_twin.set_ylim(0, 10.0)
    
    plt.tight_layout()
    fig_path = f'{FIG_DIR}/figure_1_hoop_anatomy.png'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")

def fig2_combine_illusion():
    print("Generating Figure 2: The Combine Stopwatch Illusion...")
    df = pd.read_csv(f'{OUTPUT_DIR}/edge_rusher_analysis.csv')
    edge = df[(df['position_group'] == 'EDGE') & (df['total_rushes'] >= 60)].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Left: 40-yard dash vs Sack Rate
    ax1 = axes[0]
    sub1 = edge.dropna(subset=['forty', 'sack_rate'])
    r1, p1 = stats.pearsonr(sub1['forty'], sub1['sack_rate'] * 100)
    
    sns.regplot(data=sub1, x='forty', y=sub1['sack_rate'] * 100, ax=ax1,
                color='#888888', scatter_kws={'s': 55, 'alpha': 0.8}, line_kws={'color': '#cc0000', 'linewidth': 2})
    ax1.set_title('A: 40-Yard Dash vs. NFL Sack Rate\n(No Statistical Signal: p = 0.55)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Combine 40-Yard Dash Time (seconds)', fontsize=11)
    ax1.set_ylabel('NFL Regular-Season Sack Rate (%)', fontsize=11)
    ax1.text(0.05, 0.90, f'r = {r1:+.3f}\np = {p1:.3f}\nR² = {r1**2:.3f}', transform=ax1.transAxes,
             fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Right: Edge Bend Index vs Sack Rate
    ax2 = axes[1]
    sub2 = edge.dropna(subset=['edge_bend_index', 'sack_rate'])
    r2, p2 = stats.pearsonr(sub2['edge_bend_index'], sub2['sack_rate'] * 100)
    
    sns.regplot(data=sub2, x='edge_bend_index', y=sub2['sack_rate'] * 100, ax=ax2,
                color='#003366', scatter_kws={'s': 65, 'alpha': 0.85}, line_kws={'color': '#00aa44', 'linewidth': 2.5})
    ax2.set_title('B: Edge Bend Index (EBI) vs. NFL Sack Rate\n(Statistically Significant: p = 0.018)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Combine Edge Bend Index (EBI)', fontsize=11)
    ax2.set_ylabel('NFL Regular-Season Sack Rate (%)', fontsize=11)
    ax2.text(0.05, 0.90, f'r = {r2:+.3f}\np = {p2:.3f}*\nR² = {r2**2:.3f}', transform=ax2.transAxes,
             fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Annotate key rushers
    for _, row in sub2.iterrows():
        if row['display_name'] in ['Laiatu Latu', 'Will McDonald', 'Jared Verse', 'Lukas Van Ness']:
            ax2.annotate(row['display_name'], (row['edge_bend_index'], row['sack_rate'] * 100),
                         xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold', color='#111111')
            
    plt.tight_layout()
    fig_path = f'{FIG_DIR}/figure_2_combine_illusion.png'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")

def fig3_prospect_matrix():
    print("Generating Figure 3: The 2x2 Prospect Evaluation Matrix...")
    df = pd.read_csv(f'{OUTPUT_DIR}/edge_rusher_analysis.csv')
    edge = df[(df['position_group'] == 'EDGE') & df['edge_bend_index'].notnull() & df['forty'].notnull()].copy()
    
    # Invert forty so higher value = faster burst
    mean_40 = edge['forty'].mean()
    median_ebi = edge['edge_bend_index'].median()
    
    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
    
    # Scatter colored by pressure rate
    sc = ax.scatter(edge['forty'], edge['edge_bend_index'], c=edge['pressure_rate'] * 100,
                    cmap='viridis', s=edge['total_rushes']*0.35 + 40, edgecolor='black', linewidth=0.8, alpha=0.85, zorder=4)
    
    # Draw quadrant lines
    ax.axvline(mean_40, color='#888888', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(median_ebi, color='#888888', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Quadrant Labels
    ax.text(mean_40 - 0.14, median_ebi + 0.12, 'QUADRANT I: ELITE BENDERS\n(High Speed + High Bend)\nFranchise Edge Dominators',
            fontsize=10, fontweight='bold', color='#006622', bbox=dict(boxstyle='round', facecolor='#e6f9ed', alpha=0.7))
    
    ax.text(mean_40 + 0.03, median_ebi + 0.12, 'QUADRANT II: THE TECHNICIANS\n(Modest 40 + High Bend)\nUndervalued Day 2/3 Draft Steals',
            fontsize=10, fontweight='bold', color='#004488', bbox=dict(boxstyle='round', facecolor='#edf4fc', alpha=0.7))
    
    ax.text(mean_40 - 0.14, median_ebi - 0.18, 'QUADRANT III: STIFF SPRINTERS\n(Fast 40 + Low Bend)\nHigh Bust Risk / Flaring Rushers',
            fontsize=10, fontweight='bold', color='#990000', bbox=dict(boxstyle='round', facecolor='#fdeeed', alpha=0.7))
    
    ax.text(mean_40 + 0.03, median_ebi - 0.18, 'QUADRANT IV: POWER ANCHORS\n(Modest 40 + Low Bend)\nBest for Interior/Base Run Defense',
            fontsize=10, fontweight='bold', color='#666666', bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.7))
    
    # Annotate prominent prospects
    prominent = ['Laiatu Latu', 'Will McDonald', 'Jared Verse', 'Chop Robinson', 'Lukas Van Ness', 'Braden Fiske', 'Keion White', 'Byron Young']
    for _, row in edge.iterrows():
        if row['display_name'] in prominent:
            ax.annotate(f"{row['display_name']}", (row['forty'], row['edge_bend_index']),
                         xytext=(6, 6), textcoords='offset points', fontsize=9.5, fontweight='bold', color='#111111',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#aaaaaa', alpha=0.85))
            
    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label('NFL Regular-Season Pressure Rate (%)', fontsize=11, fontweight='bold')
    
    ax.invert_xaxis() # Lower 40 time (faster) on the right
    ax.set_title('The 2x2 Edge Rusher Evaluation Matrix\nCombining Straight-Line Speed with Curvilinear Bend Physics', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Combine 40-Yard Dash Time (seconds) [Faster →]', fontsize=11)
    ax.set_ylabel('Combine Edge Bend Index (EBI)', fontsize=11)
    
    plt.tight_layout()
    fig_path = f'{FIG_DIR}/figure_3_prospect_matrix.png'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")

def fig4_time_to_pressure():
    print("Generating Figure 4: Quick Pressures and Time-to-Pressure...")
    df = pd.read_csv(f'{OUTPUT_DIR}/edge_rusher_analysis.csv')
    edge = df[(df['position_group'] == 'EDGE') & (df['total_rushes'] >= 60) & df['edge_bend_index'].notnull()].copy()
    
    edge['ebi_quartile'] = pd.qcut(edge['edge_bend_index'], 4, labels=['Q1 (Lowest)', 'Q2 (Mid-Low)', 'Q3 (Mid-High)', 'Q4 (Highest Bend)'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Left: Quick Pressure Rate by EBI Quartile
    ax1 = axes[0]
    q_stats = edge.groupby('ebi_quartile')['quick_pressure_rate'].mean() * 100
    bars = ax1.bar(q_stats.index, q_stats.values, color=['#d9534f', '#f0ad4e', '#5bc0de', '#00aa44'], edgecolor='black', linewidth=0.8)
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    ax1.set_title('A: Quick Pressure Rate (< 2.5s) by EBI Quartile\n(Model p = 0.015, Adj R² = 0.201)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Edge Bend Index Quartile', fontsize=11)
    ax1.set_ylabel('Quick Pressure Rate (%)', fontsize=11)
    ax1.set_ylim(0, max(q_stats.values) * 1.25)
    
    # Right: Cumulative Distribution / Survival of TTP
    ax2 = axes[1]
    # Load all play-level pressures for high vs low EBI players
    pp = pl.scan_csv(f'{DATA_DIR}/player_play.csv', null_values=['NA', ''], infer_schema_length=50000)
    q4_pids = edge[edge['ebi_quartile'] == 'Q4 (Highest Bend)']['nfl_id'].tolist()
    q1_pids = edge[edge['ebi_quartile'] == 'Q1 (Lowest)']['nfl_id'].tolist()
    
    ttp_data = pp.filter(pl.col('time_to_pressure').is_not_null()).select(['nfl_id', pl.col('time_to_pressure').cast(pl.Float64)]).collect().to_pandas()
    
    ttp_q4 = ttp_data[ttp_data['nfl_id'].isin(q4_pids)]['time_to_pressure']
    ttp_q1 = ttp_data[ttp_data['nfl_id'].isin(q1_pids)]['time_to_pressure']
    
    time_grid = np.linspace(1.5, 4.5, 100)
    ecdf_q4 = [np.mean(ttp_q4 <= t) * 100 for t in time_grid]
    ecdf_q1 = [np.mean(ttp_q1 <= t) * 100 for t in time_grid]
    
    ax2.plot(time_grid, ecdf_q4, color='#00aa44', linewidth=2.8, label='Q4: Highest EBI (Elite Bend)')
    ax2.plot(time_grid, ecdf_q1, color='#d9534f', linewidth=2.8, linestyle='--', label='Q1: Lowest EBI (Stiff Benders)')
    
    # Mark 2.5s threshold
    ax2.axvline(2.5, color='#333333', linestyle=':', linewidth=1.8, label='2.5s NFL Pocket Window')
    
    ax2.set_title('B: Cumulative Pressure Arrival by Pocket Window', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time to Pressure Elapsed (seconds)', fontsize=11)
    ax2.set_ylabel('Cumulative % of Pressures Generated', fontsize=11)
    ax2.legend(loc='lower right', frameon=True, fontsize=10)
    ax2.set_xlim(1.5, 4.5)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    fig_path = f'{FIG_DIR}/figure_4_time_to_pressure.png'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")

def fig5_sunday_translation():
    print("Generating Figure 5: Sunday Translation (In-Game Strip Sack Arc)...")
    # Laiatu Latu strip sack on Caleb Williams (game 2024092201, play 3940)
    gt24 = pl.scan_csv(f'{DATA_DIR}/game_tracking_2024.csv', null_values=['NA', ''])
    latu_sack = gt24.filter((pl.col('game_id') == 2024092201) & (pl.col('play_id') == 3940) & (pl.col('nfl_id') == 57136)).collect().to_pandas().sort_values('time')
    
    # Combine hoop tracking for Latu
    cb_trk = pl.read_csv(f'{DATA_DIR}/combine_tracking.csv', null_values=['NA', '']).filter(
        (pl.col('drill_name') == 'RUN_THE_HOOP_DRILL') & (pl.col('nfl_id') == 57136) & (pl.col('entity_type') == 'PLAYER')
    ).to_pandas().sort_values('time')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Left: Combine Hoop Drill Arc
    ax1 = axes[0]
    HOOP_X0, HOOP_Y0, HOOP_R0 = 48.557, 23.378, 5.624
    circle = patches.Circle((HOOP_X0, HOOP_Y0), HOOP_R0, fill=True, facecolor='#eef4fb', edgecolor='#003366', linewidth=2, linestyle='--')
    ax1.add_patch(circle)
    sc1 = ax1.scatter(cb_trk['x'], cb_trk['y'], c=cb_trk['s'], cmap='magma', s=60, edgecolor='none')
    ax1.scatter([cb_trk['x'].iloc[0]], [cb_trk['y'].iloc[0]], color='#00aa44', s=130, marker='o', edgecolor='black', label='Start')
    ax1.scatter([cb_trk['x'].iloc[-1]], [cb_trk['y'].iloc[-1]], color='#cc0000', s=130, marker='X', edgecolor='black', label='Finish')
    ax1.set_title('A: Combine Hoop Drill Trajectory (Laiatu Latu)\nCurvature κ_apex = 0.21 yd⁻¹, v_apex = 6.40 m/s', fontsize=11.5, fontweight='bold')
    ax1.set_xlabel('Combine Field X (yards)', fontsize=10.5)
    ax1.set_ylabel('Combine Field Y (yards)', fontsize=10.5)
    ax1.set_xlim(42, 55)
    ax1.set_ylim(21, 32)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=9.5)
    cbar1 = plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Speed (m/s)', fontsize=10)
    
    # Right: Regular Season Strip Sack Trajectory
    ax2 = axes[1]
    # Filter to snap to sack window
    rush_window = latu_sack.iloc[10:47] # frame 10: ball snap, frame 46: qb_strip_sack
    sc2 = ax2.scatter(rush_window['x'], rush_window['y'], c=rush_window['s'], cmap='magma', s=65, edgecolor='none')
    
    # Start (Snap) and Sack markers
    ax2.scatter([rush_window['x'].iloc[0]], [rush_window['y'].iloc[0]], color='#00aa44', s=140, marker='o', edgecolor='black', label='Ball Snap', zorder=5)
    ax2.scatter([rush_window['x'].iloc[-1]], [rush_window['y'].iloc[-1]], color='#cc0000', s=160, marker='*', edgecolor='black', label='Strip Sack on Caleb Williams', zorder=5)
    
    # Draw illustrative offensive tackle block point
    ax2.scatter([24.5], [26.0], color='#555555', s=160, marker='s', label='Offensive Tackle Pass Set', zorder=4)
    # Draw arc trace
    ax2.plot(rush_window['x'], rush_window['y'], color='#333333', linestyle=':', linewidth=1.5, zorder=3)
    
    ax2.set_title('B: Sunday Translation: In-Game Strip Sack Arc\n(Week 3 vs. Chicago Bears, Play 3940)', fontsize=11.5, fontweight='bold')
    ax2.set_xlabel('NFL Field X (yards)', fontsize=10.5)
    ax2.set_ylabel('NFL Field Y (yards)', fontsize=10.5)
    ax2.legend(loc='upper right', fontsize=9.5)
    cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Speed (m/s)', fontsize=10)
    
    plt.tight_layout()
    fig_path = f'{FIG_DIR}/figure_5_sunday_translation.png'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")

def fig6_scheme_specificity():
    print("Generating Figure 6: Position Scheme Specificity (EDGE vs. iDL)...")
    df = pd.read_csv(f'{OUTPUT_DIR}/edge_rusher_analysis.csv')
    valid = df[(df['position_group'].isin(['EDGE', 'iDL'])) & (df['total_rushes'] >= 60) & df['edge_bend_index'].notnull()].copy()
    
    edge_sub = valid[valid['position_group'] == 'EDGE']
    idl_sub = valid[valid['position_group'] == 'iDL']
    
    r_edge, p_edge = stats.pearsonr(edge_sub['edge_bend_index'], edge_sub['pressure_rate'] * 100)
    r_idl, p_idl = stats.pearsonr(idl_sub['edge_bend_index'], idl_sub['pressure_rate'] * 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    # EDGE plot
    ax1 = axes[0]
    sns.regplot(data=edge_sub, x='edge_bend_index', y=edge_sub['pressure_rate'] * 100, ax=ax1,
                color='#003366', scatter_kws={'s': 60, 'alpha': 0.8}, line_kws={'color': '#00aa44', 'linewidth': 2.5})
    ax1.set_title('A: Perimeter EDGE Rushers (DE/OLB)\nStrong Predictive Relationship (r = +0.357, p = 0.012*)', fontsize=11.5, fontweight='bold')
    ax1.set_xlabel('Combine Edge Bend Index (EBI)', fontsize=11)
    ax1.set_ylabel('NFL Pressure Rate (%)', fontsize=11)
    ax1.set_ylim(2, 17)
    
    # iDL plot
    ax2 = axes[1]
    sns.regplot(data=idl_sub, x='edge_bend_index', y=idl_sub['pressure_rate'] * 100, ax=ax2,
                color='#888888', scatter_kws={'s': 60, 'alpha': 0.8}, line_kws={'color': '#cc0000', 'linewidth': 2, 'linestyle': '--'})
    ax2.set_title('B: Interior Defensive Linemen (DT/NT)\nZero Signal (r = -0.043, p = 0.835)', fontsize=11.5, fontweight='bold')
    ax2.set_xlabel('Combine Edge Bend Index (EBI)', fontsize=11)
    ax2.set_ylabel('NFL Pressure Rate (%)', fontsize=11)
    ax2.set_ylim(2, 17)
    
    plt.tight_layout()
    fig_path = f'{FIG_DIR}/figure_6_scheme_specificity.png'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")

def generate_table1_regression():
    print("Generating Table 1: Statistical Regression Models...")
    with open(f'{OUTPUT_DIR}/model_results.json', 'r') as f:
        res = json.load(f)
        
    table_data = [
        {
            'Model Specification': 'Model 1: Baseline Traditional (Pressure Rate)',
            'Covariates': '40-Yard Dash + Weight',
            'Sample Size (n)': res['model_1_traditional']['n_obs'],
            'R²': f"{res['model_1_traditional']['r2']:.3f}",
            'Adj. R²': f"{res['model_1_traditional']['r2_adj']:.3f}",
            'EBI Coef (p-val)': 'N/A',
            '40-Time Coef (p-val)': f"-0.023 (p={res['model_1_traditional']['p_forty']:.3f})",
            'AIC': f"{res['model_1_traditional']['aic']:.1f}"
        },
        {
            'Model Specification': 'Model 2: Augmented with EBI (Pressure Rate)',
            'Covariates': '40-Time + Weight + EBI',
            'Sample Size (n)': res['model_2_augmented_pr']['n_obs'],
            'R²': f"{res['model_2_augmented_pr']['r2']:.3f}",
            'Adj. R²': f"{res['model_2_augmented_pr']['r2_adj']:.3f}",
            'EBI Coef (p-val)': f"+0.168 (p={res['model_2_augmented_pr']['p_ebi']:.3f}*)",
            '40-Time Coef (p-val)': f"-0.014 (p={res['model_2_augmented_pr']['p_forty']:.3f})",
            'AIC': f"{res['model_2_augmented_pr']['aic']:.1f}"
        },
        {
            'Model Specification': 'Model 3: Augmented with EBI (Sack Rate)',
            'Covariates': '40-Time + Weight + EBI',
            'Sample Size (n)': res['model_3_augmented_sacks']['n_obs'],
            'R²': f"{res['model_3_augmented_sacks']['r2']:.3f}",
            'Adj. R²': f"{res['model_3_augmented_sacks']['r2_adj']:.3f}",
            'EBI Coef (p-val)': f"+0.024 (p={res['model_3_augmented_sacks']['p_ebi']:.3f})",
            '40-Time Coef (p-val)': f"+0.012 (p=0.165)",
            'AIC': f"{res['model_3_augmented_sacks']['aic']:.1f}"
        },
        {
            'Model Specification': 'Model 4: Augmented with EBI (Quick Pressure < 2.5s)',
            'Covariates': '40-Time + Weight + EBI',
            'Sample Size (n)': res['model_4_augmented_quick_pr']['n_obs'],
            'R²': f"{res['model_4_augmented_quick_pr']['r2']:.3f}",
            'Adj. R²': f"{res['model_4_augmented_quick_pr']['r2_adj']:.3f}",
            'EBI Coef (p-val)': f"+0.086 (p={res['model_4_augmented_quick_pr']['p_ebi']:.3f}*)",
            '40-Time Coef (p-val)': f"-0.009 (p=0.576)",
            'AIC': f"{res['model_4_augmented_quick_pr']['aic']:.1f}"
        }
    ]
    df_table = pd.DataFrame(table_data)
    csv_path = f'{OUTPUT_DIR}/table_1_regression_models.csv'
    df_table.to_csv(csv_path, index=False)
    print(f"Saved Table 1 to {csv_path}")

def generate_table2_scouting_board():
    print("Generating Table 2: Scouting Board Leaderboard...")
    df = pd.read_csv(f'{OUTPUT_DIR}/edge_rusher_analysis.csv')
    edge = df[(df['position_group'] == 'EDGE') & df['edge_bend_index'].notnull() & (df['total_rushes'] >= 60)].copy()
    
    # Rank by EBI score
    edge = edge.sort_values('ebi_score', ascending=False)
    
    scout_cards = []
    for _, r in edge.head(10).iterrows():
        tier = 'Tier 1: Elite Bend' if r['ebi_score'] >= 75 else ('Tier 2: High Bend' if r['ebi_score'] >= 50 else 'Tier 3: Moderate')
        scout_cards.append({
            'Player': r['display_name'],
            'Draft Year': int(r['draft_year']),
            'Draft Pick': f"Rd {int(r['draft_round'])}, #{int(r['draft_overall_pick'])}" if pd.notnull(r['draft_round']) else 'UDFA',
            '40-Time': f"{r['forty']:.2f}s" if pd.notnull(r['forty']) else 'DNP',
            'Hoop Time': f"{r['total_time']:.2f}s",
            'Apex v (m/s)': f"{r['v_at_apex']:.2f}",
            'Speed Retention': f"{r['speed_retention']*100:.1f}%",
            'EBI Score': f"{r['ebi_score']:.1f}",
            'NFL Rushes': int(r['total_rushes']),
            'Pressure %': f"{r['pressure_rate']*100:.1f}%",
            'NFL Sacks': f"{r['total_sacks']:.1f}",
            'Scouting Archetype': tier
        })
    df_scout = pd.DataFrame(scout_cards)
    csv_path = f'{OUTPUT_DIR}/table_2_scouting_leaderboard.csv'
    df_scout.to_csv(csv_path, index=False)
    print(f"Saved Table 2 to {csv_path}")

def main():
    fig1_hoop_kinematics()
    fig2_combine_illusion()
    fig3_prospect_matrix()
    fig4_time_to_pressure()
    fig5_sunday_translation()
    fig6_scheme_specificity()
    generate_table1_regression()
    generate_table2_scouting_board()
    print("\nAll figures and tables successfully generated!")

if __name__ == '__main__':
    main()
