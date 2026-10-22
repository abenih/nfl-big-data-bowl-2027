"""
Process NFL Regular Season Pass Rush Performance & Model Linkage
Calculates player-level and play-level pass rush efficiency, fits Expected Pressure Rate (xPR),
and estimates statistical models linking Combine kinematics to NFL production.
"""

import os
import json
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.metrics import roc_auc_score, r2_score
import statsmodels.api as sm

DATA_DIR = '/home/benji/Downloads/y/nfl-big-data-bowl-2027'
OUTPUT_DIR = '/home/benji/Downloads/nfl_bdb_2027_bend/output'

def process_nfl_plays():
    print("Loading regular season plays from player_play.csv...")
    pp = pl.scan_csv(f'{DATA_DIR}/player_play.csv', null_values=['NA', ''], infer_schema_length=50000)
    
    # Filter pass rush plays (player_get_off is recorded when a defender rushes)
    rush_plays = pp.filter(pl.col('player_get_off').is_not_null()).select([
        'game_id', 'play_id', 'nfl_id', 'down', 'yards_to_go', 'quarter',
        pl.col('player_get_off').cast(pl.Float64),
        pl.col('time_to_pressure').cast(pl.Float64),
        pl.col('time_to_qb_hurry').cast(pl.Float64),
        pl.col('sack').cast(pl.Float64),
        pl.col('tackle_for_loss').cast(pl.Float64),
        pl.col('caused_forced_fumble').cast(pl.Float64)
    ]).collect().to_pandas()
    
    print(f"Total pass rush plays in dataset: {len(rush_plays)}")
    
    # Pressure indicator: 1 if time_to_pressure is recorded, else 0
    rush_plays['is_pressure'] = rush_plays['time_to_pressure'].notnull().astype(int)
    rush_plays['is_quick_pressure'] = ((rush_plays['time_to_pressure'].notnull()) & (rush_plays['time_to_pressure'] < 2.5)).astype(int)
    rush_plays['is_sack'] = (rush_plays['sack'] > 0).astype(int)
    
    # Play-level Expected Pressure (xPR) model
    # Model pressure probability based on down, distance, and quarter
    features = ['down', 'yards_to_go', 'quarter']
    df_clean = rush_plays.dropna(subset=features).copy()
    
    X = pd.get_dummies(df_clean[features], columns=['down', 'quarter'], drop_first=True)
    y = df_clean['is_pressure']
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    df_clean['xPR'] = clf.predict_proba(X)[:, 1]
    df_clean['PROE'] = df_clean['is_pressure'] - df_clean['xPR']
    
    print(f"Fitted xPR model. Baseline pressure rate: {y.mean():.3f}, AUC: {roc_auc_score(y, df_clean['xPR']):.3f}")
    
    # Aggregate to player level
    player_agg = df_clean.groupby('nfl_id').agg(
        total_rushes=('is_pressure', 'count'),
        total_pressures=('is_pressure', 'sum'),
        total_quick_pressures=('is_quick_pressure', 'sum'),
        total_sacks=('sack', 'sum'),
        total_tfl=('tackle_for_loss', 'sum'),
        total_ff=('caused_forced_fumble', 'sum'),
        mean_get_off=('player_get_off', 'mean'),
        std_get_off=('player_get_off', 'std'),
        mean_ttp=('time_to_pressure', 'mean'),
        mean_xpr=('xPR', 'mean'),
        total_proe=('PROE', 'sum'),
        mean_proe=('PROE', 'mean')
    ).reset_index()
    
    player_agg['pressure_rate'] = player_agg['total_pressures'] / player_agg['total_rushes']
    player_agg['quick_pressure_rate'] = player_agg['total_quick_pressures'] / player_agg['total_rushes']
    player_agg['sack_rate'] = player_agg['total_sacks'] / player_agg['total_rushes']
    
    return player_agg

def main():
    nfl_stats = process_nfl_plays()
    
    # Load player metadata and combine results
    players = pl.read_csv(f'{DATA_DIR}/players.csv', null_values=['NA', '']).to_pandas()
    cb_res = pl.read_csv(f'{DATA_DIR}/combine_results.csv', null_values=['NA', '']).to_pandas()
    kinematics = pd.read_csv(f'{OUTPUT_DIR}/combine_edge_kinematics.csv')
    
    # Clean combine results columns
    numeric_cb_cols = ['combine_height', 'combine_weight', 'hand_size', 'arm_length', 'wing_span', 
                       'ten_yd_split', 'forty', 'vertical', 'broad_jump', 'three_cone', 'short_shuttle']
    for c in numeric_cb_cols:
        if c in cb_res.columns:
            cb_res[c] = pd.to_numeric(cb_res[c], errors='coerce')
            
    # Merge all datasets
    merged = players.merge(cb_res, on='nfl_id', how='left') \
                    .merge(kinematics, on='nfl_id', how='left') \
                    .merge(nfl_stats, on='nfl_id', how='inner')
                    
    # Define position categories
    merged['is_edge'] = merged['nfl_position'].isin(['DE', 'OLB']).astype(int)
    merged['is_interior'] = merged['nfl_position'].isin(['DT', 'NT']).astype(int)
    merged['position_group'] = np.where(merged['is_edge'] == 1, 'EDGE', np.where(merged['is_interior'] == 1, 'iDL', 'OTHER'))
    
    merged_path = f'{OUTPUT_DIR}/edge_rusher_analysis.csv'
    merged.to_csv(merged_path, index=False)
    print(f"Saved merged analysis dataset to {merged_path} ({len(merged)} players)")
    
    # Run statistical regressions specifically for qualified EDGE rushers
    edge_df = merged[(merged['position_group'] == 'EDGE') & (merged['total_rushes'] >= 60)].copy()
    print(f"\n=======================================================")
    print(f"STATISTICAL MODELING ON QUALIFIED EDGE RUSHERS (n={len(edge_df)})")
    print(f"=======================================================")
    
    # Model 1: Baseline Traditional (40-yard dash + Weight)
    m1_data = edge_df.dropna(subset=['pressure_rate', 'forty', 'combine_weight'])
    X1 = sm.add_constant(m1_data[['forty', 'combine_weight']])
    y_pr = m1_data['pressure_rate']
    ols1 = sm.OLS(y_pr, X1).fit()
    print("\n--- MODEL 1: Traditional Metrics vs Pressure Rate ---")
    print(ols1.summary())
    
    # Model 2: Augmented with Combine Hoop Kinematics (Edge Bend Index)
    m2_data = edge_df.dropna(subset=['pressure_rate', 'forty', 'combine_weight', 'edge_bend_index'])
    X2 = sm.add_constant(m2_data[['forty', 'combine_weight', 'edge_bend_index']])
    y_pr2 = m2_data['pressure_rate']
    ols2 = sm.OLS(y_pr2, X2).fit()
    print("\n--- MODEL 2: Augmented with Edge Bend Index ---")
    print(ols2.summary())
    
    # Model 3: Sack Rate prediction (Sacks per rush)
    m3_data = edge_df.dropna(subset=['sack_rate', 'forty', 'combine_weight', 'edge_bend_index'])
    X3 = sm.add_constant(m3_data[['forty', 'combine_weight', 'edge_bend_index']])
    y_sack = m3_data['sack_rate']
    ols3 = sm.OLS(y_sack, X3).fit()
    print("\n--- MODEL 3: Augmented with EBI vs Sack Rate ---")
    print(ols3.summary())
    
    # Model 4: Quick Pressure Rate (< 2.5s)
    m4_data = edge_df.dropna(subset=['quick_pressure_rate', 'forty', 'combine_weight', 'edge_bend_index'])
    X4 = sm.add_constant(m4_data[['forty', 'combine_weight', 'edge_bend_index']])
    y_qpr = m4_data['quick_pressure_rate']
    ols4 = sm.OLS(y_qpr, X4).fit()
    print("\n--- MODEL 4: Augmented with EBI vs Quick Pressure Rate (< 2.5s) ---")
    print(ols4.summary())
    
    # Save model summary metrics
    results_summary = {
        'model_1_traditional': {
            'r2': float(ols1.rsquared),
            'r2_adj': float(ols1.rsquared_adj),
            'aic': float(ols1.aic),
            'bic': float(ols1.bic),
            'p_forty': float(ols1.pvalues.get('forty', np.nan)),
            'n_obs': int(ols1.nobs)
        },
        'model_2_augmented_pr': {
            'r2': float(ols2.rsquared),
            'r2_adj': float(ols2.rsquared_adj),
            'aic': float(ols2.aic),
            'bic': float(ols2.bic),
            'p_ebi': float(ols2.pvalues.get('edge_bend_index', np.nan)),
            'p_forty': float(ols2.pvalues.get('forty', np.nan)),
            'n_obs': int(ols2.nobs)
        },
        'model_3_augmented_sacks': {
            'r2': float(ols3.rsquared),
            'r2_adj': float(ols3.rsquared_adj),
            'aic': float(ols3.aic),
            'p_ebi': float(ols3.pvalues.get('edge_bend_index', np.nan)),
            'n_obs': int(ols3.nobs)
        },
        'model_4_augmented_quick_pr': {
            'r2': float(ols4.rsquared),
            'r2_adj': float(ols4.rsquared_adj),
            'aic': float(ols4.aic),
            'p_ebi': float(ols4.pvalues.get('edge_bend_index', np.nan)),
            'n_obs': int(ols4.nobs)
        }
    }
    
    with open(f'{OUTPUT_DIR}/model_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nModel results successfully saved to {OUTPUT_DIR}/model_results.json")

if __name__ == '__main__':
    main()

