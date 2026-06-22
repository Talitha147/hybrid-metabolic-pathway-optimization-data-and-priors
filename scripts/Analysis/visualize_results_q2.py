import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

import sys
sys.path.append(str(Path(__file__).parent))
from load_experiment_results import load_all_results
import visualize_results as vr

METABOLITE_NAMES = vr.METABOLITE_NAMES
PCA_INDEX = vr.PCA_INDEX

def parse_scenario_q2(exp_name):
    """
    Parses Q2 experiment name to extract the Scenario.
    Example: pCA_All_masked_3_points_8_strains -> All_masked
    """
    name = exp_name
    if name.startswith("pCA_"):
        name = name[4:]
    
    # The name ends with '_X_points_Y_strains'
    match = re.search(r"^(.*)_(\d+)_points_(\d+)_strains$", name)
    if match:
        return match.group(1)
    
    return "Unknown"

def prepare_dataframe_q2(df):
  
    df = vr.prepare_dataframe(df)
    
    df['Scenario'] = df['Experiment'].apply(parse_scenario_q2)
    
    name_map = {
        'Only_product_unknown': '1 Unknown (Prod)',
        'Only_product_and_sink_unknown': '4 Unknowns (Prod + Sink)',
        'Reactions_to_dahp_known': '5 Unknowns',
        'All_known_from_substrate': '6 Unknowns',
        'All_masked': 'All masked'
    }
    df['Scenario'] = df['Scenario'].replace(name_map)
    df['ModelType'] = df['Scenario']
    
    return df

def main():
    # Set to True to load from summary CSV instead of re-scanning all results 
    use_csv = False
    
    base_dir = "Experiments/Question_2"
    pkl_name = "experiment_results_df.pkl"
    csv_name = "experiment_results_summary.csv"
    pkl_path = Path(base_dir) / pkl_name
    csv_path = Path(base_dir) / csv_name
    
    df = None
    if use_csv and csv_path.exists():
        print(f"Loading summary CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
    
    if df is None:
        df = load_all_results(base_dir=base_dir, output_pickle=pkl_name, output_csv=csv_name)

    
    if df is None:
        print("Failed to load results.")
        return

    # Order of masks for Question 2
    mask_order = ["All masked", "6 Unknowns", "5 Unknowns", "4 Unknowns (Prod + Sink)", "1 Unknown (Prod)"]
 
    df = prepare_dataframe_q2(df)
    
    actual_scenarios = [m for m in mask_order if m in df['Scenario'].unique()]
    df['Scenario'] = pd.Categorical(df['Scenario'], categories=actual_scenarios, ordered=True)
    df['ModelType'] = pd.Categorical(df['ModelType'], categories=actual_scenarios, ordered=True)
    
    plot_dir = vr.setup_plot_dir(base_dir)

    has_trajectories = "Predictions" in df.columns and "GroundTruths" in df.columns
    
    # Loss histories
    if "Path" in df.columns:
        vr.plot_subset_loss_histories(df, plot_dir)

    # Global Summaries
    vr.plot_unified_heatmaps(df, plot_dir, orientation='vertical')
    vr.plot_unified_heatmaps(df, plot_dir, metric="RMSE_pCA_Final", orientation='vertical')
    vr.plot_unified_heatmaps(df, plot_dir, metric="RMSE_pCA_All", orientation='vertical')
    vr.plot_unified_heatmaps(df, plot_dir, metric="NRMSE_All_Species", orientation='vertical')
    
    vr.plot_absolute_rmse_box_plots(df, plot_dir)
    vr.plot_train_test_gap(df, plot_dir)
    vr.plot_error_per_subset(df, plot_dir, metric='RMSE_pCA_Final')
    vr.plot_error_per_subset(df, plot_dir, metric='NRMSE_All_Species')
    vr.plot_success_rates_bar(df, plot_dir)
    vr.plot_nan_fractions_bar(df, plot_dir, use_log=False)
    vr.plot_nan_fractions_bar(df, plot_dir, use_log=True)
    vr.plot_metabolite_nrmse_comparison(df, plot_dir)
    vr.plot_rmse_vs_strains_steps_models(df, plot_dir)
    vr.plot_rmse_vs_strains_steps_models(df, plot_dir, metric="NRMSE_All_Species")


    

    print(f"\nAll plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
