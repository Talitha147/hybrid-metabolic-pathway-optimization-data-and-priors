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

METABOLITE_NAMES = ["substrate", "c_biomass", "co2", "e4p", "pep", "dhap", "epsp", "pcoumaric_acid"]
PCA_INDEX = 7  

def parse_scenario_q3(exp_name):
    """
    Parses Q3 experiment name to extract the Scenario.
    Example: pCA_lumped_1_part_1_14_points_200_strains -> lumped_1_part_1
    """
    name = exp_name
    if name.startswith("pCA_"):
        name = name[4:]
    
    # The name ends with '_X_points_Y_strains'
    match = re.search(r"^(.*)_(\d+)_points_(\d+)_strains$", name)
    if match:
        return match.group(1)
    
    return "Unknown"

def prepare_dataframe_q3(df):
    """Post-process the dataframe for Question 3."""
    df = vr.prepare_dataframe(df)
    
    # In Q3, we want to use Scenario (lumped_X_part_Y) as the primary category for plotting.
    df['Scenario'] = df['Experiment'].apply(parse_scenario_q3)
    
    # Rename Part 1 models
    q3_part1_map = {
        "lumped_1_part_1": "Fully lumped",
        "lumped_2_part_1": "Highly lumped",
        "lumped_3_part_1": "Partially lumped"
    }
    # Rename Part 2 models
    q3_part2_map = {
        "lumped_1_part_2": "Fully masked - lumped",
        "lumped_2_part_2": "6 unknown - lumped",
        "lumped_3_part_2": "5 unknown - lumped"
    }
    df['Scenario'] = df['Scenario'].replace(q3_part1_map)
    df['Scenario'] = df['Scenario'].replace(q3_part2_map)
    df['ModelType'] = df['Scenario']
    
    return df

    
def run_plotting_suite(df, plot_dir):
    """Runs the full suite of visualization functions for a given dataframe and directory."""
    if df.empty:
        print(f"No data for plots in {plot_dir}. Skipping.")
        return

    print(f"Generating plots in {plot_dir}...")
    
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

    has_trajectories = "Predictions" in df.columns and "GroundTruths" in df.columns


def main():
    use_csv = True

    base_dir = "Experiments/Question_3"
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

    # Order of scenarios for Question 3
    scenario_order = [
        "Fully lumped", "Highly lumped", "Partially lumped",
        "Fully masked - lumped", "6 unknown - lumped", "5 unknown - lumped"
    ]
    
    df = prepare_dataframe_q3(df)
    
    df['Scenario'] = pd.Categorical(df['Scenario'], categories=scenario_order, ordered=True)
    df['ModelType'] = pd.Categorical(df['ModelType'], categories=scenario_order, ordered=True)
    
    plot_dir = vr.setup_plot_dir("Figures/Question_3")

    run_plotting_suite(df, plot_dir)

    df_part1 = df[df['Scenario'].isin(["Fully lumped", "Highly lumped", "Partially lumped"])].copy()
    df_part1['Scenario'] = df_part1['Scenario'].cat.remove_unused_categories()
    df_part1['ModelType'] = df_part1['ModelType'].cat.remove_unused_categories()
    
    plot_dir_p1 = plot_dir / "part_1"
    plot_dir_p1.mkdir(exist_ok=True)
    run_plotting_suite(df_part1, plot_dir_p1)

    
    df_part2 = df[df['Scenario'].isin(["Fully masked - lumped", "6 unknown - lumped", "5 unknown - lumped"])].copy()
    df_part2['Scenario'] = df_part2['Scenario'].cat.remove_unused_categories()
    df_part2['ModelType'] = df_part2['ModelType'].cat.remove_unused_categories()
    
    plot_dir_p2 = plot_dir / "part_2"
    plot_dir_p2.mkdir(exist_ok=True)
    run_plotting_suite(df_part2, plot_dir_p2)

    print(f"\nAll plots saved to {plot_dir}")
    print(f"Pipeline complete.")

if __name__ == "__main__":
    main()
