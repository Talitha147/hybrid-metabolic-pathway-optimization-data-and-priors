import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
from scipy.stats import ttest_rel

# Configuration
Q3_SINK_SCENARIOS = {
    "lumped_1_part_1": "Fully lumped",
    "lumped_2_part_1": "Highly lumped",
    "lumped_1_part_2": "Only substrate and product"
}

Q3_BALANCED_SINK_SCENARIOS = {
    "lumped_1_part_1_sink_balanced": "Fully lumped",
    "lumped_2_part_1_sink_balanced": "Highly lumped",
    "lumped_1_part_2_sink_balanced": "Only substrate and product"
}

SCENARIO_LABELS = ["Fully lumped", "Highly lumped", "Only substrate and product"]

def parse_scenario_q3(exp_name):

    name = exp_name
    if name.startswith("pCA_"):
        name = name[4:]
    # Ends with _X_points_Y_strains
    import re
    match = re.search(r"^(.*)_(\d+)_points_(\d+)_strains$", name)
    if match:
        return match.group(1)
    return "Unknown"

def load_detailed_results(base_path, scenario_mapping, label):
  
    results_list = []
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Warning: {base_path} not found.")
        return pd.DataFrame()
    
    folders = [f for f in base_path.iterdir() if f.is_dir()]
    
    for folder in folders:
        model_type = parse_scenario_q3(folder.name)
        if model_type not in scenario_mapping:
            continue
            
        # Parse strains and steps from folder name
        parts = folder.name.split("_")
        try:
            strains_idx = -2
            steps_idx = -4
            strains = int(parts[strains_idx])
            steps = int(parts[steps_idx])
        except (ValueError, IndexError):
            continue
            
        if strains == 500:
            continue
            
        for subset_dir in folder.glob("subset_*"):
            for seed_dir in subset_dir.glob("seed_*"):
                results_path = seed_dir / "results.json"
                if not results_path.exists():
                    continue
                    
                with open(results_path, "r") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        continue
                    
                nrmse_per_species = data.get("NRMSE_per_species", [])
                if not nrmse_per_species:
                    continue
                
                # For Q3, the species in nrmse_per_species are already the relevant ones
                mean_nrmse = np.mean(nrmse_per_species)
                
                results_list.append({
                    "Scenario": scenario_mapping[model_type],
                    "Question": label,
                    "Strains": strains,
                    "Steps": steps,
                    "NRMSE_pCA_Final": data.get("NRMSE_pCA_Final", np.nan),
                    "NRMSE_Metabolites": mean_nrmse,
                    "NRMSE_All_Species": data.get("NRMSE_All_Species", np.nan),
                    "Subset": subset_dir.name,
                    "Seed": seed_dir.name
                })
                
    return pd.DataFrame(results_list)

def main():
    print("Loading detailed results for Q3 Sink and Q3 Balanced Sink...")
    
    df_sink = load_detailed_results(
        "Experiments/Question_3",
        Q3_SINK_SCENARIOS,
        "Lumped Model"
    )
    
    df_balanced = load_detailed_results(
        "Experiments/Question_3_balanced",
        Q3_BALANCED_SINK_SCENARIOS,
        "Balanced Lumped Model"
    )
    
    if df_sink.empty or df_balanced.empty:
        print("Error: Could not load data. Please ensure both experiment suites have run.")
        return
        
    df_all = pd.concat([df_sink, df_balanced], ignore_index=True)
    
    print("\nPerforming statistical significance tests (Paired T-Tests)...")
    
    stats_results = []
    
    for scenario in SCENARIO_LABELS:
        for strains in sorted(df_all["Strains"].unique()):
            for steps in sorted(df_all["Steps"].unique()):
                sink_data = df_all[(df_all["Scenario"] == scenario) & 
                                   (df_all["Question"] == "Lumped Model") & 
                                   (df_all["Strains"] == strains) & 
                                   (df_all["Steps"] == steps)]
                
                bal_data = df_all[(df_all["Scenario"] == scenario) & 
                                  (df_all["Question"] == "Balanced Lumped Model") & 
                                  (df_all["Strains"] == strains) & 
                                  (df_all["Steps"] == steps)]
                
                if sink_data.empty or bal_data.empty:
                    continue
                
                merged = pd.merge(sink_data, bal_data, on=["Subset", "Seed"], suffixes=('_Sink', '_Bal'))
                if len(merged) < 2:
                    continue
                
                # Check for NaNs or constants
                if merged["NRMSE_pCA_Final_Sink"].nunique() <= 1 and merged["NRMSE_pCA_Final_Bal"].nunique() <= 1:
                    p_pca = 1.0
                    t_pca = 0.0
                else:
                    t_pca, p_pca = ttest_rel(merged["NRMSE_pCA_Final_Sink"], merged["NRMSE_pCA_Final_Bal"])
                    
                if merged["NRMSE_Metabolites_Sink"].nunique() <= 1 and merged["NRMSE_Metabolites_Bal"].nunique() <= 1:
                    p_met = 1.0
                    t_met = 0.0
                else:
                    t_met, p_met = ttest_rel(merged["NRMSE_Metabolites_Sink"], merged["NRMSE_Metabolites_Bal"])

                if merged["NRMSE_All_Species_Sink"].nunique() <= 1 and merged["NRMSE_All_Species_Bal"].nunique() <= 1:
                    p_all = 1.0
                    t_all = 0.0
                else:
                    t_all, p_all = ttest_rel(merged["NRMSE_All_Species_Sink"], merged["NRMSE_All_Species_Bal"])
                
                if np.isnan(p_pca):
                    p_pca = 1.0
                if np.isnan(p_met):
                    p_met = 1.0
                if np.isnan(p_all):
                    p_all = 1.0
                
                stats_results.append({
                    "Scenario": scenario,
                    "Strains": strains,
                    "Steps": steps,
                    "Mean_Sink_pCA": merged["NRMSE_pCA_Final_Sink"].mean(),
                    "Mean_Bal_pCA": merged["NRMSE_pCA_Final_Bal"].mean(),
                    "Mean_Sink_Metab": merged["NRMSE_Metabolites_Sink"].mean(),
                    "Mean_Bal_Metab": merged["NRMSE_Metabolites_Bal"].mean(),
                    "Mean_Sink_AllSpecies": merged["NRMSE_All_Species_Sink"].mean(),
                    "Mean_Bal_AllSpecies": merged["NRMSE_All_Species_Bal"].mean(),
                    "P_Value_pCA": p_pca,
                    "Significant_pCA": p_pca < 0.05,
                    "P_Value_Metab": p_met,
                    "Significant_Metab": p_met < 0.05,
                    "P_Value_AllSpecies": p_all,
                    "Significant_AllSpecies": p_all < 0.05
                })
                
    stats_df = pd.DataFrame(stats_results)
    output_dir = Path("Experiments/Comparison_Summaries")
    output_dir.mkdir(exist_ok=True)
    stats_df.to_csv(output_dir / "q3_sink_vs_balanced_statistical_significance.csv", index=False)

    sns.set_theme(style="whitegrid")

    def plot_scaling_with_highlights(metric, ylabel, filename, title):
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
        
        for i, scenario in enumerate(SCENARIO_LABELS):
            ax = axes[i]
            subset = df_all[df_all["Scenario"] == scenario]
            
            # Draw the main lineplot
            sns.lineplot(
                data=subset, x="Steps", y=metric, 
                hue="Strains", style="Question", 
                markers=True, dashes=False, palette="viridis", 
                ax=ax
            )
            
            scenario_stats = stats_df[stats_df["Scenario"] == scenario]
            better_bal_points = []
            
            # Map metric to corresponding columns in stats_df
            if "pCA" in metric:
                sig_col = "Significant_pCA"
                mean_sink_col = "Mean_Sink_pCA"
                mean_bal_col = "Mean_Bal_pCA"
            elif "All_Species" in metric:
                sig_col = "Significant_AllSpecies"
                mean_sink_col = "Mean_Sink_AllSpecies"
                mean_bal_col = "Mean_Bal_AllSpecies"
            else:
                sig_col = "Significant_Metab"
                mean_sink_col = "Mean_Sink_Metab"
                mean_bal_col = "Mean_Bal_Metab"
            
            for _, row in scenario_stats.iterrows():
                mean_sink = row[mean_sink_col]
                mean_bal = row[mean_bal_col]
                
                if row[sig_col] and mean_bal < mean_sink:
                    better_bal_points.append(row)
            
          
            
            ax.set_title(f"{scenario}", fontsize=14, fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Timesteps")

            handles, labels = ax.get_legend_handles_labels()

            filtered = [
                (h, l) for h, l in zip(handles, labels)
                if l not in ["Strains", "Question"]
            ]

            handles, labels = zip(*filtered)

            ax.legend(handles, labels, title="Strains & Model Type", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            if i != 2:
                ax.get_legend().remove()
          
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {output_dir / filename}")

    plot_scaling_with_highlights(
        "NRMSE_All_Species", 
        "NRMSE All Species", 
        "q3_sink_vs_balanced_all_species_error_scaling_highlight.png", 
        "All Species Prediction Error (Red = Balanced Lumped Sink significantly better)"
    )

    plot_scaling_with_highlights(
        "NRMSE_pCA_Final", 
        "NRMSE (p-CA final)", 
        "q3_sink_vs_balanced_pca_error_scaling_highlight.png", 
        "Final pCA Prediction Error (Red = Balanced Lumped Sink significantly better)"
    )
    
    plot_scaling_with_highlights(
        "NRMSE_Metabolites", 
        "NRMSE", 
        "q3_sink_vs_balanced_metabolite_error_scaling_highlight.png", 
        "Metabolite Trajectory Error (Red = Balanced Lumped Sink significantly better)"
    )


if __name__ == "__main__":
    main()
