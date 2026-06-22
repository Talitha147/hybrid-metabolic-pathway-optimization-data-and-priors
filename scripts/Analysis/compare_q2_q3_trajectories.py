import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent))
import visualize_results as vr

# Scenarios to compare with eachother 
Q2_SCENARIOS = ["All_masked", "All_known_from_substrate", "Reactions_to_dahp_known"]
Q3_SCENARIOS = ["lumped_1_part_2", "lumped_2_part_2", "lumped_3_part_2"]

# Mapping of species indices for the lumped models
# All species: [substrate, biomass, co2, e4p, pep, dhap, epsp, pcoumaric_acid]
PAIR_SPECIES_INDICES = {
    0: [0, 7],                      
    1: [0, 1, 2, 3, 4, 7],          
    2: [0, 1, 2, 3, 4, 5, 7]        
}

SCENARIO_LABELS = ["Fully masked", "6 Unknowns", "5 Unknowns"]

def load_detailed_results(base_path, model_scenarios=None, is_q2=False, is_q3=False, is_q1=False):

    results_list = []
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Warning: {base_path} not found.")
        return pd.DataFrame()
    
    folders = [f for f in base_path.iterdir() if f.is_dir()]
    
    for folder in folders:
        if is_q2:
            from visualize_results_q2 import parse_scenario_q2
            model_type = parse_scenario_q2(folder.name)
        elif is_q3:
            from visualize_results_q3 import parse_scenario_q3
            model_type = parse_scenario_q3(folder.name)
        elif is_q1:
            model_type = "hybrid" # Q1 is just 'hybrid'
        else:
            continue
            
        if model_scenarios and model_type not in model_scenarios:
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
            
        # If Q1, we compare it against ALL 3 lumped scenarios
        target_pairs = range(3) if is_q1 else ([model_scenarios.index(model_type)] if model_scenarios else [0])
        
        # Look into subsets and seeds
        for subset_dir in folder.glob("subset_*"):
            for seed_dir in subset_dir.glob("seed_*"):
                results_path = seed_dir / "results.json"
                if not results_path.exists():
                    continue
                    
                with open(results_path, "r") as f:
                    data = json.load(f)
                    
                nrmse_per_species = data.get("NRMSE_per_species", [])
                if not nrmse_per_species:
                    continue
                
                for pair_idx in target_pairs:
                    # For Q1 and Q2, we need to filter these species based on the pair_idx
                    if is_q2 or is_q1:
                        relevant_indices = PAIR_SPECIES_INDICES[pair_idx]
                        filtered_nrmse = [nrmse_per_species[i] for i in relevant_indices if i < len(nrmse_per_species)]
                        mean_nrmse = np.mean(filtered_nrmse) if filtered_nrmse else np.nan
                    else:
                        # For Q3, the species in nrmse_per_species are already the relevant ones
                        mean_nrmse = np.mean(nrmse_per_species)
                    
                    results_list.append({
                        "Scenario": SCENARIO_LABELS[pair_idx],
                        "Question": "Baseline (Q1)" if is_q1 else ("Full Model (Q2)" if is_q2 else "Lumped Model (Q3)"),
                        "Strains": strains,
                        "Steps": steps,
                        "NRMSE_pCA_Final": data.get("NRMSE_pCA_Final", np.nan),
                        "NRMSE_Metabolites": mean_nrmse,
                        "Subset": subset_dir.name,
                        "Seed": seed_dir.name
                    })
                
    return pd.DataFrame(results_list)

def main():
    print("Loading detailed results for Questions 1, 2, and 3 ...")
    
    df_q1 = load_detailed_results("Experiments/Question_1/Hybrid", is_q1=True)
    df_q2 = load_detailed_results("Experiments/Question_2", Q2_SCENARIOS, is_q2=True)
    df_q3 = load_detailed_results("Experiments/Question_3", Q3_SCENARIOS, is_q3=True)
    
    if df_q2.empty and df_q3.empty:
        print("Error: Could not load data.")
        return
        
    df_all = pd.concat([df_q2, df_q3], ignore_index=True)

   
    print("\nPerforming statistical significance tests (Paired T-Tests)...")
    from scipy.stats import ttest_rel
    
    stats_results = []
    
    for scenario in SCENARIO_LABELS:
        for strains in sorted(df_all["Strains"].unique()):
            for steps in sorted(df_all["Steps"].unique()):
                q2_data = df_all[(df_all["Scenario"] == scenario) & 
                                 (df_all["Question"] == "Full Model (Q2)") & 
                                 (df_all["Strains"] == strains) & 
                                 (df_all["Steps"] == steps)]
                
                q3_data = df_all[(df_all["Scenario"] == scenario) & 
                                 (df_all["Question"] == "Lumped Model (Q3)") & 
                                 (df_all["Strains"] == strains) & 
                                 (df_all["Steps"] == steps)]
                
                if q2_data.empty or q3_data.empty:
                    continue
                
                merged = pd.merge(q2_data, q3_data, on=["Subset", "Seed"], suffixes=('_Q2', '_Q3'))
                if len(merged) < 2:
                    continue
                
                t_pca, p_pca = ttest_rel(merged["NRMSE_pCA_Final_Q2"], merged["NRMSE_pCA_Final_Q3"])
                t_met, p_met = ttest_rel(merged["NRMSE_Metabolites_Q2"], merged["NRMSE_Metabolites_Q3"])
                
                stats_results.append({
                    "Scenario": scenario,
                    "Strains": strains,
                    "Steps": steps,
                    "Mean_Q2_pCA": merged["NRMSE_pCA_Final_Q2"].mean(),
                    "Mean_Q3_pCA": merged["NRMSE_pCA_Final_Q3"].mean(),
                    "Mean_Q2_Metab": merged["NRMSE_Metabolites_Q2"].mean(),
                    "Mean_Q3_Metab": merged["NRMSE_Metabolites_Q3"].mean(),
                    "P_Value_pCA": p_pca,
                    "Significant_pCA": p_pca < 0.05,
                    "P_Value_Metab": p_met,
                    "Significant_Metab": p_met < 0.05
                })
   
    stats_df = pd.DataFrame(stats_results)
    output_dir = Path("Experiments/Comparison_Summaries")
    output_dir.mkdir(exist_ok=True)
    stats_df.to_csv(output_dir / "q2_q3_statistical_significance.csv", index=False)

    sns.set_theme(style="whitegrid")

    
    def plot_scaling_with_highlights(metric, ylabel, filename, title):
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
        
      
        HIGHLIGHT_COLOR = "red"
        
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
            
            # Add Highlights for where Q3 is significantly better
            scenario_stats = stats_df[stats_df["Scenario"] == scenario]
            better_q3_points = []
            
            for _, row in scenario_stats.iterrows():
                sig_col = "Significant_pCA" if "pCA" in metric else "Significant_Metab"
                mean_q2 = row[f"Mean_Q2_{'pCA' if 'pCA' in metric else 'Metab'}"]
                mean_q3 = row[f"Mean_Q3_{'pCA' if 'pCA' in metric else 'Metab'}"]
                
                if row[sig_col] and mean_q3 < mean_q2:
                    better_q3_points.append(row)
            
            if better_q3_points:
                better_df = pd.DataFrame(better_q3_points)
                # Plot a tiny red circle on top of the significant points
                ax.scatter(better_df["Steps"], 
                           better_df[f"Mean_Q3_{'pCA' if 'pCA' in metric else 'Metab'}"], 
                           color="red", marker="o", s=15, zorder=10, label='_nolegend_')
            
            ax.set_title(f"{scenario}", fontsize=14, fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Timepoints")


            handles, labels = ax.get_legend_handles_labels()

            # Remove section headers
            filtered = [
                (h, l) for h, l in zip(handles, labels)
                if l not in ["Strains", "Question"]
            ]

            handles, labels = zip(*filtered)

            ax.legend(handles, labels, title="Strains & Model Type", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            
            if i != 2:
                ax.get_legend().remove()
            # else:
            #     ax.legend(title="Strains & Model Type", bbox_to_anchor=(1.05, 1), loc='upper left')

        # plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {output_dir / filename}")

    plot_scaling_with_highlights("NRMSE_pCA_Final", "NRMSE (p-CA final)", 
                                 "q2_q3_pca_error_scaling_highlight.png", 
                                 "Final pCA Prediction Error (Red = Q3 significantly better)")
    
    plot_scaling_with_highlights("NRMSE_Metabolites", "NRMSE Metabolites (%)", 
                                 "q2_q3_metabolite_error_scaling_highlight.png", 
                                 "Metabolite Trajectory Error (Red = Q3 significantly better)")

   

if __name__ == "__main__":
    main()
