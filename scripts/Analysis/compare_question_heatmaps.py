import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).parent))
import visualize_results as vr

Q3_part1 = ["lumped_1_part_1", "lumped_2_part_1", "lumped_3_part_1"]
Q3_part2 = ["lumped_1_part_2", "lumped_2_part_2", "lumped_3_part_2"] 
# Q2_SCENARIOS_TO_PLOT = ["All_masked", "All_known_from_substrate", "Reactions_to_dahp_known"]
Q2_SCENARIOS_TO_PLOT = ["All masked", "6 Unknowns", "5 Unknowns", "4 Unknowns (Prod + Sink)", "1 Unknown (Prod)"]

METRIC = "NRMSE_pCA_Final"

def load_and_prep(base_dir, scenarios, is_q2=False, is_q3=False):
    csv_path = Path(base_dir) / "experiment_results_summary.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found.")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    df = vr.prepare_dataframe(df)
    
    df = df[df["Strains"] != 500]
    
    if is_q2:
        from visualize_results_q2 import parse_scenario_q2
        df['ModelType'] = df['Experiment'].apply(parse_scenario_q2)
        name_map = {
            'Only_product_unknown': '1 Unknown (Prod)',
            'Only_product_and_sink_unknown': '4 Unknowns (Prod + Sink)',
            'Reactions_to_dahp_known': '5 Unknowns',
            'All_known_from_substrate': '6 Unknowns',
            'All_masked': 'All masked'
        }
        df['ModelType'] = df['ModelType'].replace(name_map)
    elif is_q3:
        from visualize_results_q3 import parse_scenario_q3
        df['ModelType'] = df['Experiment'].apply(parse_scenario_q3)
        
    return df[df["ModelType"].isin(scenarios)]

def main():
    print(f"Loading data for comparison grid (Metric: {METRIC}, excluding 500 strains)...")
    
    # Load data from different questions
    df_q3_part1_sink = load_and_prep("Experiments/Question_3_sink", Q3_part1, is_q3=True)
    df_q3_part2_sink = load_and_prep("Experiments/Question_3_sink", Q3_part2, is_q3=True)
    df_q3_part1 = load_and_prep("Experiments/Question_3", Q3_part1, is_q3=True)
    df_q3_part2 = load_and_prep("Experiments/Question_3", Q3_part2, is_q3=True)
    df_q2 = load_and_prep("Experiments/Question_2", Q2_SCENARIOS_TO_PLOT, is_q2=True)

    # Define the grid structure (Questions in Columns, Scenarios in Rows)
    # Format: (Title, DataFrame, Scenarios_List, Optional_Metric)
    # groups = [
    #     ("Question 3 Sink", df_q3_part2_sink, Q3_part2, "NRMSE_pCA_Final"),
    #     ("Question 2 (Masked)", df_q2, Q2_SCENARIOS_TO_PLOT, "NRMSE_pCA_Final")
    # ]

    # groups = [
    #     ("pCA final", df_q3_part1_sink, Q3_part1, "NRMSE_pCA_Final"),
    #     ("All metabolites (40 timesteps)", df_q3_part1_sink, Q3_part1, "NRMSE_All_Species")
    # ]

    # groups = [
    #     ("pCA final", df_q3_part2_sink, Q3_part2, "NRMSE_pCA_Final"),
    #     ("All metabolites (40 timesteps)", df_q3_part2_sink, Q3_part2, "NRMSE_All_Species")
    # ]

    groups = [
    ("pCA final", df_q2, Q2_SCENARIOS_TO_PLOT, "NRMSE_pCA_Final"),
    ("All metabolites (40 timesteps)", df_q2, Q2_SCENARIOS_TO_PLOT, "NRMSE_All_Species")
    ]
    
    # groups = [
    #     ("pCA final", df_q3_part1, Q3_part1, "NRMSE_pCA_Final"),
    #     ("All metabolites (40 timesteps)", df_q3_part1_sink, Q3_part1, "NRMSE_pCA_Final")
    # ]

    # Filter out empty groups
    groups = [g for g in groups if not g[1].empty]
    
    if not groups:
        print("No data found for any of the specified scenarios.")
        return

    n_cols = len(groups)
    n_rows = max(len(g[2]) for g in groups)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows), 
                             sharex=False, sharey=False, squeeze=False)
    
    sns.set_theme(style="white", font_scale=1.1)

    for col_idx, group in enumerate(groups):
        col_title, df, order = group[0], group[1], group[2]
        # Use group-specific metric if provided, else global default
        group_metric = group[3] if len(group) > 3 else METRIC
        
        scale = vr.GLOBAL_HEATMAP_SCALES.get(group_metric, {"vmin": None, "vmax": None})
        vmin, vmax = scale.get("vmin"), scale.get("vmax")
        
        heatmap_df = df.groupby(["ModelType", "Strains", "steps"])[group_metric].mean().reset_index()
        
        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]
            
            if row_idx < len(order):
                scenario = order[row_idx]
                subset = heatmap_df[heatmap_df["ModelType"] == scenario]
                
                if subset.empty:
                    ax.text(0.5, 0.5, f"No Data:\n{scenario}", ha='center', va='center', fontsize=10)
                    ax.axis('off')
                    continue
                    
                pivot = subset.pivot(index="Strains", columns="steps", values=group_metric)
                
                # Draw heatmap
                sns.heatmap(pivot, ax=ax, annot=True, fmt=".4f" if "RMSE" in group_metric and "N" not in group_metric else ".2f", 
                            cmap="crest", vmin=vmin, vmax=vmax, cbar=True,
                            linewidths=0.5, annot_kws={"size": 9})
                
                # ax.set_title(f"{scenario}\n({group_metric})", fontsize=10, fontweight='bold')
                
                ax.set_ylabel("Strains")
                ax.set_xlabel("Steps")
            else:
                ax.axis('off')
        
        # Add column title (Question Name)
        axes[0, col_idx].text(0.5, 1.3, col_title, transform=axes[0, col_idx].transAxes,
                             fontsize=14, fontweight='bold', ha='center', va='bottom')

    # plt.suptitle("Multi-Question Metric Comparison", fontsize=16, fontweight='bold', y=0.98)

    
    output_dir = Path("Experiments/Comparison_Summaries")
    output_dir.mkdir(exist_ok=True)
    
    # save_path = output_dir / f"comparison_heatmaps_q3_part2_pCAfinal_and_All.png"
    save_path = output_dir / f"comparison_heatmaps_q2_pCAfinal_and_All.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nSuccess! Vertical comparison heatmap saved to: {save_path}")


if __name__ == "__main__":
    main()
