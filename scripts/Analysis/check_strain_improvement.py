import pandas as pd
import numpy as np
from scipy import stats
import os


def analyze_timepoint_improvement(csv_path="Experiments/Question_1_new/experiment_results_summary.csv"):
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}.")
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['NRMSE_All_Species'])
    
    df_hybrid = df[df['ModelType'] == 'Hybrid']
    
    baseline_tp = 7
    strain_sizes = sorted(df_hybrid['Strains'].unique())
    
    results = []
    
    for strain in strain_sizes:
        group = df_hybrid[df_hybrid['Strains'] == strain]
        
        baseline_vals = group[group['Steps'] == baseline_tp]['NRMSE_All_Species'].values
        higher_vals = group[group['Steps'] > baseline_tp]['NRMSE_All_Species'].values
        
        if len(baseline_vals) < 2 or len(higher_vals) < 2:
            continue
            
        t_stat, p_val = stats.ttest_ind(baseline_vals, higher_vals, equal_var=False)
        
        mean_baseline = np.mean(baseline_vals)
        mean_higher = np.mean(higher_vals)
        pct_improvement = (mean_baseline - mean_higher) / mean_baseline * 100
        
        is_significant = p_val < 0.05 and mean_higher < mean_baseline
        
        results.append({
            "Strains": strain,
            "Mean_NRMSE_7TP": mean_baseline,
            "Mean_NRMSE_>7TP": mean_higher,
            "Pct_Improvement": pct_improvement,
            "P_Value": p_val,
            "Significant": "yes" if is_significant else "no"
        })

    results_df = pd.DataFrame(results)
    
    print("\n=== Hybrid Model: Timepoint Improvement Analysis (7 vs >7 TPs) ===")
    print(results_df.to_string(index=False))
    
    results_df.to_csv("Experiments/Question_1_new/timepoint_improvement_analysis.csv", index=False)
    print(f"\nAnalysis saved to Experiments/Question_1_new/timepoint_improvement_analysis.csv")

if __name__ == "__main__":
    analyze_timepoint_improvement()
