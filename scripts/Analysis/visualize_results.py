import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error


METABOLITE_NAMES = ["substrate", "c_biomass", "co2", "e4p", "pep", "dhap", "epsp", "pcoumaric_acid"]
PCA_INDEX = -1  
MODEL_TYPES = ["XGBoost", "Hybrid", "NODE"]
TS = np.linspace(0, 40, 40)

GLOBAL_HEATMAP_SCALES = {
    "NRMSE_pCA_Final": {"vmin": 0, "vmax": 20},
    "RMSE_pCA_Final": {"vmin": 0, "vmax": 0.0025},
    "RMSE_pCA_All": {"vmin": 0, "vmax": 0.004},
    "NRMSE_All_Species": {"vmin": 0, "vmax": 40},
    "Train_RMSE_pCA_Final": {"vmin": 0, "vmax": 0.002},
    "Train_NRMSE_All_Species": {"vmin": 0, "vmax": 40},
}

MODEL_COLORS = {
    "Hybrid": "#2ca02c",   # Green
    "NODE": "#1f77b4",     # Blue
    "XGBoost": "#ff7f0e",  # Orange
    "Kinetic": "#d62728",  # Red
    "Standard": "#7f7f7f", # Gray
    # Q2 Scenarios 
    "All_masked": "#1f77b4",
    "All_known_from_substrate": "#ff7f0e",
    "Reactions_to_dahp_known": "#2ca02c",
    "Only_product_and_sink_unknown": "#d62728",
    "Only_product_unknown": "#9467bd",
    "All masked": "#90d743",
    "6 Unknowns": "#35b779",
    "5 Unknowns": "#21918c",
    "4 Unknowns (Prod + Sink)": "#31688e",
    "1 Unknown (Prod)": "#443983",
    # Q3 Scenarios
    "lumped_1_part_1": "#1f77b4",
    "lumped_2_part_1": "#ff7f0e",
    "lumped_3_part_1": "#2ca02c",
    "Fully lumped": "#1f77b4",
    "Highly lumped": "#ff7f0e",
    "Partially lumped": "#2ca02c",
    "lumped_1_part_2": "#d62728",
    "lumped_2_part_2": "#9467bd",
    "lumped_3_part_2": "#8c564b",
    "Fully masked - lumped": "#d62728",
    "6 unknown - lumped": "#9467bd",
    "5 unknown - lumped": "#8c564b",
}

def get_model_color(model_name):
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
   
def get_grid_dims(n_items, max_cols=3):

    if n_items == 5:
        return 2, 3
    if n_items == 0:
        return 0, 0
    n_cols = min(n_items, max_cols)
    n_rows = (n_items + n_cols - 1) // n_cols
    return n_rows, n_cols

def setup_plot_dir(base_path):
    plot_dir = Path(base_path) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir

def prepare_dataframe(df):

    df = df.copy()
    if 'steps' not in df.columns and 'Steps' in df.columns:
        df['steps'] = df['Steps']
    
    # Ensure numeric
    df['Strains'] = pd.to_numeric(df['Strains'])
    df['steps'] = pd.to_numeric(df['steps'])
    
    if not isinstance(df['ModelType'].dtype, pd.CategoricalDtype):
        df['ModelType'] = df['ModelType'].astype(str)
        
    if 'NRMSE_Train_Test_Gap_pCA_Final' not in df.columns and 'NRMSE_pCA_Final' in df.columns and 'Train_NRMSE_pCA_Final' in df.columns:
        df['NRMSE_Train_Test_Gap_pCA_Final'] = df['NRMSE_pCA_Final'] - df['Train_NRMSE_pCA_Final']

    if 'RMSE_Train_Test_Gap_pCA_Final' not in df.columns and 'RMSE_pCA_Final' in df.columns and 'Train_RMSE_pCA_Final' in df.columns:
        df['RMSE_Train_Test_Gap_pCA_Final'] = df['RMSE_pCA_Final'] - df['Train_RMSE_pCA_Final']

    if 'NRMSE_Train_Test_Gap_all' not in df.columns and 'NRMSE_All_Species' in df.columns and 'Train_NRMSE_All_Species' in df.columns:
        df['NRMSE_Train_Test_Gap_all'] = df['NRMSE_All_Species'] - df['Train_NRMSE_All_Species']
    return df

def get_model_types(df):
    if isinstance(df["ModelType"].dtype, pd.CategoricalDtype):
       
        return [c for c in df["ModelType"].cat.categories if c in df["ModelType"].values]
    return sorted(df["ModelType"].unique())

def plot_unified_heatmaps(df, plot_dir, metric="NRMSE_pCA_Final", orientation='horizontal', vmin=None, vmax=None):

    heatmap_df = df.groupby(["ModelType", "Strains", "steps"])[metric].mean().reset_index()
    if heatmap_df.empty: return

    model_types = get_model_types(heatmap_df)
    n_models = len(model_types)

    if orientation == 'horizontal':
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=False, squeeze=False)
    else:
        fig, axes = plt.subplots(n_models, 1, figsize=(8, 6 * n_models), sharex=True, squeeze=False)
    
    axes_flat = axes.flatten()

    if vmin is None:
        vmin = GLOBAL_HEATMAP_SCALES.get(metric, {}).get('vmin')
    if vmax is None:
        vmax = GLOBAL_HEATMAP_SCALES.get(metric, {}).get('vmax')
    
    if vmin is None:
        vmin = heatmap_df[metric].min()
    if vmax is None:
        vmax = heatmap_df[metric].max()
        
    if vmin == vmax: vmin -= 0.1; vmax += 0.1

    sns.set_theme(style="white", font_scale=1.1)

    for i, mtype in enumerate(model_types):
        ax = axes_flat[i]
        subset = heatmap_df[heatmap_df["ModelType"] == mtype]
        pivot = subset.pivot(index="Strains", columns="steps", values=metric)
        
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".4f", cmap="crest", 
                    vmin=vmin, vmax=vmax, linewidths=0.5, cbar=False,
                    annot_kws={"size": 8})
        ax.set_title(f"{mtype}", fontsize=13, weight="bold")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Strains")

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="crest", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, location="right", shrink=0.8)

    plt.savefig(plot_dir / f"heatmap_mean_{metric}.png", dpi=150, bbox_inches='tight')
    plt.close()





def plot_absolute_rmse_box_plots(df, plot_dir):

    print("Generating absolute RMSE box plots...")

    for metric in [
        "RMSE_pCA_Final",
        "Train_RMSE_pCA_Final",
        "RMSE_All_Species_Train_TS",
        "Train_NRMSE_All_Species_Train_TS",
        "RMSE_All_Species_All_TS",
        "NRMSE_All_Species",
        "Train_NRMSE_All_Species"
    ]:

        if metric not in df.columns or df[metric].isna().all():
            continue

        m_types = sorted(df["ModelType"].unique())
        palette = {m: get_model_color(m) for m in m_types}

        g = sns.catplot(
            data=df,
            x="Strains",
            y=metric,
            hue="ModelType",
            col="steps",
            kind="box",
            height=4,
            aspect=1.2,
            sharey=True,
            palette=palette,
            col_wrap=3
        )

        for ax, step in zip(g.axes.flat, g.col_names):
            step_int = int(step)
            label = "Timepoint" if step_int == 1 else "Timepoints"
            ax.set_title(f"{step_int} {label}")

        for ax in g.axes.flat:
            ax.tick_params(labelbottom=True)
            ax.tick_params(labelleft=True)

        for ax in g.axes.flat:
            ax.set_xlabel("Strains")
            if "NRMSE" in metric:
                ax.set_ylabel("NRMSE")
            else:
                ax.set_ylabel("RMSE")

        plt.savefig(
            plot_dir / f"boxplot_{metric}.png",
            dpi=150,
            bbox_inches="tight"
        )
        plt.close()


def plot_train_test_gap(df, plot_dir):
    print("Generating Train-Test NRMSE gap plots...")
   
    
    m_types = get_model_types(df[df["NRMSE_Train_Test_Gap_pCA_Final"].notna()])
    palette = {m: get_model_color(m) for m in m_types}

    g = sns.catplot(data=df, x="Strains", y="NRMSE_Train_Test_Gap_pCA_Final", hue="ModelType", col="steps", 
                    kind="box", height=4, aspect=1.2, sharey=True, palette=palette, col_wrap=3)

    for ax in g.axes.flat:
        ax.tick_params(labelbottom=True)
        ax.tick_params(labelleft=True)

    for ax, step in zip(g.axes.flat, g.col_names):
        step_int = int(step)
        label = "Timepoint" if step_int == 1 else "Timepoints"
        ax.set_title(f"{step_int} {label}")
    
    for ax in g.axes.flat:
        ax.set_xlabel("Strains") 
        ax.set_ylabel("NRMSE Gap")
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.savefig(plot_dir / "train_test_gap_final_pca_nrmse_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()


    m_types = get_model_types(df[df["RMSE_Train_Test_Gap_pCA_Final"].notna()])
    palette = {m: get_model_color(m) for m in m_types}

    g = sns.catplot(data=df, x="Strains", y="RMSE_Train_Test_Gap_pCA_Final", hue="ModelType", col="steps", 
                    kind="box", height=4, aspect=1.2, sharey=True, palette=palette, col_wrap=3)

    for ax in g.axes.flat:
        ax.tick_params(labelbottom=True)
        ax.tick_params(labelleft=True)

    for ax, step in zip(g.axes.flat, g.col_names):
        step_int = int(step)
        label = "Timepoint" if step_int == 1 else "Timepoints"
        ax.set_title(f"{step_int} {label}")

    for ax in g.axes.flat:
        ax.set_xlabel("Strains")
        ax.set_ylabel("RMSE Gap")
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.savefig(plot_dir / "train_test_gap_final_pca_rmse_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()


    m_types = get_model_types(df[df["NRMSE_Train_Test_Gap_all"].notna()])
    palette = {m: get_model_color(m) for m in m_types}

    g = sns.catplot(data=df, x="Strains", y="NRMSE_Train_Test_Gap_all", hue="ModelType", col="steps", 
                    kind="box", height=4, aspect=1.2, sharey=True, palette=palette, col_wrap=3)

    for ax in g.axes.flat:
        ax.tick_params(labelbottom=True)
        ax.tick_params(labelleft=True)
    
    for ax, step in zip(g.axes.flat, g.col_names):
        step_int = int(step)
        label = "Timepoint" if step_int == 1 else "Timepoints"
        ax.set_title(f"{step_int} {label}")
    
    for ax in g.axes.flat:
        ax.set_xlabel("Strains")
        ax.set_ylabel("NRMSE Gap")
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.savefig(plot_dir / "train_test_gap_nrmse_all_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_success_rates_bar(df, plot_dir):
   

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    print("Generating nested clustered success rate plot...")

    df['is_success'] = (df['Status'] == 'success').astype(int)

    # Calculate success rate
    grouped = df.groupby(['ModelType', 'steps', 'Strains'])['is_success'].agg(['sum','count']).reset_index()
    grouped['Success Rate (%)'] = (grouped['sum'] / grouped['count']) * 100

    model_types = grouped['ModelType'].unique()
    steps_list = sorted(grouped['steps'].unique())
    strains = sorted(grouped['Strains'].unique())

    n_steps = len(steps_list)
    n_strains = len(strains)

    bar_width = 4
    step_gap = 2
    model_gap = 6

    fig, ax = plt.subplots(figsize=(14, 6))

    cmap = cm.get_cmap("viridis", n_strains)
    strain_colors = {strain: cmap(i) for i, strain in enumerate(strains)}

    x_positions = []
    x_labels = []
    current_x = 0

    for model in model_types:
        model_start = current_x

        for step in steps_list:
            step_start = current_x

            for strain in strains:
                subset = grouped[
                    (grouped['ModelType'] == model) &
                    (grouped['steps'] == step) &
                    (grouped['Strains'] == strain)
                ]

                value = subset['Success Rate (%)'].values[0] if not subset.empty else 0

                shade_factor = 0.85 + 0.15 * (
                    steps_list.index(step) / (n_steps - 1 if n_steps > 1 else 1)
                )
                base_color = strain_colors[strain]
                color = tuple([c * shade_factor for c in base_color[:3]] + [1])

                ax.bar(current_x, value, width=bar_width, color=color)

                # x_positions.append(current_x)
                # x_labels.append(str(strain))

                current_x += bar_width

            # space between strain clusters
            current_x += step_gap

            # label step
            step_center = step_start + (n_strains * bar_width) / 2
            ax.text(step_center, -10, f"{step}", ha='center', va='top', fontsize=8, alpha=0.8)

        # space between models
        current_x += model_gap

        model_center = model_start + (
            (len(steps_list) * (n_strains * bar_width + step_gap)) / 2
        )
        # Use rotation and adjusted vertical alignment for long scenario names
        ax.text(model_center, -25, model, ha='center', va='top', fontsize=9, fontweight='bold', rotation=25)

    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105)
    # ax.set_xticks(x_positions)
    # ax.set_xticklabels(x_labels, rotation=0)
    ax.set_xticks([])


    for strain in strains:
        ax.bar(0, 0, color=strain_colors[strain], label=strain)

    ax.legend(title="Strains", bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()

    plt.savefig(plot_dir / "success_rates_nested_clusters.png", dpi=150)
    plt.close()


def plot_nan_fractions_bar(df, plot_dir, use_log=False):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    suffix = "_log" if use_log else ""
  
    col = 'eval_excluded_fraction'
    if col not in df.columns:
        if 'eval_excluded_points' in df.columns and 'eval_total_points' in df.columns:
            df[col] = df['eval_excluded_points'] / df['eval_total_points']
        else:
            print(f"Column '{col}' not found. Skipping NaN fraction plot.")
            return

    plot_df = df[df[col].notna()].copy()
    if plot_df.empty:
        print("No NaN fraction data available. Skipping plot.")
        return

    grouped = plot_df.groupby(['ModelType', 'steps', 'Strains'])[col].mean().reset_index()
    grouped[f'{col} (%)'] = grouped[col] * 100

    model_types = grouped['ModelType'].unique()
    steps_list = sorted(grouped['steps'].unique())
    strains = sorted(grouped['Strains'].unique())

    n_steps = len(steps_list)
    n_strains = len(strains)

    bar_width = 4
    step_gap = 2
    model_gap = 6

    fig, ax = plt.subplots(figsize=(14, 6))

    cmap = plt.get_cmap("Set1")
    strain_colors = {strain: cmap(i % 9) for i, strain in enumerate(strains)}

    current_x = 0
    trans = ax.get_xaxis_transform()
    y_pos_step = -0.03
    y_pos_model = -0.10

    for model in model_types:
        model_start = current_x

        for step in steps_list:
            step_start = current_x

            for strain in strains:
                subset = grouped[
                    (grouped['ModelType'] == model) &
                    (grouped['steps'] == step) &
                    (grouped['Strains'] == strain)
                ]

                value = subset[f'{col} (%)'].values[0] if not subset.empty else 0

                shade_factor = 0.85 + 0.15 * (
                    steps_list.index(step) / (n_steps - 1 if n_steps > 1 else 1)
                )
                base_color = strain_colors[strain]
                color = tuple([c * shade_factor for c in base_color[:3]] + [1])

                ax.bar(current_x, value, width=bar_width, color=color)
                current_x += bar_width

            # space between strain clusters
            current_x += step_gap

            # label step
            step_center = step_start + (n_strains * bar_width) / 2
            ax.text(step_center, y_pos_step, f"{step}", ha='center', va='top', transform=trans, fontsize=8, alpha=0.8)

        # space between models
        current_x += model_gap

        model_center = model_start + (
            (len(steps_list) * (n_strains * bar_width + step_gap)) / 2
        )
        # Use rotation and adjusted vertical alignment for long scenario names
        ax.text(model_center, y_pos_model, model, ha='center', va='top', transform=trans, fontsize=9, fontweight='bold', rotation=25)

    if use_log:
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 110) 
        ax.set_ylabel("Mean fraction of invalid predictions (%) (log scale)")
    else:
        ax.set_ylim(0, 100)
    ax.set_xticks([])

    for strain in strains:
        ax.bar(0, 0, color=strain_colors[strain], label=strain)

    ax.legend(title="Strains", bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(plot_dir / f"nan_fractions_nested_clusters{suffix}.png", dpi=150)
    plt.close()


def plot_error_per_subset(df, plot_dir, metric="RMSE_pCA_Final"):
    
    model_types = get_model_types(df)
    color_map = {mtype: get_model_color(mtype) for mtype in model_types}
    
    df_stats = (
        df
        .groupby(['ModelType', 'Steps', 'Strains', 'Subset'])[metric]
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': 'mean_rmse', 'std': 'std_rmse'})
    )
    
    step_values = sorted(df_stats['Steps'].unique())
    strain_values = sorted(df_stats['Strains'].unique())
    model_types = get_model_types(df_stats)
    
    n_steps = len(step_values)
    
   
    fig, axes = plt.subplots(n_steps, 1, figsize=(14, 4 * n_steps), sharex=True)
    
    if n_steps == 1:
        axes = [axes]
    
    for ax, step in zip(axes, step_values):
        data_step = df_stats[df_stats['Steps'] == step]
        
        x_positions = []
        x_labels = []
        current_x = 0
        gap = 2
        
        strain_offsets = {}
       
        for strain in strain_values:
            data_strain = data_step[data_step['Strains'] == strain]
            subsets = sorted(data_strain['Subset'].unique())
            
            positions = np.arange(current_x, current_x + len(subsets))
            strain_offsets[strain] = (positions, subsets)
            
            x_positions.extend(positions)
            x_labels.extend(subsets)
            
            current_x += len(subsets) + gap
        
   
        for model in model_types:
            for strain in strain_values:
                positions, subsets = strain_offsets[strain]
                
                data_model = (
                    data_step[
                        (data_step['ModelType'] == model) &
                        (data_step['Strains'] == strain)
                    ]
                    .set_index('Subset')
                    .reindex(subsets)
                    .reset_index()
                )
                
                ax.errorbar(
                    positions,
                    data_model['mean_rmse'],
                    yerr=data_model['std_rmse'],
                    fmt='o-',
                    color=color_map.get(model, 'black'),
                    capsize=3,
                    alpha=0.85,
                    label=model if strain == strain_values[0] else None
                )
        
        for strain in strain_values[:-1]:
            last_pos = strain_offsets[strain][0][-1]
            ax.axvline(last_pos + 1, linestyle='--', alpha=0.2)
        
 
        y_max = ax.get_ylim()[1]
        for strain in strain_values:
            positions, _ = strain_offsets[strain]
            center = positions.mean()
            ax.text(center, y_max, f"{strain}",
                    ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f"{step} timepoints")
        ax.grid(True, alpha=0.3)
    

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(x_labels, rotation=45)

    for i in range(len(axes)):
        axes[i].set_ylabel("RMSE (p-CA final)")
    

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Model Type", loc='upper right')
    
    plt.tight_layout()
    plt.savefig(plot_dir / f"error_per_subset_{metric}.png", dpi=150, bbox_inches='tight')
    plt.close()



def plot_pca_scatter_per_seed_grid(df, plot_dir):

    print("Generating per-seed scatter grids...")
    subsets = sorted(df["Subset"].unique())
    model_types = get_model_types(df)
    
    strains_list = sorted(df["Strains"].unique())
    steps_list = sorted(df["steps"].unique())

    for subset in subsets:
        s_df = df[df["Subset"] == subset]
        seeds = sorted(s_df["Seed"].unique())
        
        for seed in seeds:
            ss_df = s_df[s_df["Seed"] == seed]
            seed_plot_dir = plot_dir / subset / seed
            seed_plot_dir.mkdir(parents=True, exist_ok=True)
            
            for mtype in model_types:
                ssm_df = ss_df[ss_df["ModelType"] == mtype]
                if ssm_df.empty: continue
                
                fig, axes = plt.subplots(len(steps_list), len(strains_list),
                                         figsize=(5 * len(strains_list), 4 * len(steps_list)),
                                         squeeze=False, sharex=False, sharey=True)
                
                for r_step, step in enumerate(steps_list):
                    for c_strain, strain in enumerate(strains_list):
                        ax = axes[r_step, c_strain]
                        run_row = ssm_df[(ssm_df["steps"] == step) & (ssm_df["Strains"] == strain)]
                        if run_row.empty:
                            ax.axis('off')
                            continue
                        row = run_row.iloc[0]
                        if row["Status"] != "success":
                            ax.text(0.5, 0.5, "Failed", ha='center')
                            continue

                        t_gt_raw = row.get("GroundTruths")
                        t_pr = row.get("final_pCA_preds")

                        if t_gt_raw is None or isinstance(t_gt_raw, float) or \
                           t_pr is None or isinstance(t_pr, float):
                            ax.text(0.5, 0.5, "No Traj Data", ha='center')
                            continue

                        t_gt = [float(np.array(x)[-1, PCA_INDEX]) for x in t_gt_raw]

                        tr_gt_raw = row.get("Train_GroundTruths")
                        tr_pr = row.get("Train_final_pCA_preds")
                        if (tr_pr is None or isinstance(tr_pr, float)) and "Train_Predictions" in row:
                            tr_preds = row.get("Train_Predictions")
                            if hasattr(tr_preds, "__iter__") and not isinstance(tr_preds, float):
                                tr_pr = [float(np.array(x)[-1, PCA_INDEX]) for x in tr_preds]
                            else:
                                tr_pr = None

                        tr_gt = [float(np.array(x)[-1, PCA_INDEX]) for x in tr_gt_raw] if (tr_gt_raw is not None and not isinstance(tr_gt_raw, float)) else None

                        if tr_gt and tr_pr:
                            ax.scatter(tr_gt, tr_pr, alpha=0.4, marker='x', color='orange', label='Train')
                        ax.scatter(t_gt, t_pr, alpha=0.6, marker='o', color='blue', label='Test')

                        lims = [min(min(t_gt + (tr_gt or [0])), min(t_pr or [0])),
                                max(max(t_gt + (tr_gt or [1])), max(t_pr or [1]))]
                        ax.plot(lims, lims, 'k--', alpha=0.5)

                  
                        if c_strain == 0:
                            ax.set_ylabel("Concentration (M)", fontsize=9)
                  
                        if r_step == len(steps_list) - 1:
                            ax.set_xlabel(f"{strain} strains", fontsize=10)
                
                        if c_strain == len(strains_list) - 1:
                            ax.annotate(f"{step} timepoints", xy=(1.03, 0.5),
                                        xycoords='axes fraction', fontsize=9,
                                        va='center', ha='left', rotation=270)

                plt.tight_layout()
                plt.savefig(seed_plot_dir / f"pca_scatter_grid_{mtype}.png", dpi=120)
                plt.close()



def plot_seed_time_series_grids(df, plot_dir):

    print("Generating per-seed time-series grids (8 metabolites)...")
    subsets = sorted(df["Subset"].unique())
    model_types = get_model_types(df)
    
    strains_list = sorted(df["Strains"].unique())
    steps_list = sorted(df["steps"].unique())

    for subset in subsets:
        s_df = df[df["Subset"] == subset]
        seeds = sorted(s_df["Seed"].unique())
        
        for seed in seeds:
            ss_df = s_df[s_df["Seed"] == seed]
            seed_plot_dir = plot_dir / subset / seed
            seed_plot_dir.mkdir(parents=True, exist_ok=True)
            
            for mtype in model_types:
                ssm_df = ss_df[ss_df["ModelType"] == mtype]
                if ssm_df.empty: continue
                
           
                success_rows = ssm_df[ssm_df["Status"] == "success"]
                if success_rows.empty: continue
                
                first_preds = np.array(success_rows.iloc[0]["Predictions"])
                n_metabs = first_preds.shape[-1]
                
                for m_idx in range(n_metabs):
                    # Determine metabolite name
                    if m_idx == n_metabs - 1:
                        m_name = "pcoumaric_acid"
                    elif m_idx < len(METABOLITE_NAMES):
                        m_name = METABOLITE_NAMES[m_idx]
                    else:
                        m_name = f"Metab_{m_idx}"

                    fig, axes = plt.subplots(len(steps_list), len(strains_list),
                                             figsize=(5 * len(strains_list), 4 * len(steps_list)),
                                             squeeze=False, sharex=False, sharey=True)
          
                    for r_step, step in enumerate(steps_list):
                        for c_strain, strain in enumerate(strains_list):
                            ax = axes[r_step, c_strain]
                            run_row = ssm_df[(ssm_df["steps"] == step) & (ssm_df["Strains"] == strain)]
                            if run_row.empty:
                                ax.axis('off')
                                continue

                            row = run_row.iloc[0]
                            if row["Status"] != "success":
                                ax.text(0.5, 0.5, "Failed", ha='center')
                                continue

                            preds = np.array(row["Predictions"])  # (N, T, M)
                            truths = np.array(row["GroundTruths"])
                            ts_indices = row.get("ts_indices", [])

                            if truths.ndim == 0 or truths.shape[0] == 0 or preds.ndim == 0 or preds.shape[0] == 0:
                                ax.text(0.5, 0.5, "No Traj Data", ha='center')
                                continue

                            n_to_plot = min(10, truths.shape[0])
                            colors = plt.cm.tab10(np.linspace(0, 1, 10))

                            for i in range(n_to_plot):
                                color = colors[i % 10]
                                ax.plot(TS, truths[i, :, m_idx], 'o', color=color, alpha=0.2, markersize=3)
                                ax.plot(TS, preds[i, :, m_idx], '-', color=color, alpha=0.8, linewidth=1.5)
                                if isinstance(ts_indices, (list, np.ndarray, pd.Series)) and len(ts_indices) > 0:
                                    ax.plot(TS[ts_indices], truths[i, ts_indices, m_idx], 'x', color='red', markersize=5, alpha=0.9)

                            if c_strain == 0:
                                ax.set_ylabel("Concentration (M)", fontsize=9)
                   
                            if r_step == len(steps_list) - 1:
                                ax.set_xlabel(f"{strain} strains", fontsize=10)

                            if c_strain == len(strains_list) - 1:
                                ax.annotate(f"{step} timepoints", xy=(1.03, 0.5),
                                            xycoords='axes fraction', fontsize=9,
                                            va='center', ha='left', rotation=270)

                    plt.tight_layout()
                    plt.savefig(seed_plot_dir / f"time_series_grid_{mtype}_{m_name}.png", dpi=120)
                    plt.close()



def plot_metabolite_nrmse_comparison(df, plot_dir):
    print("Generating metabolite NRMSE summary comparison...")
    metab_data = []
    
    for _, row in df.iterrows():
        nrmse_list = row.get("NRMSE_per_species")
        
        if isinstance(nrmse_list, list) and len(nrmse_list) == len(METABOLITE_NAMES):
            for m_idx, name in enumerate(METABOLITE_NAMES):
                metab_data.append({
                    "ModelType": row["ModelType"], 
                    "Strains": row["Strains"], 
                    "Steps": row["steps"], 
                    "Metabolite": name, 
                    "NRMSE (%)": nrmse_list[m_idx]
                })

        elif f"NRMSE_per_species_{METABOLITE_NAMES[0]}" in df.columns:
            for name in METABOLITE_NAMES:
                val = row.get(f"NRMSE_per_species_{name}")
                if pd.notna(val):
                    metab_data.append({
                        "ModelType": row["ModelType"], 
                        "Strains": row["Strains"], 
                        "Steps": row["steps"], 
                        "Metabolite": name, 
                        "NRMSE (%)": val
                    })

   
        elif "Predictions" in row and "GroundTruths" in row:
            preds = np.array(row.get("Predictions", []))
            gt = np.array(row.get("GroundTruths", []))
            if preds.ndim == 3 and gt.shape == preds.shape:
           
                for m_idx, name in enumerate(METABOLITE_NAMES):
                    y_true = gt[:, 1:, m_idx].flatten()
                    y_pred = preds[:, 1:, m_idx].flatten()
                    mask = np.isfinite(y_true) & np.isfinite(y_pred)
                    if np.sum(mask) == 0: continue
                    rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))
                  
                    mean_val = np.mean(y_true[mask])
                    nrmse = (rmse / mean_val * 100) if mean_val != 0 else np.nan
                    metab_data.append({"ModelType": row["ModelType"], "Strains": row["Strains"], "Steps": row["steps"], "Metabolite": name, "NRMSE (%)": nrmse})
            
    m_df = pd.DataFrame(metab_data)
    if m_df.empty: return
    g = sns.FacetGrid(m_df, row="Steps", col="Strains", height=4, aspect=1.5, sharey=True)
    g.map_dataframe(sns.barplot, x="Metabolite", y="NRMSE (%)", hue="ModelType", hue_order=get_model_types(m_df), palette="muted")
    for ax in g.axes.flatten():
        for label in ax.get_xticklabels(): label.set_rotation(45)
    g.add_legend()
    plt.savefig(plot_dir / "metabolite_nrmse_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_hybrid_metabolite_nrmse_condensed(df, plot_dir):
    print("Generating condensed Hybrid metabolite NRMSE comparison...")
    
   
    hybrid_df = df[df["ModelType"].str.contains("Hybrid", case=False, na=False)]
    if hybrid_df.empty:
        print("No Hybrid model data found for condensed metabolite plot.")
        return

    metab_data = []
    for _, row in hybrid_df.iterrows():
        nrmse_list = row.get("NRMSE_per_species")
        
       
        if isinstance(nrmse_list, list) and len(nrmse_list) == len(METABOLITE_NAMES):
            for m_idx, name in enumerate(METABOLITE_NAMES):
                metab_data.append({
                    "Strains": row["Strains"], 
                    "Steps": row["steps"], 
                    "Metabolite": name, 
                    "NRMSE (%)": nrmse_list[m_idx]
                })
     
        elif f"NRMSE_per_species_{METABOLITE_NAMES[0]}" in df.columns:
            for name in METABOLITE_NAMES:
                val = row.get(f"NRMSE_per_species_{name}")
                if pd.notna(val):
                    metab_data.append({
                        "Strains": row["Strains"], 
                        "Steps": row["steps"], 
                        "Metabolite": name, 
                        "NRMSE (%)": val
                    })
  
        elif "Predictions" in row and "GroundTruths" in row:
            preds = np.array(row.get("Predictions", []))
            gt = np.array(row.get("GroundTruths", []))
            if preds.ndim == 3 and gt.shape == preds.shape:
                for m_idx, name in enumerate(METABOLITE_NAMES):
                    y_true = gt[:, 1:, m_idx].flatten()
                    y_pred = preds[:, 1:, m_idx].flatten()
                    mask = np.isfinite(y_true) & np.isfinite(y_pred)
                    if np.sum(mask) == 0: continue
                    rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))
                    mean_val = np.mean(y_true[mask])
                    nrmse = (rmse / mean_val * 100) if mean_val != 0 else np.nan
                    metab_data.append({"Strains": row["Strains"], "Steps": row["steps"], "Metabolite": name, "NRMSE (%)": nrmse})

    m_df = pd.DataFrame(metab_data)
    if m_df.empty: return
    
  
    m_df = m_df.sort_values("Strains")
    
    steps_list = sorted(m_df["Steps"].unique())
    n_steps = len(steps_list)
    
    g = sns.FacetGrid(m_df, col="Steps", col_wrap=1, height=5, aspect=2.0, sharey=True)
    g.map_dataframe(sns.barplot, x="Metabolite", y="NRMSE (%)", hue="Strains", palette="viridis", order=METABOLITE_NAMES)
    
    for ax in g.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        ax.set_title(f"Steps: {ax.get_title().split('=')[-1].strip()}", fontweight='bold')
    
    g.add_legend(title="Strain Size")
    plt.savefig(plot_dir / "hybrid_metabolite_nrmse_condensed.png", dpi=150, bbox_inches='tight')
    plt.close()



def plot_rmse_vs_strains_steps_models(df, plot_dir, metric="RMSE_pCA_Final", exclude_models=None, filename_suffix=""):

    print("Generating RMSE vs Strains plot...")

    success_df = df[df["Status"] == "success"]
    if success_df.empty:
        return

    if metric not in success_df.columns:
        print("{metric} column missing.")
        return

    if exclude_models is not None:
        success_df = success_df[~success_df["ModelType"].isin(exclude_models)]
        if success_df.empty:
            print(f"No data left after excluding models: {exclude_models}")
            return

    stats = (
        success_df
        .groupby(["ModelType", "steps", "Strains"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    model_types = get_model_types(stats)
    steps_list = sorted(stats["steps"].unique())

    plt.figure(figsize=(9,6))

    # Palettes per model
    linestyles_list = ["-", "--", "-.", ":"]
    linestyles = {}
    palettes = {}
    
    for i, mtype in enumerate(model_types):
        base_color = get_model_color(mtype)

        palettes[mtype] = sns.light_palette(base_color, n_colors=len(steps_list) + 2)[2:]
        linestyles[mtype] = linestyles_list[i % len(linestyles_list)]

    for model in model_types:

        palette = palettes.get(model, sns.color_palette("viridis", len(steps_list)))

        for step, color in zip(steps_list, palette):

            sub = stats[
                (stats["ModelType"] == model) &
                (stats["steps"] == step)
            ].sort_values("Strains")

            if sub.empty:
                continue

            plt.errorbar(
                sub["Strains"],
                sub["mean"],
                yerr=sub["std"],
                fmt="o",
                linestyle=linestyles.get(model, "-"),
                color=color,
                capsize=4,
                linewidth=1.6,
                markersize=6,
                label=f"{model} – {step} timepoints"
            )

    plt.xlabel("Strains")
    plt.ylabel(f"RMSE (p-CA final)")

    plt.grid(alpha=0.3)

    plt.legend(ncol=2, fontsize=9, loc='upper right')
    plt.tight_layout()

    plt.savefig(plot_dir / f"rmse_vs_strains_{metric}_palette{filename_suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_rmse_per_strain_histograms(df, plot_dir):
   
    print("Generating per-model RMSE strain grid plots...")
    success_df = df[df["Status"] == "success"].copy()
    if success_df.empty:
        print("No success data to plot.")
        return

    # Collect all error data first to allow global grouping
    all_error_data = []
    required_data = ["ModelType", "steps", "Strains", "GroundTruths"]
    for col in required_data:
        if col not in success_df.columns:
            print(f"Skipping per-strain plot: column {col} missing.")
            return

    for _, row in success_df.iterrows():
        mtype = row["ModelType"]
        step = row["steps"]
        strain_count = row["Strains"]
        
        # Test data
        t_gt_raw = row.get("GroundTruths")
        t_pr = row.get("final_pCA_preds")
        
        if t_gt_raw is not None and hasattr(t_gt_raw, "__iter__") and not isinstance(t_gt_raw, float) and \
           t_pr is not None and hasattr(t_pr, "__iter__") and not isinstance(t_pr, float) and \
           len(t_gt_raw) > 0 and len(t_pr) == len(t_gt_raw):
            t_gt = [float(np.array(x)[-1, PCA_INDEX]) for x in t_gt_raw]
            t_errs = np.abs(np.array(t_gt) - np.array(t_pr))
            for gt, err in zip(t_gt, t_errs):
                all_error_data.append({
                    "ModelType": mtype,
                    "steps": step,
                    "Strains": strain_count,
                    "gt_pCA": gt,
                    "Error": err
                })
        
      
        tr_gt_raw = row.get("Train_GroundTruths")
        tr_pr = row.get("Train_final_pCA_preds")
        if (tr_pr is None or isinstance(tr_pr, float)) and "Train_Predictions" in row:
            tr_preds = row.get("Train_Predictions")
            if hasattr(tr_preds, "__iter__") and not isinstance(tr_preds, float):
                tr_pr = [float(np.array(x)[-1, PCA_INDEX]) for x in tr_preds]

        if tr_gt_raw is not None and hasattr(tr_gt_raw, "__iter__") and not isinstance(tr_gt_raw, float) and \
           tr_pr is not None and hasattr(tr_pr, "__iter__") and not isinstance(tr_pr, float) and \
           len(tr_gt_raw) > 0 and len(tr_pr) == len(tr_gt_raw):
            tr_gt = [float(np.array(x)[-1, PCA_INDEX]) for x in tr_gt_raw]
            tr_errs = np.abs(np.array(tr_gt) - np.array(tr_pr))
            for gt, err in zip(tr_gt, tr_errs):
                all_error_data.append({
                    "ModelType": mtype,
                    "steps": step,
                    "Strains": strain_count,
                    "gt_pCA": gt,
                    "Error": err,
                    "IsTrain": True
                })

    if not all_error_data:
        print("No error data found for per-strain plot.")
        return

    error_df = pd.DataFrame(all_error_data)
    # Filter out infinite/NaN errors so valid predictions can still form bars
    error_df = error_df[np.isfinite(error_df["Error"])]
    error_df["gt_pCA_rounded"] = error_df["gt_pCA"].round(4)
    
    print("Calculating global Y-axis limits for consistency...")
    global_stats = error_df.groupby(["ModelType", "steps", "Strains", "gt_pCA_rounded"])["Error"].agg(["mean", "std"]).reset_index()
    y_values = (global_stats["mean"] + global_stats["std"].fillna(0)).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    finite_y = y_values[np.isfinite(y_values)]
    if finite_y.size == 0:
        print("No finite RMSE statistics found for per-strain plot.")
        return
    global_max_y = finite_y.max()
    y_limit = global_max_y * 1.05 if np.isfinite(global_max_y) else 1.0

    model_types = get_model_types(error_df)
    steps_list = sorted(error_df["steps"].unique())
    strains_list = sorted(error_df["Strains"].unique())

    for mtype in model_types:
        m_df = error_df[error_df["ModelType"] == mtype]
        if m_df.empty: continue
        
        fig, axes = plt.subplots(len(steps_list), len(strains_list), 
                                 figsize=(6 * len(strains_list), 4 * len(steps_list)), 
                                 squeeze=False, sharey=True)


        for r, step in enumerate(steps_list):
            for c, strain_count in enumerate(strains_list):
                ax = axes[r, c]
                cell_df = m_df[(m_df["steps"] == step) & (m_df["Strains"] == strain_count)]
                
                if cell_df.empty:
                    ax.axis('off')
                    continue
                
                stats = cell_df.groupby("gt_pCA_rounded")["Error"].agg(["mean", "std"]).reset_index()
                stats = stats.sort_values("gt_pCA_rounded")
                
                x = np.arange(len(stats))
                color = get_model_color(mtype)
                
                ax.bar(x, stats["mean"], yerr=stats["std"], color=color, 
                       alpha=0.8, capsize=2, edgecolor='black', linewidth=0.2)
                
                ax.set_ylim(0, y_limit)
                
                if c == 0:
                    ax.set_ylabel("RMSE (p-CA final)", fontsize=10)
                
                if r == len(steps_list) - 1:
                    ax.set_xlabel(f"{strain_count} strains", fontsize=10)
                
                if c == len(strains_list) - 1:
                    ax.annotate(f"{step} timepoints", xy=(1.03, 0.5),
                                xycoords='axes fraction', fontsize=10,
                                va='center', ha='left', rotation=270)
                
                ax.set_xticks([])
                
                ax.grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        save_path = plot_dir / f"rmse_per_strain_grid_{mtype}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved per-model grid: {save_path}")


def plot_subset_loss_histories(df, plot_dir):
 
    print(f"Plotting Loss Histories Per Subset")
    
    groups = df.groupby(["Experiment", "Subset"])
    
    for (exp_name, subset_id), subset_df in groups:
        # Create directory for subset
        subset_plot_dir = plot_dir / subset_id
        subset_plot_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        seeds = sorted(subset_df["Seed"].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
        
        has_data = False
        for i, seed in enumerate(seeds):
            seed_row = subset_df[subset_df["Seed"] == seed].iloc[0]
            seed_path = Path(seed_row["Path"]).parent
            
            loss_path = seed_path / "loss_history.csv"
            val_loss_path = seed_path / "val_loss_history.npy"
            
            # Try loading CSV first
            train_loss = None
            if loss_path.exists():
                try:
                    loss_df = pd.read_csv(loss_path)
                    if "loss" in loss_df.columns:
                        train_loss = loss_df["loss"].values
                except Exception as e:
                    print(f"      Warning: Could not read {loss_path}: {e}")
            
            # Fallback to .npy for training loss
            if train_loss is None:
                train_loss_npy = seed_path / "loss_history.npy"
                if train_loss_npy.exists():
                    try:
                        train_loss = np.load(train_loss_npy)
                    except Exception as e:
                        print(f"      Warning: Could not read {train_loss_npy}: {e}")
            
            # Load validation loss
            val_loss = None
            if val_loss_path.exists():
                try:
                    val_loss = np.load(val_loss_path)
                except Exception as e:
                    print(f"      Warning: Could not read {val_loss_path}: {e}")
            
            if train_loss is not None:
                has_data = True
                steps = np.arange(len(train_loss))
                ax.plot(steps, train_loss, color=colors[i], label=f"Seed {seed} Train", alpha=0.8, linewidth=1.5)
                
                if val_loss is not None:
                    # Determine validation steps. Usually val is sampled every print_freq steps.
                    if len(val_loss) > 1:
                        val_steps = np.linspace(0, steps[-1], len(val_loss))
                        ax.plot(val_steps, val_loss, color=colors[i], label=f"Seed {seed} Val", 
                                linestyle="--", alpha=0.6, linewidth=1.2)
                    elif len(val_loss) == 1:
                        ax.plot([0], val_loss, color=colors[i], marker="o", markersize=4,
                                linestyle="None", alpha=0.6)
        
        if not has_data:
            plt.close(fig)
            continue
            
        ax.set_yscale("log")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss (Log Scale)")
        ax.set_title(f"Loss History: {exp_name} ({subset_id})")
        
        ax.autoscale(True)
        
       
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', frameon=True)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        
        plt.tight_layout()
        save_path = subset_plot_dir / f"losses_{exp_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✓ Saved: {save_path}")




def main():
    
    use_csv = False

    base_dir = "Experiments/Question_1_new"
    pkl_path = os.path.join(base_dir, "experiment_results_df.pkl")
    csv_path = os.path.join(base_dir, "experiment_results_summary.csv")
    
    df = None
    if use_csv and os.path.exists(csv_path):
        print(f"Loading summary CSV from {csv_path}")
        df = pd.read_csv(csv_path)
    elif os.path.exists(pkl_path):
        print(f"Loading full Pickle from {pkl_path}")
        df = pd.read_pickle(pkl_path)
    elif os.path.exists(csv_path):
        print(f"Pickle not found. Loading summary CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
     
        from load_experiment_results import load_all_results
        print(f"\nStep 1: Loading results from {base_dir} (this may take a while)...")
        df = load_all_results(base_dir=base_dir, output_pickle="experiment_results_df.pkl", output_csv="experiment_results_summary.csv")
        if df is None:
            print(f"Error: Neither {pkl_path} nor {csv_path} found, and scan failed.")
            return

    df = prepare_dataframe(df)
    plot_dir = setup_plot_dir(base_dir)
    
    # Check if we have the data needed for detailed plots
    has_trajectories = "Predictions" in df.columns and "GroundTruths" in df.columns
    has_paths = "Path" in df.columns

    if has_paths:
        plot_subset_loss_histories(df, plot_dir)
    else:
        print("Skipping loss histories (Path column missing).")

    plot_unified_heatmaps(df, plot_dir)
    plot_unified_heatmaps(df, plot_dir, metric="RMSE_pCA_Final")
    plot_unified_heatmaps(df, plot_dir, metric="RMSE_pCA_All")
    plot_unified_heatmaps(df, plot_dir, metric="NRMSE_All_Species")
    plot_unified_heatmaps(df, plot_dir, metric="Train_RMSE_pCA_Final")
    plot_unified_heatmaps(df, plot_dir, metric="Train_NRMSE_All_Species")
    plot_rmse_vs_strains_steps_models(df, plot_dir)
    plot_rmse_vs_strains_steps_models(df, plot_dir, metric="NRMSE_All_Species")
    plot_rmse_vs_strains_steps_models(df, plot_dir, metric="RMSE_pCA_Final", exclude_models=["NODE"], filename_suffix="_no_NODE")

    plot_absolute_rmse_box_plots(df, plot_dir)
    plot_train_test_gap(df, plot_dir)
    plot_error_per_subset(df, plot_dir, metric='RMSE_pCA_Final')
    plot_error_per_subset(df, plot_dir, metric='NRMSE_All_Species')
    plot_success_rates_bar(df, plot_dir)
    plot_nan_fractions_bar(df, plot_dir, use_log=False)
    plot_nan_fractions_bar(df, plot_dir, use_log=True)
    plot_metabolite_nrmse_comparison(df, plot_dir)
    plot_hybrid_metabolite_nrmse_condensed(df, plot_dir)
    plot_rmse_per_strain_histograms(df, plot_dir)


    if has_trajectories:
        plot_pca_scatter_per_seed_grid(df, plot_dir)
        plot_seed_time_series_grids(df, plot_dir)


    print(f"\nAll plots saved to {plot_dir}")

if __name__ == "__main__":
    main()
