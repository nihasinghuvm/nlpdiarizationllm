import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from diarizationlm import compute_metrics_on_json_dict

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def compute_metrics_from_folder(json_folder):
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    results = []

    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                jd = json.load(f)

            fname = os.path.basename(file_path)
            sbc = fname.replace(".json", "")

            metrics = compute_metrics_on_json_dict(jd)

            results.append({
                "filename": fname,
                "SBC Number": sbc,
                "WER": metrics["WER"],
                "WDER": metrics["WDER"],
                "cpWER": metrics["cpWER"]
            })
            print(f"Processed {fname}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    df = pd.DataFrame(results).sort_values("SBC Number")
    return df


def compare_five_conditions(baseline_folder, dlm_500_folder, dlm_50_folder, dlm_20_folder, dlm_adapt_folder):

    print("\n=== BASELINE ===")
    df_base = compute_metrics_from_folder(baseline_folder).rename(columns={
        "WER": "WER_baseline",
        "WDER": "WDER_baseline",
        "cpWER": "cpWER_baseline"
    }).drop(columns=["filename"])

    print("\n=== DLM-500 ===")
    df_500 = compute_metrics_from_folder(dlm_500_folder).rename(columns={
        "WER": "WER_dlm_500",
        "WDER": "WDER_dlm_500",
        "cpWER": "cpWER_dlm_500"
    }).drop(columns=["filename"])

    print("\n=== DLM-50 ===")
    df_50 = compute_metrics_from_folder(dlm_50_folder).rename(columns={
        "WER": "WER_dlm_50",
        "WDER": "WDER_dlm_50",
        "cpWER": "cpWER_dlm_50"
    }).drop(columns=["filename"])

    print("\n=== DLM-20 ===")
    df_20 = compute_metrics_from_folder(dlm_20_folder).rename(columns={
        "WER": "WER_dlm_20",
        "WDER": "WDER_dlm_20",
        "cpWER": "cpWER_dlm_20"
    }).drop(columns=["filename"])

    print("\n=== DLM-ADAPTIVE ===")
    df_adapt = compute_metrics_from_folder(dlm_adapt_folder).rename(columns={
        "WER": "WER_dlm_adapt",
        "WDER": "WDER_dlm_adapt",
        "cpWER": "cpWER_dlm_adapt"
    }).drop(columns=["filename"])

    df = (
        df_base
        .merge(df_500, on="SBC Number")
        .merge(df_50, on="SBC Number")
        .merge(df_20, on="SBC Number")
        .merge(df_adapt, on="SBC Number")
    )

    df.to_csv("comparison_results_dlm_adaptive.csv", index=False)

    return df


def box_plots(df, output_dir="analysis_plots_box"):
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["WDER", "cpWER"]
    conditions = ["baseline", "dlm_500", "dlm_50", "dlm_20", "dlm_adapt"]

    for metric in metrics:
        melted = pd.DataFrame({
            "Condition": np.repeat(conditions, len(df)),
            metric: np.concatenate([df[f"{metric}_{c}"].values for c in conditions])
        })

        plt.figure(figsize=(12, 7))
        sns.boxplot(data=melted, x="Condition", y=metric, palette="husl")
        sns.swarmplot(data=melted, x="Condition", y=metric, color="black", alpha=0.6)

        plt.title(f"{metric} — Box Plot", fontsize=20, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"box_{metric}.png"), dpi=300)
        plt.close()

def p_to_stars(p):
    if p < 0.0001: return "****"
    elif p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "ns"


def add_improvements(df):
    for metric in ["WER", "WDER", "cpWER"]:
        for cond in ["dlm_500", "dlm_50", "dlm_20", "dlm_adapt"]:
            df[f"{metric}_improv_{cond}"] = (
                (df[f"{metric}_baseline"] - df[f"{metric}_{cond}"]) /
                df[f"{metric}_baseline"] * 100
            )

def perform_wilcoxon_tests(df):
    metrics = ["WDER", "cpWER"]
    conditions = ["dlm_500", "dlm_50", "dlm_20", "dlm_adapt"]

    rows = []

    for metric in metrics:
        base = df[f"{metric}_baseline"]

        for cond in conditions:
            model = df[f"{metric}_{cond}"]
            n = min(len(base), len(model))

            stat, p = stats.wilcoxon(base.iloc[:n], model.iloc[:n])

            rows.append({
                "metric": metric,
                "condition": cond,
                "p_value": float(p),
                "significance": p_to_stars(p)
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv("wilcoxon_results_dlm_adaptive.csv", index=False)
    print(" Saved wilcoxon_results_dlm_adaptive.csv")

    return out_df

def bar_chart_with_sem(df, output_dir="analysis_plots_adapt"):
    os.makedirs(output_dir, exist_ok=True)

    conditions = ["baseline", "dlm_500", "dlm_50", "dlm_20", "dlm_adapt"]
    colors = ["#87AE73", "#87CEEB", "#F4C2C2", "#9370DB", "orange"]
    metrics = ["WER", "WDER", "cpWER"]

    plt.rcParams.update({"font.size": 16})

    for metric in metrics:
        means = [df[f"{metric}_{c}"].mean() for c in conditions]
        sems = [df[f"{metric}_{c}"].sem()  for c in conditions]

        plt.figure(figsize=(12, 7))
        bars = plt.bar(conditions, means, yerr=sems, color=colors,
                       capsize=8, edgecolor="black", alpha=0.85)

        # Add mean labels
        for i, bar in enumerate(bars):
            y = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2,
                     y + sems[i] + 0.003,
                     f"{y:.3f}",
                     ha="center", fontsize=14, fontweight="bold")

        plt.title(f"{metric} — Mean ± SEM", fontweight="bold", fontsize=22)
        plt.ylabel(metric, fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"bar_{metric}_sem.png"), dpi=300)
        plt.close()

def print_summary(df):
    print("\n================ SUMMARY ================")

    for metric in ["WER", "WDER", "cpWER"]:
        print(f"\n{metric} Means:")
        print(f"  Baseline: {df[f'{metric}_baseline'].mean():.4f}")
        print(f"  DLM-500 : {df[f'{metric}_dlm_500'].mean():.4f}")
        print(f"  DLM-50  : {df[f'{metric}_dlm_50'].mean():.4f}")
        print(f"  DLM-20  : {df[f'{metric}_dlm_20'].mean():.4f}")
        print(f"  DLM-Adaptive  : {df[f'{metric}_dlm_adapt'].mean():.4f}")

    for metric in ["WER", "WDER", "cpWER"]:
        print(f"\n{metric} Medians:")
        print(f"  Baseline: {df[f'{metric}_baseline'].median():.4f}")
        print(f"  DLM-500 : {df[f'{metric}_dlm_500'].median():.4f}")
        print(f"  DLM-50  : {df[f'{metric}_dlm_50'].median():.4f}")
        print(f"  DLM-20  : {df[f'{metric}_dlm_20'].median():.4f}")
        print(f"  DLM-Adaptive  : {df[f'{metric}_dlm_adapt'].median():.4f}")

    print("\n========================================\n")


def global_metric_plots(df, output_dir="analysis_plots_global_exclude"):
    """
    Creates three global plots:
        1. WER across all SBC numbers
        2. cpWER across all SBC numbers
        3. WDER across all SBC numbers
    For each plot, all techniques are shown.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["WER", "cpWER", "WDER"]
    conditions = ["baseline", "dlm_500", "dlm_50", "dlm_20", "dlm_adapt"]

    colors = {
        "baseline": "#87AE73",
        "dlm_500": "#87CEEB",
        "dlm_50": "#F4C2C2",
        "dlm_20": "#9370DB",
        "dlm_adapt": "orange",
    }

    #df = df[~df["SBC Number"].isin(exclude)]

    x = df["SBC Number"].astype(str)

    for metric in metrics:
        plt.figure(figsize=(16, 8))

        for cond in conditions:
            plt.plot(
                x,
                df[f"{metric}_{cond}"],
                marker="o",
                linewidth=2.5,
                markersize=6,
                label=cond,
                color=colors[cond]
            )

        plt.title(f"{metric} Across SBC Numbers", fontsize=22, fontweight="bold")
        plt.xlabel("SBC Number", fontsize=16, fontweight="bold")
        plt.ylabel(metric, fontsize=16, fontweight="bold")
        plt.xticks(rotation=90)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(title="Technique", fontsize=12)
        plt.tight_layout()

        outfile = os.path.join(output_dir, f"{metric}_global_plot.png")
        plt.savefig(outfile, dpi=300)
        plt.close()

def find_lowest_metrics(df, output_csv="lowest_metrics_per_sbc.csv"):
    """
    For each SBC number, determine which condition
    has the lowest WDER and lowest cpWER.
    """

    conditions = ["baseline", "dlm_500", "dlm_50", "dlm_20", "dlm_adapt"]

    records = []

    for _, row in df.iterrows():
        sbc = row["SBC Number"]

        # Extract values per model
        wder_values = {cond: row[f"WDER_{cond}"] for cond in conditions}
        cpwer_values = {cond: row[f"cpWER_{cond}"] for cond in conditions}

        # Identify the minimums
        best_wder_model = min(wder_values, key=wder_values.get)
        best_cpwer_model = min(cpwer_values, key=cpwer_values.get)

        records.append({
            "SBC Number": sbc,
            "Lowest WDER": wder_values[best_wder_model],
            "Best WDER Model": best_wder_model,
            "Lowest cpWER": cpwer_values[best_cpwer_model],
            "Best cpWER Model": best_cpwer_model,
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    return out_df



if __name__ == "__main__":

    baseline_folder = "json_baseline"
    dlm_500_folder = "json_output_dlm_500"
    dlm_50_folder = "json_output_dlm_50"
    dlm_20_folder = "json_output_dlm_20"
    dlm_adapt_folder = "json_output_dlm_adaptive"

    comp = compare_five_conditions(
        baseline_folder,
        dlm_500_folder,
        dlm_50_folder,
        dlm_20_folder,
        dlm_adapt_folder
    )

    add_improvements(comp)

    bar_chart_with_sem(comp)
    results = perform_wilcoxon_tests(comp)
    box_plots(comp)
    global_metric_plots(comp) 
    lowest_df = find_lowest_metrics(comp)
    print(lowest_df)
 
    print_summary(comp)

    print("\n=== COMPLETE ===")
