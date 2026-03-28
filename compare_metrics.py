import os
import re
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.expanduser("~/realistic_log_project/results")
IMBALANCED_FILE = os.path.join(RESULTS_DIR, "results_Logs_plus_Telemetry.txt")
BALANCED_FILE   = os.path.join(RESULTS_DIR, "results_Balanced_Logs_plus_Telemetry.txt")

metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "AUC-PRC", "FPR"]

def parse_results(filepath):
    if not os.path.exists(filepath):
        return None
    
    data = {}
    with open(filepath, "r") as f:
        content = f.read()
        for m in metrics_list:
            # Look for lines like "Accuracy  : 0.9995"
            match = re.search(rf"{m}\s*:\s*([\d\.]+)", content)
            if match:
                data[m] = float(match.group(1))
            else:
                data[m] = 0.0 # Default if not found (e.g. old file missing FPR)
    return data

def main():
    print("="*60)
    print(" 📊 AIOps Evaluation — 97:3 vs 50:50 Comparison Chart")
    print("="*60)

    imbalanced_data = parse_results(IMBALANCED_FILE)
    balanced_data   = parse_results(BALANCED_FILE)

    if not imbalanced_data:
        print(f"⚠️ Could not find {IMBALANCED_FILE}. Please make sure the results file exists.")
        return
    if not balanced_data:
        print(f"⚠️ Could not find {BALANCED_FILE}. Please make sure the balanced results file exists.")
        return

    # Extract values
    vals_imb = [imbalanced_data[m] for m in metrics_list]
    vals_bal = [balanced_data[m] for m in metrics_list]

    # CLI Table
    print(f"\n{'Metric':<15} | {'97:3 (Imbalanced)':<18} | {'50:50 (Balanced)'}")
    print("-" * 55)
    for m, i_val, b_val in zip(metrics_list, vals_imb, vals_bal):
        diff = round(b_val - i_val, 4)
        print(f"{m:<15} | {i_val:<18.4f} | {b_val:.4f}  ({diff:+.4f})")

    # Plot
    x = np.arange(len(metrics_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, vals_imb, width, label='97:3 (Imbalanced)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, vals_bal, width, label='50:50 (Balanced)', color='#ff7f0e')

    ax.set_ylabel('Score')
    ax.set_title('AIOps Evaluation: 97:3 vs 50:50 Dataset Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list)
    ax.legend(loc='lower right')

    ax.bar_label(rects1, fmt='%.3f', padding=3, fontsize=9)
    ax.bar_label(rects2, fmt='%.3f', padding=3, fontsize=9)

    fig.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "Evaluation_Comparison_Chart.png")
    
    # Ensure results directory exists before saving (just in case)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(chart_path, dpi=300)
    
    print(f"\n✅ Created comparison chart: {chart_path}")
    print("   Open this PNG file to view the side-by-side performance.")

if __name__ == "__main__":
    main()
