import os
import argparse
import numpy as np
import pickle
import pandas as pd

from utils.general_utils import set_seed
from utils.plot_utils import update_risk_dict

from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt


def plot_km_curves(all_data, model_name):
    """ Plot KM curves for all datasets in one figure. """
    # Set up the figure with subplots for each dataset
    n_datasets = len(all_data)
    fig, axes = plt.subplots(nrows=1, ncols=n_datasets, figsize=(6 * n_datasets, 5), sharex=True)

    # If there's only one dataset, axes is not an array
    if n_datasets == 1:
        axes = [axes]
        
    # For each dataset, plot the KM curve
    for ax, (dataset_name, pooled_km_data) in zip(axes, all_data.items()):
        
        times = pooled_km_data["Time"]
        events = np.array([1 if x == 0 else 0 for x in pooled_km_data["Censorship"]]) # CHEK IF IT IS THE CORRECT ONE
        scores = pooled_km_data["Risk scores"]
        
        # Divide patients into high-risk and low-risk groups based on median risk score
        median_score = np.median(scores)
        high_risk = scores > median_score
        low_risk = scores < median_score   

        # Cox Proportional Hazards model to compute HR and p-value
        cph = CoxPHFitter()
        df = pd.DataFrame({
            "time": np.concatenate([times[high_risk], times[low_risk]]),
            "event": np.concatenate([events[high_risk], events[low_risk]]),
            "group": np.concatenate([np.ones(high_risk.sum()), np.zeros(low_risk.sum())]) 
        })
        cph.fit(df, duration_col="time", event_col="event")

        # Obtain hazard ratio and p-value
        hr = cph.hazard_ratios_["group"]
        p_val = cph.summary.loc["group", "p"]

        # KaplanMeier plot
        kmf = KaplanMeierFitter()

        # Plot KM curves for high-risk and low-risk groups
        for group, label_name, colour in zip([high_risk, low_risk], ["High Risk", "Low Risk"], ['red', 'blue']):
            kmf.fit(times[group], events[group], label=label_name)
            kmf.plot(ci_show=True, color=colour, show_censors=True, linewidth=2, ax=ax)

        # Plot settings
        ax.text(
            0.10, 0.10,
            f"HR = {hr:.3f}\np = {p_val:.2e}",
            transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='white'),
            fontsize=10
        )
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Disease-specific survival probability")
        ax.set_xlabel("Time (days)")
        ax.set_title(dataset_name.upper().replace("_", " "), fontsize=15, fontweight='bold')  # No individual subplot title
        ax.legend(loc='upper right', ncol=2, frameon=False,  fontsize=10)

    fig.text(
        -0.01, 0.5,
        model_name.replace("_", " "),
        va='center', rotation='vertical',
        fontsize=15, fontweight='bold'
    )

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(f"results/figures/KM_{model_name}.pdf", format="pdf", bbox_inches="tight")

def get_results_over_all_folds(result_dir):
  """ Store all predicted risk scores and survival data of all test samples over all folds. """
  risk_dict = dict()
  for i in range(5):
    fold_dir = os.path.join(result_dir, f"Fold_{i}/pred_test_risk_scores.pkl")
    with open(fold_dir, 'rb') as f:
      risk_dict_fold = pickle.load(f)

    risk_dict = update_risk_dict(risk_dict, risk_dict_fold)
  
  return risk_dict
  

def main(args):
    set_seed(args.seed)
    all_datasets = ['dss_survival_brca', 'dss_survival_blca', 'dss_survival_luad', 'dss_survival_kirc']
    all_types_km_data = {}

    # Obtain predicted risk scores for all datasets
    for data_type in all_datasets:
        result_dir = os.path.join(args.result_dir, data_type, args.exp_code)

        if not os.path.exists(result_dir):
            print(f"Result directory {result_dir} does not exist!")
            continue

        # Get all test data & results
        pooled_km_data = get_results_over_all_folds(result_dir)
        new_data_type = ('tcga ' + data_type.split('_')[-1]).upper()
        all_types_km_data[new_data_type] = pooled_km_data

    # Plot KM curves for all datasets
    plot_km_curves(all_types_km_data, args.model_name)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Plot KM curves')
    
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--result_dir', default='results',help='results directory')
    parser.add_argument('--exp_code', type=str, default='test', help='experiment code for saving results')
    parser.add_argument('--model_name', type=str, default='test', help='experiment code for saving results')

    args = parser.parse_args()
    main(args)