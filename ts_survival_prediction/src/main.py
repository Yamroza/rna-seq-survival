import argparse
import sys
import os 
import torch 
import numpy as np
import wandb

from utils.general_utils import set_seed, save_json
from utils.train_utils import save_exp_settings
from survival.train import survival_train
from survival.test import survival_test


def k_fold_test(args, device):
    """ K-fold cross-validation - Test only. """
    final_res = {}
    losses = []
    c_indices = []
    c_indices_ipcw = []

    for i in range(args.folds):
        print("Testing fold ", i)
        results = survival_test(args, i, device)
        final_res[f'Fold{i}'] = results

        losses.append(results["loss"])
        c_indices.append(results["c_index"])
        c_indices_ipcw.append(results["c_index_ipcw"])

    summary = {
        "loss": {
            "avg": float(np.mean(losses)),
            "std": float(np.std(losses))
        },
        "c_index": {
            "avg": float(np.mean(c_indices)),
            "std": float(np.std(c_indices))
        },
        "c_index_ipcw": {
            "avg": float(np.mean(c_indices_ipcw)),
            "std": float(np.std(c_indices_ipcw))
        }
    }

    final_res["Summary"] = summary
    save_json(args.result_dir, f'Final_results_{args.omics_type}.json', final_res)
    return final_res


def k_fold_train(args, device):
    """ K-fold cross-validation - Train and Test. """
    final_res = {}
    save_exp_settings(args)
    for i in range(args.folds):
        results = survival_train(args, i, device)
        final_res[f'Fold{i}'] = results
    
def main(args):
    """ K-fold cross-validation for Survival Prediction """
    wandb.init(project="survival-prediction-with-adapters", config=vars(args))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.result_dir = os.path.join(args.result_dir, args.task, args.exp_code)
    args.log_dir = os.path.join(args.log_dir, args.task, args.exp_code)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.mode == "train":
        k_fold_train(args, device)
    elif args.mode == "test":
        k_fold_test(args, device)
    else:
        sys.exit("Unspecified mode! Abborting..")
    
    print("FINISHED!\n\n\n")
    wandb.finish()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Survival Prediction on RNA-seq data')

    # optimizer settings
    parser.add_argument('--max_epochs', type=int, default=30, help='maximum number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['cosine', 'linear', 'constant'], default='cosine')
    parser.add_argument('--warmup_steps', type=int, default=-1, help='warmup iterations')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='warmup epochs')

    # misc 
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--num_workers', type=int, default=2)

    # Model args
    parser.add_argument('--model', default='mlp', choices=['mlp', 'snn', 'pathway_mlp', 'pathway_snn', 'gene_dimaf'], help='Model type')
    parser.add_argument('--network_size', default='small', choices=['small', 'big'], help='Size of the network')
    parser.add_argument('--aggregation_type', default='concat', choices=['concat', 'sum', 'mean', 'wm'])

    # experiment task / label args
    parser.add_argument('--task', type=str, default='dss_survival_brca')
    parser.add_argument('--target_col', type=str, default='dss_survival_days')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--mode', type=str, default='train', choices=['test', 'train']) 
    parser.add_argument('--loss_fn', type=str, default='cox', choices=['cox'], help='Loss function to use for training')

    # dataset args
    parser.add_argument('--omics_type', default='rna_clean_norm')
    parser.add_argument('--data_source', type=str, default='data/files/tcga_brca/', help='manually specify the data source')
    parser.add_argument('--expression_data_path', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_type', type=str, default='csv')

    # logging args
    parser.add_argument('--result_dir', default='results',help='results directory')
    parser.add_argument('--log_dir', default='logs',help='results directory')
    parser.add_argument('--exp_code', type=str, default='test', help='experiment code for saving results')

    # limiting training set size args
    parser.add_argument('--train_subset', type=float, default=1, help='how big part of a training set should be used')

    args = parser.parse_args()

    main(args)