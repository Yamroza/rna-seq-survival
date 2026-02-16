import os
import torch
import numpy as np

from model.model_factory import obtain_model
from survival.losses import CoxLoss

from utils.general_utils import save_json, input_to_device, save_pkl
from utils.test_utils import compute_survival_metrics, LoggingMeter
from utils.data_utils import obtain_data_info, obtain_dataloader


def test(model, test_dl, device, survival_info_train=None, result_dir_scores=None):
    """ Test a survival prediction model for a single fold. """
    model.eval()

    all_case_ids, all_slide_ids = [], []
    all_risk_scores, all_censorships, all_event_times = [], [], []
    test_log = {}

    # Loop over data
    with torch.no_grad():
        for idx, batch in enumerate(test_dl):
            # Get the data and labels
            rna = input_to_device(batch['rna'], device)
            label = batch['label'].to(device)
            event_time = batch['survival_time'].to(device)
            censorship = batch['censorship'].to(device)

            # forward pass
            output_results, log_dict = model(rna, label=label, censorship=censorship)
            all_case_ids.append(np.array(batch['case_id']))
            # all_slide_ids.append(np.array(batch['slide_id']))

            # Logging
            for key, val in log_dict.items():
                if key not in test_log:
                    test_log[key] = LoggingMeter(key)
                test_log[key].update(val, n=len(label))

            all_risk_scores.append(output_results['risk'].detach().cpu().numpy())
            all_censorships.append(censorship.cpu().numpy())
            all_event_times.append(event_time.cpu().numpy())
    
        all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
        all_censorships = np.concatenate(all_censorships).squeeze(1)
        all_event_times = np.concatenate(all_event_times).squeeze(1)

        # Compute c-index & variant
        c_index, c_index_ipcw = compute_survival_metrics(all_censorships, all_event_times, all_risk_scores, survival_info_train)

        # Save the results
        if result_dir_scores:
            risk_scores_dict = {'case_ids': np.concatenate(all_case_ids, axis=0),
                                # 'slide_ids': np.concatenate(all_slide_ids, axis=0),
                                'Risk scores': all_risk_scores, 
                                'Censorship': all_censorships,
                                'Time': all_event_times}
            save_pkl(result_dir_scores, f"pred_test_risk_scores.pkl", risk_scores_dict)
        
        results = {item: meter.avg for item, meter in test_log.items()}
        results.update({'c_index': c_index})
        results.update({'c_index_ipcw': c_index_ipcw})

    return results

def survival_test(args, fold, device):
    """ Load and test survival prediction model. """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir_fold =  os.path.join(args.result_dir, f"Fold_{fold}/")
    pretrained_model_path = os.path.join(result_dir_fold, "model_checkpoint.pth")

    # Initialize the loss function
    if args.loss_fn == 'cox':
        loss_fn = CoxLoss()
        num_classes = 1
    else:
        raise ValueError(f"Loss function {args.loss_fn} not implemented.")

    # Obtain the dataloader
    print('\nObtaining dataloader...', end='\n')
    test_dl, rna_dims = obtain_dataloader(args, fold=fold, mode="test")

    # Try to obtain the training data info
    try:
        train_data_info = obtain_data_info(args, fold=fold, mode="train")
    except Exception as e:
        train_data_info = None
        print("No training data available for this fold. Skipping IPCW c-index calculation.")

    # Load model
    model = obtain_model(args, loss_fn=loss_fn, num_classes=num_classes, rna_dims=rna_dims, device=device)
    model.from_pretrained(pretrained_model_path, device)

    # Test the model
    results = test(model, test_dl, device, survival_info_train=train_data_info, result_dir_scores=result_dir_fold)
    save_json(result_dir_fold, f"test_summary.json", results)
    return results


