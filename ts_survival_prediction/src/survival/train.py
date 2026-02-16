import os
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from sksurv.metrics import concordance_index_censored

from survival.losses import CoxLoss
from model.model_factory import obtain_model

from survival.test import test

from utils.data_utils import obtain_dataloader
from utils.general_utils import save_json, input_to_device
from utils.train_utils import get_optim, get_lr_scheduler, log_results
from utils.test_utils import LoggingMeter


def survival_train(args, fold, device):
    """ Train a survival prediction model for a single fold. """

    # Set up results and log dir.
    result_dir_fold = os.path.join(args.result_dir, f"Fold_{fold}")
    log_dir_fold = os.path.join(args.log_dir, f"Fold_{fold}")

    os.makedirs(result_dir_fold, exist_ok=True)
    os.makedirs(log_dir_fold, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir_fold)

    # Obtain the dataloaders
    print('\nObtaining dataloaders...', end='\n')
    train_dl, rna_dims = obtain_dataloader(args, fold=fold, mode="train")
    test_dl, _ = obtain_dataloader(args, fold=fold, mode="test")

    # Initialize the loss function
    if args.loss_fn == 'cox':
        loss_fn = CoxLoss()
        num_classes = 1
    else:
        raise ValueError(f"Loss function {args.loss_fn} not implemented.")
    
    # Initialize model
    print('\nInit Model...', end=' ')
    model = obtain_model(args, loss_fn=loss_fn, num_classes=num_classes, rna_dims=rna_dims, device=device)

    print('\nInit optimizer ...')
    optimizer = get_optim(model=model, args=args)
    lr_scheduler = get_lr_scheduler(args, optimizer, len(train_dl))

    #####################
    # The training loop #
    #####################
    
    for epoch in range(args.max_epochs):
        # Train
        print('#' * 10, f'TRAIN Epoch: {epoch}', '#' * 10)
        train_results, train_data_info = train_loop(model, train_dl, optimizer, lr_scheduler, device)
        log_results(writer, train_results, epoch, mode='train')
        
        # Save last model
        torch.save(model.state_dict(), os.path.join(result_dir_fold, "model_checkpoint.pth"))


    print(f'End of training. Testing on Split {fold}...:')

    results = test(model, test_dl, device, survival_info_train=train_data_info)
    save_json(result_dir_fold, f"train_test_summary.json", results)

    writer.close()
    return results

def train_loop(model, dataloader, optimizer, lr_scheduler, device):
    """
        Train loop for survival prediction
    """
    model.train()
    train_log = {}
    all_risk_scores, all_censorships, all_event_times = [], [], []

    # Loop over all data samples
    for idx, batch in enumerate(dataloader):
        # Get the data and labels
        rna = input_to_device(batch['rna'], device)

        label = batch['label'].to(device)
        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)

        # Forward pass
        output_results, log_dict = model(rna, label=label, censorship=censorship)

        # Backward pass
        loss = output_results['loss']
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Logging
        for key, val in log_dict.items():
            if key not in train_log:
                train_log[key] = LoggingMeter(key)
            train_log[key].update(val, n=len(label))
        
        all_risk_scores.append(output_results['risk'].detach().cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())
        
    
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    
    # Compute c-index
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    
    results = {item: meter.avg for item, meter in train_log.items()}
    results.update({'c_index': c_index})
    results['lr'] = optimizer.param_groups[0]['lr']
    train_data_info = {'censorship': all_censorships, 'time':all_event_times}
    return results, train_data_info
    