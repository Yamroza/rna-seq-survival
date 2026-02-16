import torch.optim as optim
import os
import json
from transformers import (get_constant_schedule_with_warmup, 
                         get_linear_schedule_with_warmup, 
                         get_cosine_schedule_with_warmup)

def get_optim(args, model):
    """Get the specified optimizer, adapted from https://github.com/mahmoodlab/MMP/blob/main/src/utils/utils.py """
    def exclude(
        n, p): return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n

    def include(n, p): return not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(
        n, p) and p.requires_grad]
    parameters = [
        {"params": gain_or_bias_params, "weight_decay": 0.},
        {"params": rest_params, "weight_decay": args.wd},
    ]
    optimizer = optim.AdamW(parameters, lr=args.lr)
    
    return optimizer


def get_lr_scheduler(args, optimizer, n_data):
    """Get the specified learning rate scheduler, adapted from https://github.com/mahmoodlab/MMP/blob/main/src/utils/utils.py"""
    scheduler_name = args.lr_scheduler
    warmup_steps = args.warmup_steps
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs
    assert not (warmup_steps > 0 and warmup_epochs > 0), "Cannot have both warmup steps and epochs"
 
    if warmup_steps > 0:
        warmup_steps = warmup_steps
    elif warmup_epochs > 0:
        warmup_steps = warmup_epochs * n_data
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=warmup_steps)
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=n_data * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=n_data * epochs,
        )
    return lr_scheduler

    
def log_results(writer, results, epoch, mode='train'):
    """ Log results to TensorBoard. """
    for item, value in results.items():
        writer.add_scalar(f'{mode}_{item}', value, epoch)

def save_exp_settings(args):
	""" Save settings of an experiment. """
	os.makedirs(args.result_dir, exist_ok=True)

	# Save the args namespace to a JSON file
	args_dict = vars(args)  # Convert Namespace to dictionary
	json_path = os.path.join(args.result_dir, 'args.json')

	with open(json_path, 'w') as json_file:
		json.dump(args_dict, json_file, indent=4)