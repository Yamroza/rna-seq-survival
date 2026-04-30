import numpy as np

from data.survival_dataset import RNASurvivalDataset
from torch.utils.data import DataLoader, Subset


def obtain_dataloader(args, fold, mode="train"):
    """ Obtain the dataset and dataloader. """
    type = obtain_data_type(args) # gene or pathway level
    dataset = RNASurvivalDataset(args, mode, type, fold)

    # choosing a random subset
    if mode == "train" and hasattr(args, 'train_subset') and args.train_subset < 1.0:
        np.random.seed(args.seed)
        
        num_samples = len(dataset)
        subset_size = int(num_samples * args.train_subset)
    
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        subset_indices = indices[:subset_size]
        
        dataset = Subset(dataset, subset_indices)
        
        print(f"Using a subset of {args.train_subset*100}% for training.")

    print(f"Dataset {mode} for fold {fold} is constructed and checked!")
    print(f'Split: {fold}, n: {len(dataset)}')

    shuffle_mode = mode == "train"
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle_mode, num_workers=args.num_workers)
    
    # choose correct rna_dims depending on the mode
    rna_dims = dataset.dataset.get_rna_dims() if isinstance(dataset, Subset) else dataset.get_rna_dims()
    return dataloader, rna_dims


def obtain_data_info(args, fold, mode="train"):
    """ Obtain the data info (all censorships and survival times) for a specific fold. """
    type = obtain_data_type(args)
    dataset = RNASurvivalDataset(args, mode=mode, type=type, fold=fold)

    all_censorships, all_event_times = dataset.get_all_labels()
    return {'censorship': all_censorships, 'time': all_event_times}

def obtain_data_type(args):
    """ Obtain the type of data based on the omics type (gene or pathway level). """
    if args.model in ['mlp', 'snn']:
        return 'rna'
    elif args.model in ['pathway_mlp', 'pathway_snn', 'gene_dimaf']:
        return 'pathways'
    else:
        raise ValueError(f"Omics type of model {args.model} not recognized.")

