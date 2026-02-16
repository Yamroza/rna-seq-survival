from data.survival_dataset import RNASurvivalDataset
from torch.utils.data import DataLoader


def obtain_dataloader(args, fold, mode="train"):
    """ Obtain the dataset and dataloader. """
    type = obtain_data_type(args) # gene or pathway level
    dataset = RNASurvivalDataset(args, mode, type, fold)

    print(f"Dataset {mode} for fold {fold} is constructed and checked!")
    print(f'Split: {fold}, n: {len(dataset)}')

    shuffle_mode = mode == "train"
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle_mode, num_workers=args.num_workers)
    
    return dataloader, dataset.get_rna_dims()


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

