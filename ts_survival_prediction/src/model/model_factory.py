from model.FNN import FNN
from model.pathway_FNN import PathwayFFN
from model.DIMAF_SD import Gene_DIMAF


def obtain_model(args, loss_fn, num_classes, rna_dims, device):
    """ 
        Obtain the model based on the specified type. 
        - args:        Arguments containing model type and other configurations.
        - rna_dims:    Dimensions of the RNA data --> number of genes or [len(pathway) for pathway in pathways].
        - device:      Device to run the model on (CPU or GPU).
        - loss_fn:     Loss function to be used.
        - num_classes: Number of output classes (default: 1 for coxPH loss)."""

    # Dictionary for hidden layers based on network size
    network_size_dict = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}

    # Obtain the model
    if args.model == 'mlp':
        model = FNN(rna_dims=rna_dims, block_type='mlp', hidden_layers=network_size_dict[args.network_size], loss_fn=loss_fn, num_classes=num_classes)
    elif args.model == 'snn':
        model = FNN(rna_dims=rna_dims, block_type='snn', hidden_layers=network_size_dict[args.network_size], loss_fn=loss_fn, num_classes=num_classes)
    elif args.model == 'pathway_snn':
        model = PathwayFFN(rna_dims=rna_dims, block_type='snn', hidden_layers=network_size_dict[args.network_size], loss_fn=loss_fn, num_classes=num_classes, aggregation_type=args.aggregation_type)
    elif args.model == 'pathway_mlp':
        model = PathwayFFN(rna_dims=rna_dims, block_type='mlp', hidden_layers=network_size_dict[args.network_size], loss_fn=loss_fn, num_classes=num_classes, aggregation_type=args.aggregation_type)
    elif args.model == 'gene_dimaf':
        model = Gene_DIMAF(rna_dims=rna_dims, loss_fn=loss_fn, num_classes=num_classes, aggregation_type=args.aggregation_type)
    else:
        raise ValueError(f"Model type {args.model} not recognized.")
    
    # Put model on device
    model.to(device)
    
    return model