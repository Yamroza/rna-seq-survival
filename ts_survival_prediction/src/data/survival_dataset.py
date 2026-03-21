import os
import torch
import pandas as pd
import numpy as np

from utils.general_utils import save_pkl, load_pkl, overlap_col_df, _series_intersection
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset

import warnings
warnings.filterwarnings('ignore')


class RNASurvivalDataset(Dataset):
    """
        RNA-seq Dataset for survival prediction
    """
    def __init__(self,
                 args, 
                 mode,
                 type, 
                 fold):
        """
        Args:
            - args          : All the arguments given by the user (Obj)
            - mode          : Specifies if we are in 'train' or 'test' mode (str)
            - fold          : Specifies the fold number (int)
            - type          : Type of RNA data, either 'rna' or 'pathways'
        """

        self.mode = mode
        self.fold = fold
        
        # Data source
        self.data_source = args.data_source
        self.split_dir = os.path.join(self.data_source, f'splits/{self.fold}/')
        self.data_type = args.data_type

        # RNA args
        self.omics_type = args.omics_type
        self.type = type
        self.scaler = None
        self.expression_data_path = args.expression_data_path

        # Label args
        self.survival_time_col = args.target_col
        self.censorship_col = args.target_col.split('_')[0] + '_censorship'

        # Setup and check clinical data
        self.init_df()

        # Setup and check RNA data
        self.init_df_rna()


    def check_data_file(self):
        """Check the clinical survival dataset. """

        # Filter out NAN labels
        is_nan_censorship = self.data_df[self.censorship_col].isna()
        if sum(is_nan_censorship) > 0:
            print('# of NaNs in Censorship col, dropping:', sum(is_nan_censorship))
            self.data_df = self.data_df[~is_nan_censorship]

        is_nan_survival = self.data_df[self.survival_time_col].isna()
        if sum(is_nan_survival) > 0:
            print('# of NaNs in Survival time col, dropping:', sum(is_nan_survival))
            self.data_df = self.data_df[~is_nan_survival]

        # Check that each case_id has only one survival value
        num_unique_surv_times = self.data_df.groupby('case_id')[self.survival_time_col].unique().apply(len)
        assert (num_unique_surv_times == 1).all(), 'Each case_id must have only one unique survival value.'

        # check that all survival values are numeric
        assert not pd.to_numeric(self.data_df[self.survival_time_col], errors='coerce').isna().any(), 'Survival values must be numeric.'

        # check that all survival values are positive
        assert (self.data_df[self.survival_time_col] >= 0).all(), 'Survival values must be positive.'

        # check that all censorship values are binary integers
        assert self.data_df[self.censorship_col].isin([0, 1]).all(), 'Censorship values must be binary integers.'

        # Should be no duplicates in splits file
        assert len(list(self.data_df['case_id'].astype(str).unique())) == len(list(self.data_df['case_id'].astype(str))), 'There are duplicates in the given splits file...'

    def check_rna_files(self):
        """ Check that the rna files have no duplicates. """
        # For RNA we have a sample per person.
        assert not self.df_rna['case_id'].duplicated().any(), "There are duplicates in the rna data!"

    def init_df(self):
        """ Set up clinical data of this split. """
        split_file = os.path.join(self.split_dir, f'{self.mode}_filtered.csv')
        self.data_df = pd.read_csv(split_file)
        self.check_data_file()
    
    def init_df_rna(self):
        """ Set up RNA data of this split. """
        # Read RNA data
        self.feat_dir_rna = os.path.join(self.expression_data_path, f"{self.omics_type}")
        if os.path.isfile(self.feat_dir_rna):
            if self.data_type == 'csv':
                self.df_rna = pd.read_csv(self.feat_dir_rna, engine='python')#, index_col=0)
                self.df_rna = self.df_rna.rename(columns={'Unnamed: 0': 'case_id'})
            elif self.data_type == 'json':
                df_raw = pd.read_json(self.feat_dir_rna, lines=True)
                self.df_rna = pd.concat([
                    df_raw[['id']].rename(columns={'id': 'case_id'}), 
                    pd.DataFrame(df_raw['embedding'].tolist())
                ], axis=1)
        else:
            raise FileNotFoundError(f"{self.feat_dir_rna} not found!")
        
        # Check for duplicates
        self.check_rna_files()

        # Keep only the patients for which we have clinical data and RNA data
        case_ids_overlap = overlap_col_df(self.df_rna, self.data_df, 'case_id')
        sample_list = sorted(case_ids_overlap)

        self.data_df = self.data_df[self.data_df['case_id'].isin(sample_list)].reset_index(drop=True)
        self.df_rna = self.df_rna[self.df_rna['case_id'].isin(sample_list)].reset_index(drop=True)
        self.df_rna = self.df_rna.set_index('case_id')

        # Set up hallmark pathways is necessary
        if self.type == "pathways":
            self.setup_rna_pathways()

        # Initialize and apply scaler for RNA data
        self.setup_scaler()
        self.apply_scaler()
    
    def setup_scaler(self):
        """ Fit or load scaler for RNA data. """
        if self.mode == 'train':
            # Fit the scaler on the training data
            self.scaler = StandardScaler().fit(self.df_rna)
            save_pkl(self.split_dir, f'scaler_{self.omics_type}.pkl', self.scaler)
        else:
            try:
                # Read the scaler from the pickle file
                self.scaler = load_pkl(self.split_dir, f'scaler_{self.omics_type}.pkl')
            except FileNotFoundError:
                print(f"Cannot access the scaler from training. Make sure '{os.path.join(self.split_dir, f'scaler_{self.omics_type}.pkl')} exists.")
                self.scaler = None

    def apply_scaler(self):
        """ Apply fitted scaler (from train data) to RNA test data. """
        assert not self.scaler == None, "Cannot scale the data, scaler is not defined!"

        cols = self.df_rna.columns
        case_list = self.df_rna.index.values
        self.df_rna = pd.DataFrame(self.scaler.transform(self.df_rna), columns=cols)
        self.df_rna.insert(0, 'case_id', case_list)
        self.df_rna = self.df_rna.set_index('case_id')

    def setup_rna_pathways(self):
        """ Load Hallmarks biological pathways, which serve as the prototypes. """
        signatures = pd.read_csv(os.path.join(self.data_source, f"hallmarks_signatures.csv"))
        self.rna_names = []
        self.pathway_names = []
        self.pathway_sizes = []

        # For each pathway 
        for col in signatures.columns:
            omic = signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.df_rna.columns))

            # Store all genes involved in this pathway
            self.rna_names.append(omic)

            # Store the name of the pathway
            self.pathway_names.append(col)

            # Store the number of genes in the pathway
            self.pathway_sizes.append(len(omic))

    def get_rna_dims(self):
        """ Get the dimensions of the RNA data. """
        if self.type == 'pathways':
            return self.pathway_sizes
        else:
            return self.df_rna.shape[1]

    def __len__(self):
        """ Get the number of samples. """
        return len(self.data_df)
    
    def get_all_labels(self):
        """ Get censorships and survival times of all samples. """
        cs_list = list(self.data_df.loc[:][self.censorship_col])
        st_list = list(self.data_df.loc[:][self.survival_time_col])
        return np.array(cs_list), np.array(st_list)
    
    def get_labels(self, idx):
        """ Get the survival time (days), censorship and target label (either continuous or discrete time labels). """
        labels = self.data_df.loc[idx][[self.survival_time_col, self.censorship_col, self.survival_time_col]]
        return list(labels)
    
    def __getitem__(self, idx):
        # Obtain labels
        survival_time, censorship, label = self.get_labels(idx)
        out = {'survival_time': torch.Tensor([survival_time]),
            'censorship': torch.Tensor([censorship]),
            'label': torch.Tensor([label])}
        
        # Obtain patient id
        case_id = self.data_df.loc[idx]['case_id']
        # slide_id = self.data_df.loc[idx]['slide_id']
        out['case_id'] = case_id
        # out['slide_id'] = slide_id

        # Obtain RNA data
        if self.type == 'pathways':
            pathway_summary = []
            # For each pathway:
            for i in range(len(self.pathway_names)):
                # Store all expression data of genes in this pathway of this patient
                pathway_summary.append(torch.Tensor(self.df_rna.loc[case_id][self.rna_names[i]]))
            
            out['rna'] = pathway_summary
        else:
            # Obtain all gene expression data of this patient
            rna_data = torch.Tensor(self.df_rna.loc[case_id].values)
            out['rna'] = rna_data

        return out


    



        




