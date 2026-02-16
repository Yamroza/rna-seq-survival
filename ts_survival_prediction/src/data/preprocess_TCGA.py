import pandas as pd 
import argparse

def check_files(data_type): 
    """ Check which patients have both RNA, WSI, and clinical data in all folds. """
    folds = [0, 1, 2, 3, 4]
    cases_from_folds  = []

    for i in folds:
        df2 = pd.read_csv(f'files/tcga_{data_type}/splits/{i}/test_filtered.csv', delimiter=',')
        df3 = pd.read_csv(f'files/tcga_{data_type}/splits/{i}/train_filtered.csv', delimiter=',')
        cases_to_keep = list(df2[df2.columns[0]].values.flatten())
        cases_splits3 = list(df3[df3.columns[0]].values.flatten())
        cases_to_keep.extend(cases_splits3)
        cases_from_folds.append(set(cases_to_keep))

    assert all(s == cases_from_folds[0] for s in cases_from_folds), "Not all sets are the same"

    # Return a list of patients to keep
    return cases_to_keep


def preprocess_data(df_raw, cases_to_keep, remove=False):
    """ Preprocess the raw data and save it in the correct format. """

    if remove:
        df_raw_T = remove_non_HM_genes(df_raw)
    else:
        df_raw_T = df_raw

    df_raw_T = df_raw_T.set_index('sample')
    df_raw_T = df_raw_T.transpose()
    df_raw_T = df_raw_T.sort_index(axis=0)
    df_raw_T.index.name = None
    df_raw_T = df_raw_T.reset_index()
    df_raw_T.columns.name = None

    df_raw_T = df_raw_T.rename(columns={'index': 'sample'})

    # Drop the samples from the normal tissue
    df_filtered = df_raw_T[~df_raw_T['sample'].str.endswith('-11')].reset_index(drop=True)

    # # Keep only the samples from the primary tissue
    df_filtered['sample'] = df_filtered['sample'].str.replace(r'-01', '', regex=True)

    # # Keep only the samples from the splits files; samples used in training
    df_filtered_complete = df_filtered[df_filtered['sample'].isin(cases_to_keep)].reset_index(drop=True)
    df_filtered_complete = df_filtered_complete.rename(columns={'sample': 'Unnamed: 0'})
    
    # print(df_filtered_complete.head())
    return df_filtered_complete
    
def remove_non_HM_genes(df):
    """ Remove genes that are never in the Hallmark Gene Sets from the dataframe. """
    hallmarks = pd.read_csv('files/hallmarks_signatures.csv').values.flatten()
    hallmarks = hallmarks[~pd.isnull(hallmarks)]
    hallmarks = pd.unique(hallmarks)

    df_filtered = df[df['sample'].isin(hallmarks)].reset_index(drop=True)
    return df_filtered

def main(args):
    # Perform data preprocessing steps here
    df_raw = pd.read_csv(f'files/tcga_{args.data}/HiSeqV2_PANCAN_{args.data.upper()}', delimiter='\t')

    # Check if the data is loaded correctly
    cases_to_keep = check_files(args.data)

    # Preprocess data
    df_filtered_complete = preprocess_data(df_raw, cases_to_keep, args.remove)

    if args.remove:
        name = f'{args.name}_rm'
    else:
        name = f'{args.name}'

    # Save the preprocessed data
    df_filtered_complete.to_csv(f'files/tcga_{args.data}/{name}.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess RNA-seq data')

    parser.add_argument('--remove', action="store_true", help="If true, remove genes that are never in the Hallmark Gene Sets")
    parser.add_argument('--data', default='brca', choices=['blca', 'brca', 'kirc', 'luad'], help='Data cohort to use')
    parser.add_argument('--name', default='test', help='Resulting file name')

    args = parser.parse_args()
    main(args)