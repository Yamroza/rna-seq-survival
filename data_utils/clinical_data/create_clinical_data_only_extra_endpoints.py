import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="merge tcga clinical data with survival endpoints")
    
    parser.add_argument("--data_path", type=str, default="../../data", help="root path to the data folder")
    parser.add_argument("--cohort", type=str, default="OV", help="cohort name (e.g., OV, BRCA)")

    args = parser.parse_args()
    
    df_final = prepare_clinical_data(
        data_path=args.data_path,
        cohort=args.cohort,
    )
    
    print(f"final clinical data shape: {df_final.shape}")
    print("done.")
    

def prepare_clinical_data(data_path, cohort, save=True):
    """
    combines rna data with extra clinical endpoints
    and cleans survival information.
    """
    
    # 1. load rna data
    cbio_path = os.path.join(data_path, "0_data_for_mlp", f"TCGA-{cohort}.star_tpm.csv")
    print(f"loading rna data from: {cbio_path}")
    
    df_cbio = pd.read_csv(cbio_path)
    df_cbio = df_cbio[["Unnamed: 0"]]
    df_cbio = df_cbio.rename(columns={
        "Unnamed: 0": "patient_id"
    })

    # 2. load extra endpoints (dss survival)
    extra_path = os.path.join(data_path, "clinical_data", "extra_endpoints.csv")
    print(f"loading extra endpoints from: {extra_path}")
    
    df_extr = pd.read_csv(extra_path, sep=';', index_col=0)
    df_extr = df_extr[["bcr_patient_barcode", "DSS.time.cr", "DSS_cr"]]
    df_extr = df_extr.rename(columns={
        "bcr_patient_barcode": "patient_id",
        "DSS.time.cr": "dss_survival_days",
        "DSS_cr": "dss_censorship"
    })

    # 3. merge and clean data
    joined = pd.merge(df_cbio, df_extr, on=['patient_id'])
    
    # remove rows with missing survival or censorship info
    n_nan_survival = joined['dss_survival_days'].isna().sum()
    n_nan_censor = joined['dss_censorship'].isna().sum()
    
    print(f"removing {n_nan_survival} rows with NaN survival days")
    print(f"removing {n_nan_censor} rows with NaN censorship")
    
    joined = joined[joined['dss_survival_days'].notna()]
    joined = joined[joined['dss_censorship'].notna()]

    # 4. standardize censorship signs to match project data
    # mapping: 0 -> 1, 1 -> 0, 2 -> 1
    joined['dss_censorship'] = joined['dss_censorship'].map({0: 1, 1: 0, 2: 1})

    # 5. final formatting
    # use patient_id as case_id and drop duplicates
    joined = joined.rename(columns={'patient_id': 'case_id'})
    joined = joined.drop_duplicates(subset='case_id', keep='first')

    # 6. saving
    output_dir = os.path.join(data_path, "clinical_data")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{cohort}_clinical.csv")
    joined.to_csv(out_path)
    print(f"clinical data saved to: {out_path}")

    return joined


if __name__ == "__main__":
    main()