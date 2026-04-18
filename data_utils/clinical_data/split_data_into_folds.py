import os
import argparse
import pandas as pd
from sklearn.model_selection import GroupKFold

def main():
    parser = argparse.ArgumentParser(description="split tcga clinical data into folds using GroupKFold")
    
    parser.add_argument("--data_path",  type=str, default="../data/clinical_data", help="path to clinical data folder")
    parser.add_argument("--cohort",     type=str, default="ov", help="cohort name (prefix for clinical csv)")
    parser.add_argument("--n_folds",    type=int, default=5, help="number of folds for cross-validation")

    args = parser.parse_args()

    out_dir = split_data_into_folds(
        data_path=args.data_path,
        cohort=args.cohort,
        n_folds=args.n_folds,
    )
    
    print(f"all folds saved to: {out_dir}")


def split_data_into_folds(data_path, cohort, n_folds=5, save=True):
    """
    splits clinical data into k-folds using GroupKFold based on patient location.
    ensures critical observations (max event times) are moved to train sets.
    """
    filename = f"{cohort}_clinical"
    file_path = os.path.join(data_path, f"{filename}.csv")
    
    print(f"loading data from: {file_path}")
    df = pd.read_csv(file_path, index_col=0)

    # 1. identifying "location" from tcga barcode (e.g., TCGA-04-1331 -> 04)
    # this is used to group samples from the same center together
    df['location'] = df['case_id'].apply(lambda x: x.split('-')[1])

    # 2. handle critical observations
    # we find observations with the absolute maximum event time (censorship == 0)
    events_df = df[df['dss_censorship'] == 0]
    max_event_time = events_df['dss_survival_days'].max()
    
    critical_idx = events_df[events_df['dss_survival_days'] == max_event_time].index
    
    # separate critical rows from the rest for stable splitting
    df_rest = df.drop(index=critical_idx).reset_index(drop=True)
    critical_df = df.loc[critical_idx].reset_index(drop=True)

    # 3. prepare for splitting
    gkf = GroupKFold(n_splits=n_folds)
    split_dir = os.path.join(data_path, filename, "splits")
    
    os.makedirs(split_dir, exist_ok=True)

    print(f"starting {n_folds}-fold split by location...")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df_rest, groups=df_rest['location'])):
        fold_path = os.path.join(split_dir, str(fold))
        if save:
            os.makedirs(fold_path, exist_ok=True)

        train_df = df_rest.iloc[train_idx].reset_index(drop=True)
        test_df = df_rest.iloc[test_idx].reset_index(drop=True)

        # 4. handle survival-specific constraints
        # we check if any event in test set is longer than max event in train set
        train_events = train_df[train_df['dss_censorship'] == 0]
        test_events = test_df[test_df['dss_censorship'] == 0]

        if len(test_events) > 0:
            max_train_event_time = train_events['dss_survival_days'].max()
            # find samples in test that exceed train timeline
            problematic = test_events[test_events['dss_survival_days'] > max_train_event_time]

            if len(problematic) > 0:
                # move problematic samples from test to train to ensure model coverage
                train_df = pd.concat([train_df, problematic], ignore_index=True)
                test_df = test_df.drop(problematic.index).reset_index(drop=True)

        # 5. save folds
        train_df.to_csv(os.path.join(fold_path, "train_filtered.csv"), index=False)
        test_df.to_csv(os.path.join(fold_path, "test_filtered.csv"), index=False)

        print(
            f"fold {fold}: "
            f"train={len(train_df)}, "
            f"test={len(test_df)}, "
            f"critical_samples_moved={len(critical_df)}"
        )

    return split_dir


if __name__ == "__main__":
    main()