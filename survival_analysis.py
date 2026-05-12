import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import random


def preprocess_survival_data(
    df: pd.DataFrame,
    continuous_cols: list,
    categorical_cols: list,
    cluster_cols: list = ["cluster_1", "cluster_2"],  # cluster_0 omitted as reference
    time_col: str = "OS_days_bis",
    event_col: str = "Death_disease",
    lock_date_str: str = "2020-04-13",
) -> pd.DataFrame:
    """Preprocesses survival data by applying administrative censoring imputation,

    mapping survival indicators to 0/1, imputing missing covariates, and
    standardizing continuous predictors for regularized Cox regression.
    """

    df_processed = df.copy()

    # Ensure censored status (1.0 in original encoding)
    df_processed.loc[time_col, event_col] = 1.0

    # 2. Map Event Indicator to Standard Binary (0 = Censored, 1 = Event)
    # Original encoding: 1.0 = No Event/Censored, 2.0 = Event/Death
    status_series = df_processed[event_col].map({1.0: 0, 2.0: 1})
    df_processed["status"] = status_series.fillna(0).astype(int)

    # Compile all required feature columns
    all_covariates = continuous_cols + categorical_cols
    analysis_cols = [time_col, "status"] + cluster_cols + all_covariates
    df_subset = df_processed[analysis_cols].copy()

    # 3. Impute missing values to prevent dropping records
    if all_covariates:
        imputer = SimpleImputer(strategy="median")
        df_subset[all_covariates] = imputer.fit_transform(df_subset[all_covariates])

    # 4. Standardize continuous covariates explicitly for L2 Regularization
    if continuous_cols:
        scaler = StandardScaler()
        df_subset[continuous_cols] = scaler.fit_transform(df_subset[continuous_cols])

    # Ensure cluster columns and categorical columns are represented as clean integers
    for col in cluster_cols + categorical_cols:
        if col in df_subset.columns:
            df_subset[col] = df_subset[col].fillna(0).astype(int)

    return df_subset.dropna(subset=[time_col, "status"])


def run_stratified_monte_carlo_cv(
    df: pd.DataFrame,
    time_col: str = "OS_days_bis",
    event_col: str = "status",
    n_splits: int = 10,
    test_size: float = 0.2,
    penalizer: float = 0.5,  # L2 penalty term acting as regularizer for sparse data
    random_state: int = 42,
):
    """Executes 10-fold Stratified Monte-Carlo Cross-Validation fitting a Penalized Cox model."""
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )

    c_indices = []
    fold_hazard_ratios = []

    X = df.drop(columns=[time_col, event_col])
    y_event = df[event_col].values  # Used strictly for stratification

    print(
        f"Starting {n_splits}-fold Stratified Monte-Carlo CV (Penalizer = {penalizer})...\n"
    )

    for fold, (train_idx, test_idx) in enumerate(sss.split(X, y_event), 1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Initialize Penalized Cox Proportional Hazards Fitter
        cph = CoxPHFitter(penalizer=penalizer)

        try:
            # Fit model on training fold
            cph.fit(train_df, duration_col=time_col, event_col=event_col)

            # Predict partial hazards on test fold
            test_preds = cph.predict_partial_hazard(test_df)

            # Calculate out-of-fold Concordance Index (C-index)
            c_index = concordance_index(
                test_df[time_col], -test_preds, test_df[event_col]
            )
            c_indices.append(c_index)

            # Store hazard ratios (exp(coefficients))
            fold_hazard_ratios.append(cph.hazard_ratios_)

        except Exception as e:
            print(f"Fold {fold} failed to converge: {e}")
            continue

    # Compile and report results
    if c_indices:
        mean_c_index = np.mean(c_indices)
        std_c_index = np.std(c_indices)
        print("=== Cross-Validation Performance ===")
        print(f"Mean Out-of-Fold C-index: {mean_c_index:.4f} ± {std_c_index:.4f}\n")

        print("=== Mean Estimated Hazard Ratios (HR) ===")
        hr_df = pd.concat(fold_hazard_ratios, axis=1).mean(axis=1)
        for feature, hr in hr_df.items():
            print(f"{feature:20s}: HR = {hr:.4f}")
    else:
        print("Model fitting failed across all folds due to extreme data sparsity.")


if __name__ == "__main__":
    # Initialize seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    input_filepath = "dataframes/Final_cohort_with_kmeans_clusters.csv"
    time_col = "OS_days_bis"

    # Load original dataset
    try:
        raw_df = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(
            f"Warning: File {input_filepath} not found. Ensure the working directory contains the file."
        )
        # Exiting cleanly if file is unavailable
        exit()

    # Define continuous vs categorical variables cleanly
    continuous_features = ["Tumor_volume"]
    categorical_features = ["TP53"]

    # Preprocess and execute pipeline
    processed_df = preprocess_survival_data(
        df=raw_df,
        continuous_cols=continuous_features,
        categorical_cols=categorical_features,
        cluster_cols=["cluster_1", "cluster_2"],  # cluster_0 omitted as baseline reference
        time_col=time_col,
        event_col="Death_disease",
    )

    run_stratified_monte_carlo_cv(
        df=processed_df,
        time_col=time_col,
        n_splits=10,
        penalizer=0.5,
    )