import random
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def preprocess_survival_data(
    df: pd.DataFrame,
    continuous_cols: list,
    categorical_cols: list,
    cluster_cols: list = ["cluster_1", "cluster_2"],  # cluster_0 omitted as reference
    time_col: str = "OS_days_bis",
    event_col: str = "Death_disease",
) -> pd.DataFrame:
    """Preprocesses survival data by applying administrative censoring imputation,
    mapping survival indicators to 0/1, standardizing continuous predictors, and
    one-hot encoding categorical features while dropping collinear baseline reference columns.
    """
    df_processed = df.copy()

    # 2. Map Event Indicator to Standard Binary (0 = Censored, 1 = Event)
    status_series = df_processed[event_col].map({1.0: 0, 2.0: 1})
    df_processed["status"] = status_series.fillna(0).astype(int)

    # Compile initial feature columns
    all_covariates = continuous_cols + categorical_cols
    analysis_cols = [time_col, "status"] + cluster_cols + all_covariates
    df_subset = df_processed[analysis_cols].copy()

    # 3. Impute missing values separately based on data type distribution
    if continuous_cols:
        cont_imputer = SimpleImputer(strategy="median")
        df_subset[continuous_cols] = cont_imputer.fit_transform(df_subset[continuous_cols])

    # if categorical_cols:
    #     cat_imputer = SimpleImputer(strategy="most_frequent")
    #     df_subset[categorical_cols] = cat_imputer.fit_transform(df_subset[categorical_cols])

    # 4. Standardize continuous covariates explicitly for L2 Regularization
    if continuous_cols:
        scaler = StandardScaler()
        df_subset[continuous_cols] = scaler.fit_transform(df_subset[continuous_cols])

    # Ensure pre-encoded cluster columns are represented as clean integers
    for col in cluster_cols:
        if col in df_subset.columns:
            df_subset[col] = df_subset[col].fillna(0).astype(int)

    # 5. One-hot encode categorical features and drop perfectly collinear columns
    if categorical_cols:
        df_subset = pd.get_dummies(
            df_subset,
            columns=categorical_cols,
            drop_first=True,
            dtype=int
        )

    return df_subset.dropna(subset=[time_col, "status"])


def run_stratified_monte_carlo_cv(
    df: pd.DataFrame,
    time_col: str = "OS_days_bis",
    event_col: str = "status",
    n_splits: int = 100,
    test_size: float = 0.25,
    penalizer: float = 0.5,  # L2 penalty term acting as regularizer for sparse data
    random_state: int = 42,
):
    """Executes 10-fold Stratified Monte-Carlo Cross-Validation fitting a Penalized Cox model."""
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )

    c_indices = []
    fold_hazard_ratios = []
    fold_p_values = []

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
            fold_p_values.append(cph.summary['p'])

        except Exception as e:
            print(f"Fold {fold} failed to converge: {e}")
            continue

    # Compile and report results
    if c_indices:

        print("\n=== Cross-Validation Results ===")
        print(f"Mean Out-of-Fold C-index: {np.mean(c_indices):.4f} ± {np.std(c_indices):.4f}\n")

        # Compile average HR and p-values across all successfully converged folds
        hr_mean = pd.concat(fold_hazard_ratios, axis=1).median(axis=1)
        p_mean = pd.concat(fold_p_values, axis=1).min(axis=1)

        cv_summary = pd.DataFrame({'Mean HR': hr_mean, 'Mean CV p-value': p_mean})
        print("=== Average Estimates Across Folds ===")
        print(cv_summary.round(4))
    else:
        print("CV failed across all folds due to extreme data sparsity.")


if __name__ == "__main__":
    # Initialize seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    input_filepath = "dataframes/Final_cohort_with_kmeans_clusters.csv"
    time_col = "OS_days_max"

    # Load original dataset
    try:
        raw_df = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(
            f"Warning: File {input_filepath} not found. Ensure the working directory contains the file."
        )
        exit()

    # Define continuous vs categorical variables cleanly
    continuous_features = []
    categorical_features = ["tumor_stage", "Necroses", "Vascular_invasion",
                            "Diagnosis", "Distant_metastases"]  # Will dynamically generate columns like TP53_2.0, TP53_9.0

    # Preprocess and execute pipeline
    processed_df = preprocess_survival_data(
        df=raw_df,
        continuous_cols=continuous_features,
        categorical_cols=categorical_features,
        cluster_cols=["cluster_1", "cluster_2"],  # cluster_0 omitted as baseline reference
        time_col=time_col,
        event_col="Death_disease",
    )

    print(f"\n{time_col}")
    run_stratified_monte_carlo_cv(
        df=processed_df,
        time_col=time_col,
        n_splits=200,
        penalizer=0.5,
    )