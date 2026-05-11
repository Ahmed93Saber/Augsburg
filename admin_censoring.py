import pandas as pd
import numpy as np


def load_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    """Loads the dataset from a CSV or Excel file."""
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")


def impute_administrative_censoring(
    df: pd.DataFrame,
    lock_date_str: str = "2020-04-13",
    diag_col: str = "Date_of_diagnosis",
    lfu_col: str = "date_of_LFU",
    death_date_col: str = "Date_of_death",
    event_date_col: str = "Date_of_event",
    efs_col: str = "EFS_days",
    os_col: str = "OS_days",
    event_col: str = "Event",
    death_col: str = "Death_disease",
) -> pd.DataFrame:
    """Identifies cases missing all follow-up/event dates, imputes right-censored survival

    durations based on an administrative lock date, and adjusts event indicators to censored (1.0).
    """
    df_clean = df.copy()

    # Parse dates to datetime objects for accurate arithmetic
    lock_date = pd.to_datetime(lock_date_str)
    diag_dates = pd.to_datetime(df_clean[diag_col], errors="coerce")

    # 1. Identify the target cohort: missing LFU, Death, and Event dates
    target_mask = (
        df_clean[lfu_col].isna()
        & df_clean[death_date_col].isna()
        & df_clean[event_date_col].isna()
    )

    # 2. Calculate administrative durations for target cases that have a valid diagnosis date
    valid_diag_mask = target_mask & diag_dates.notna()
    computed_durations = (lock_date - diag_dates[valid_diag_mask]).dt.days

    # Calculate the median duration to impute for cases completely missing the diagnosis date
    median_duration = (
        computed_durations.median() if not computed_durations.empty else np.nan
    )

    # 3. Apply Imputations for EFS_days and OS_days
    # A. Cases with a diagnosis date
    df_clean.loc[valid_diag_mask, efs_col] = computed_durations
    df_clean.loc[valid_diag_mask, os_col] = computed_durations

    # B. Cases without a diagnosis date
    missing_diag_mask = target_mask & diag_dates.isna()
    if missing_diag_mask.any() and pd.notna(median_duration):
        df_clean.loc[missing_diag_mask, efs_col] = median_duration
        df_clean.loc[missing_diag_mask, os_col] = median_duration

    # 4. Required Indicator Adjustments
    # Set binary indicators to 1.0 (censored / no event) for the entire target cohort
    df_clean.loc[target_mask, event_col] = 1.0
    df_clean.loc[target_mask, death_col] = 1.0

    imputed_count = target_mask.sum()
    print(
        f"Processing complete: {imputed_count} cases updated via administrative censoring."
    )
    if pd.notna(median_duration):
        print(
            f"Applied median duration of {median_duration:.0f} days to {missing_diag_mask.sum()} cases missing diagnosis dates."
        )

    return df_clean


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Saves the processed DataFrame to the specified path."""
    if output_path.endswith(".csv"):
        df.to_csv(output_path, index=False)
    elif output_path.endswith(".xlsx"):
        df.to_excel(output_path, index=False)
    else:
        raise ValueError("Output path must end in .csv or .xlsx")
    print(f"File successfully saved to {output_path}")


# --- Pipeline Execution ---
if __name__ == "__main__":
    input_file = r"W:\pathologie\bioinfo-archive\UK_Augsburg_Claus\Pediatric_adrenocortical_carcinoma\Ahmed_image\analysis\ped.xlsx"
    sheet = 'CuratedV3'
    admin_date = "2020-04-13"  # last recorded follow-up date in the dataset, used for administrative censoring
    output_file = "dataframes/pediatric_imputed_survival.csv"

    # Execute modular steps
    raw_df = load_data(input_file, sheet_name=sheet)
    processed_df = impute_administrative_censoring(raw_df, lock_date_str=admin_date)
    save_data(processed_df, output_file)