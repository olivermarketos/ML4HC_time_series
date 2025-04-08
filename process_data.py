import pandas as pd
import numpy as np
import os
from tqdm import tqdm

STATIC_VARS = ["Age","Gender","Height","ICUType", "Weight"]

FEATURES = {'Albumin': 'Serum Albumin (g/dL)',
    'ALP': 'Alkaline phosphatase (IU/L)',
    'ALT': 'Alanine transaminase (IU/L)',
    'AST': 'Aspartate transaminase (IU/L)',
    'Bilirubin': 'Bilirubin (mg/dL)',
    'BUN': 'Blood urea nitrogen (mg/dL)',
    'Cholesterol': 'Cholesterol (mg/dL)',
    'Creatinine': 'Serum creatinine (mg/dL)',
    'DiasABP': 'Invasive diastolic arterial blood pressure (mmHg)',
    'FiO2': 'Fractional inspired O2 (0-1)',
    'GCS': 'Glasgow Coma Score (3-15)',
    'Glucose': 'Serum glucose (mg/dL)',
    'HCO3': 'Serum bicarbonate (mmol/L)',
    'HCT': 'Hematocrit (%)',
    'HR': 'Heart rate (bpm)',
    'K': 'Serum potassium (mEq/L)',
    'Lactate': 'Lactate (mmol/L)',
    'Mg': 'Serum magnesium (mmol/L)',
    'MAP': 'Invasive mean arterial blood pressure (mmHg)',
    'MechVent': 'Mechanical ventilation respiration (0:false or 1:true)',
    'Na': 'Serum sodium (mEq/L)',
    'NIDiasABP': 'Non-invasive diastolic arterial blood pressure (mmHg)',
    'NIMAP': 'Non-invasive mean arterial blood pressure (mmHg)',
    'NISysABP': 'Non-invasive systolic arterial blood pressure (mmHg)',
    'PaCO2': 'partial pressure of arterial CO2 (mmHg)',
    'PaO2': 'Partial pressure of arterial O2 (mmHg)',
    'pH': 'Arterial pH (0-14)',
    'Platelets': 'Platelets (cells/nL)',
    'RespRate': 'Respiration rate (bpm)',
    'SaO2': 'O2 saturation in hemoglobin (%)',
    'SysABP': 'Invasive systolic arterial blood pressure (mmHg)',
    'Temp': 'Temperature (°C)',
    'TroponinI': 'Troponin-I (μg/L)',
    'TroponinT': 'Troponin-T (μg/L)',
    'Urine': 'Urine output (mL)',
    'WBC': 'White blood cell count (cells/nL)',
    'Weight_VAR': 'Weight (kg)'}

TIMESERIES_VARS = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
    'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT',
    'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP',
    'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC',
    'Weight_VAR' # Derived feature: Weight as a time-series variable
]

ID_TIME_COLS = ["RecordID", "Time"]

COLS_TO_DROP_FROM_FEATURES = ["RecordID", "Time", "ICUType"]

STATIC_FEATURES_FOR_MODEL = ["Age", "Gender", "Height", "Weight"] # Excludes ICUType

# Plausibility Bounds (Grouped for clarity)
PLAUSIBILITY_BOUNDS = {
    "Height": (100, 250),
    "Weight": (20, 300), # Added upper bound assumption
    "Weight_VAR": (20, 300),
    "HR": (10, 300), # Added upper bound assumption
    "pH": (6.0, 8.0),
    "Temp": (12, 45),
    "Value_Not_Negative_Or_Zero": ["DiasABP", "MAP", "NIDiasABP", "NIMAP", "NISysABP", "SysABP", "RespRate", "PaO2", "Platelets", "WBC"]
    # Add others if needed
}

def safe_mkdirs(path):
    """Creates directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_data_in_df(set_path):

    try:
        os.path.exists(set_path)
    except FileNotFoundError:
        print(f"Path {set_path} does not exist.")
    file_names = [f for f in os.listdir(set_path) if f.endswith(".txt")]

    df_list = []

    # load the data into stacked dataframe
    for file in file_names:
        file_path = os.path.join(set_path, file)
        temp_df = pd.read_csv(file_path, sep=",", names=["Time", "Parameter", "Value"], header=0)
        
        # Extract the record ID (assuming exactly one 'RecordID' row)
        record_id = temp_df.loc[temp_df["Parameter"] == "RecordID", "Value"].iloc[0]
        
        # Attach that RecordID to every row
        temp_df["RecordID"] = record_id
        
        df_list.append(temp_df)

    full_df = pd.concat(df_list, ignore_index=True)
    return full_df

def aggregate_duplicates(df):
    """Some records have double entries for the same time point. 
    This function aggregates them in order to have a single value for each time point and allow pivoting the data.
    Aggregation follows special rules based on the observed data.

    Args:
        df (_type_): dataframe with columns ['RecordID', 'Parameter', 'Time', 'Value']

    Raises:
        ValueError: If there are more than 2 values for the same key raises error as it is unexpected to have 3 simultaneous values for the same time point (except for urine).

    Returns:
        df (Dataframe): dataframe with columns ['RecordID', 'Parameter', 'Time', 'Value'] with aggregated values for duplicate rows.
    """
    aggregated_rows = []

    for key, values in tqdm(df.groupby(['RecordID', 'Parameter', 'Time'])['Value'], desc="Aggregating duplicates"):
        values = values.reset_index(drop=True)
        record_id, param, time = (key)

        if values.shape[0] < 2:  # no duplicate measurements
            aggregated_rows.append({
                'RecordID': record_id,
                'Parameter': param,
                'Time': time,
                'Value': values[0]
                })
            
        elif values.shape[0] == 2: # exactly 2 duplicate measurements

            val1, val2 = values
            val_range = val2 - val1

            if param == "Urine":
                agg_value = values.sum()
            elif param == "Temp":
                agg_value = values.max() # all of the duplicates had a min which was close to 0, discard them
            else:
                agg_value = values.mean()

            aggregated_rows.append({
                'RecordID': record_id,
                'Parameter': param,
                'Time': time,
                'Value': agg_value
            })
        elif param == "Urine" :  # some urine samples have 3 recorded values at a time point
            agg_value = values.sum()
            aggregated_rows.append({
                'RecordID': record_id,
                'Parameter': param,
                'Time': time,
                'Value': agg_value
            })
        elif param == "HCT":
            agg_value = values.mean()
            aggregated_rows.append({
                'RecordID': record_id,
                'Parameter': param,
                'Time': time,
                'Value': agg_value
            })

        else:
            print(f"RecordID: {record_id}, Parameter: {param}, Time: {time}")
            raise ValueError("More than 2 values for the same key. Unexpected result.")
    
    return pd.DataFrame(aggregated_rows)

def pivot_data(df):
    """Pivot the dataframe to have a time-grid with each parameter as a column.

    Args:
        df (Dataframe): aggregated dataframe with columns ['RecordID', 'Parameter', 'Time', 'Value']

    Returns:
        Dataframe: pivoted dataframe with columns ['RecordID', 'Time', 'Parameter1', 'Parameter2', ...]
    """

    pivot_df = df.pivot(index=["RecordID","Time"], columns="Parameter", values="Value")
    if "RecordID" in pivot_df.columns:
        pivot_df = pivot_df.drop("RecordID", axis=1)
    
    pivot_df["Weight_VAR"] = pivot_df["Weight"]
    pivot_df = pivot_df.reset_index()
        
    return pivot_df

def propagate_static_vars(df):
    df[STATIC_VARS] = df.groupby('RecordID')[STATIC_VARS].transform('first') # fill the static variable rows for each patient
    df  = df[ID_TIME_COLS+STATIC_VARS + TIMESERIES_VARS] # reaarange the columns
    return df

def process_static_vars(df):
    
    for c in df[STATIC_VARS]:
        if c == "Age":
            pass # youngest age is 15, oldest is 90 so are within a reasonable range
            
        if c == "Gender":
            idx = (df[c] < 0) | (pd.isna(df[c]))
            df.loc[idx, c] = 1  # assume missing genders are male, was only a couple
        elif c == "Height":
            idx = (df[c] < 100) | (df[c] > 250)
            df.loc[idx, c] = np.nan

        elif c == "Weight":
            idx = (df[c] < 20) 
            df.loc[idx, c] = np.nan

    return df

def process_timeseries_vars(df):
    
    for c in df[TIMESERIES_VARS]:
        if c == "HR": # heart rate
            idx = df[c] < 10
            df.loc[idx, c] = np.nan

        elif c in ["DiasABP", "MAP", "NIDiasABP", "NIMAP", "NISysABP", "SysABP", "RespRate", "PaO2"]: 
            idx = df[c] <= 0
            df.loc[idx, c] = np.nan

        elif c == "Weight_VAR":
            idx = (df[c] < 20) 
            df.loc[idx, c] = np.nan
            
        elif c == "pH":
            idx = (df[c] < 6) | (df[c] > 8)
            df.loc[idx, c] = np.nan
        
        elif c == "Temp":
            idx = (df[c] < 12) | (df[c] > 45)
            df.loc[idx, c] = np.nan
    return df

def convert_time_to_minutes(df):
    df['Time'] = df['Time'].map(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1])) # convert time to minutes
    df['Time'] = df['Time'].astype(int)

    return df
def convert_time(df):
    """
    Convert the time to hours, rounded up to nearest hour and fill in dataframe with missing hours"
    """
    df['Time'] = df['Time'].map(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1])) # convert time to minutes
    df["Time"] = np.ceil(df["Time"] / 60) # round up to the nearest hour

    df['Time'] = df['Time'].astype(int)

    return df

def aggregate_hourly_measurements(df, variables):
    agg_dict = {col: 'mean' for col in variables if col != 'Urine'}
    agg_dict['Urine'] = 'sum'
    df = df.groupby(['RecordID', 'Time'], as_index=False).agg(agg_dict)

    # Get unique RecordIDs
    record_ids = df['RecordID'].unique()

    # Create a full MultiIndex for every RecordID and every hour from 0 to 48
    full_index = pd.MultiIndex.from_product([record_ids, range(49)], names=['RecordID', 'Time'])

    # Set the index of the DataFrame to RecordID and Time
    df = df.set_index(['RecordID', 'Time'])

    # Reindex the DataFrame using the complete index
    df_complete = df.reindex(full_index)

    # If desired, reset the index to turn RecordID and Time back into columns
    df_complete = df_complete.reset_index()

    return df_complete

def remove_outliers(df):
    # Remove outliers based on the 1st and 99th percentiles
    df.loc[(df["Parameter"]=="Gender") & (df["Value"] <0), "Value"] = 1

    df.loc[(df["Parameter"]=="RecordID"), "Value"] = np.nan
    df.loc[(df["Parameter"]=="Height") & ((df["Value"] <100)| (df["Value"]> 250)), "Value"] = np.nan

    df.loc[(df["Parameter"]=="Weight") & (df["Value"] <20), "Value"] = np.nan

    df.loc[(df["Parameter"]=="HR") & (df["Value"] <10), "Value"] = np.nan

    df.loc[(df["Parameter"]=="pH") & ((df["Value"] <6)| (df["Value"] >8)), "Value"] = np.nan

    df.loc[(df["Parameter"]=="Temp") & ((df["Value"] <12)| (df["Value"] >45)), "Value"] = np.nan

    cols_to_check = ["DiasABP", "MAP", "NIDiasABP", "NIMAP", "NISysABP", "SysABP", "RespRate", "PaO2"]

    df.loc[(df['Parameter'].isin(cols_to_check)) & (df['Value'] <= 0), 'Value'] = np.nan

    df.loc[(df['Parameter']=="ICUType"), 'Value'] = np.nan
    df["RecordID"] = df["RecordID"].astype(int)

    return df

def process_to_time_grid(df):
    df = pivot_data(df)
    print(f"Pivoted DataFrame {df.shape}:")
    print(df, "\n")

    df = convert_time(df)
    df = aggregate_hourly_measurements(df, STATIC_VARS+TIMESERIES_VARS)
    df = propagate_static_vars(df)

    # if set_path != "data/set-c":  # process outliers therefore skip for test set
    df = process_static_vars(df)

    df = process_timeseries_vars(df)
    return df

def process_to_time_tuple(df):

    df = remove_outliers(df)
    df = convert_time_to_minutes(df)

    df = df.dropna(subset=["Value"])
    print(f"size of df after removing NaN and outliers: {df.shape} \n")

    # sort values by time and patient_id
    df = df.sort_values(by=["RecordID", "Time"])
    df = df.reset_index(drop=True)
    return df


    
def main(data_paths = ["data/set-a", "data/set-b", "data/set-c"], format = "time_grid"):
    # Load the data

    set_paths = data_paths
    
    for set_path in set_paths:

        set_name = set_path.split("/")[-1]
        print(f"Processing data in {set_path}")

        df = load_data_in_df(set_path)
        print("Original DataFrame:")
        print(df,"\n")

    
        df = aggregate_duplicates(df)
        print("Aggregated DataFrame:")
        print(df,"\n")


        if format == "time_grid":
            df = process_to_time_grid(df)
            print("Final df shape: ", df.shape, "\n")
            print(df, "\n")
            print(f"Saving the processed data to data/time_grid_processed_raw_sparse_data_{set_name}.parquet \n")
            df.to_parquet(f"data/time_grid_processed_raw_sparse_data_{set_name}.parquet", engine="pyarrow")

            
        elif format == "time_tuple":

            print(f"size of {set_name}: {df.shape}")
            df = process_to_time_tuple(df)
            print(f"Saving the processed data to data/tuple_processed_raw_{set_name}.parquet \n")
            df.to_parquet(f"data/tuple_processed_raw_{set_name}.parquet", index=False)



if __name__ == "__main__":
    
    format = "time_grid" 
    # format = "time_tuple"
    data_paths = ["data/set-a", "data/set-b", "data/set-c"] # path to patient records data
    main(data_paths=data_paths, format= format)