import pandas as pd
import numpy as np
import os

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

TIMESERIES_VARS = list(FEATURES.keys())
RECORDID_and_TIME_vars = ["RecordID", "Time"]


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
    for key, values in df.groupby(['RecordID', 'Parameter', 'Time'])['Value']:
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

def pivot(df):
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
    df  = df[RECORDID_and_TIME_vars+STATIC_VARS + TIMESERIES_VARS] # reaarange the columns
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

def convert_time(df):
    df['Time'] = df['Time'].map(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1])) # convert time to minutes
    df["Time"] = np.ceil(df["Time"] / 60) # round up to the nearest hour
    return df

def aggregate_hourly_measurements(df, variables):
    agg_dict = {col: 'mean' for col in variables if col != 'Urine'}
    agg_dict['Urine'] = 'sum'
    merged_df = df.groupby(['RecordID', 'Time'], as_index=False).agg(agg_dict)
    return merged_df

def main():
    # Load the data

    set_paths = ["data/set-a", "data/set-b", "data/set-c"]

    for set_path in set_paths:
        print(f"Processing data in {set_path}")
        df = load_data_in_df(set_path)
        print("Original DataFrame:")
        print(df,"\n")

        print("Try pivoting the original DataFrame:")
        try:
            test = df.pivot(index="RecordID", columns="Parameter", values="Value")
            print(test)
        except ValueError as e:
            print("Dataframe contains duplicates with more than 2 rows.")
            print("Error:", e, "\n")

        print("Aggregating data...")
        df = aggregate_duplicates(df)
        print("Aggregated DataFrame:")
        print(df,"\n")

        df = pivot(df)
        print("Pivoted DataFrame:")
        print(df)

        df = propagate_static_vars(df)

        if set_path != "data/set-c":  # process outliers therefore skip for test set
            df = process_static_vars(df)

            df = process_timeseries_vars(df)

        df = convert_time(df)

        df = aggregate_hourly_measurements(df, STATIC_VARS+TIMESERIES_VARS)
        print("Saving the processed data...")

        set_name = set_path.split("/")[-1]
        df.to_parquet(f"data/processed_raw_data_{set_name}.parquet", engine="pyarrow")
if __name__ == "__main__":
    main()