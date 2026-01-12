import pandas as pd
import numpy as np

# 1. Load your file - Force morphology to be a string to avoid the Date bug
df = pd.read_excel('clinicopathological_data.xlsx', dtype={'diagnoses.morphology': str})

ID_COL = 'cases.case_id'

# 2. Expanded list of "bad" values to skip
placeholders = ['0', 0, '0.0', 'Not Reported', 'not reported', 'Unknown', 'not hispanic or latino', 'Not reported']

def get_best_value(series):
    # Convert everything to string for comparison
    str_series = series.astype(str).str.strip()
    valid_values = series[~str_series.isin(placeholders)].dropna()
    
    if not valid_values.empty:
        return valid_values.mode()[0]
    else:
        # If no real data, return the first placeholder found instead of "Unknown"
        return series.iloc[0]

agg_logic = {}
for col in df.columns:
    if col == ID_COL: continue
    
    if 'treatments' in col or 'treatment_type' in col:
        agg_logic[col] = lambda x: "; ".join(sorted(set(x.astype(str).replace(placeholders, np.nan).dropna())))
    else:
        agg_logic[col] = get_best_value

# 3. Flatten
df_unique = df.groupby(ID_COL).agg(agg_logic).reset_index()

# 4. Fix the Morphology column specifically
# If it still looks like a date, we strip the time part
if 'diagnoses.morphology' in df_unique.columns:
    df_unique['diagnoses.morphology'] = df_unique['diagnoses.morphology'].astype(str).str.replace('-01 00:00:00', '/3')

# 5. Save
df_unique.to_csv('organized_clinical_data_v2.csv', index=False)
print("V2 Created! Check the Morphology column and 'Unknown' counts.")