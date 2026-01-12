import pandas as pd
import numpy as np

# 1. Load your file
df = pd.read_excel('clinicopathological_data.xlsx')

# 2. Define our ID column
ID_COL = 'cases.case_id'

# 3. Clean '0' values
# Many of your columns have '0' where they should have data. 
# We replace 0 with NaN so they don't mess up our "max" or "last" logic.
df = df.replace(['0', 0, 'Not Reported', 'not reported', 'Unknown'], np.nan)

# 4. Define specific logic for different types of columns
agg_logic = {}

for col in df.columns:
    if col == ID_COL:
        continue
    
    # CASE A: Treatment and Regimen columns (The "Unique Features" you don't want to lose)
    if 'treatments' in col or 'treatment_type' in col:
        # Join all unique treatments with a semicolon (e.g. "Chemotherapy; Surgery")
        agg_logic[col] = lambda x: "; ".join(set(x.dropna().astype(str)))
    
    # CASE B: Numerical columns (Age, days_to_last_follow_up)
    elif df[col].dtype in ['int64', 'float64']:
        # Take the maximum (this avoids picking up '0' if a real value exists)
        agg_logic[col] = 'max'
        
    # CASE C: Categorical columns (Gender, Race, Stage, Site)
    else:
        # Take the 'last' occurrence (usually the most updated medical record)
        agg_logic[col] = 'last'

# 5. Execute the Flattening
df_unique = df.groupby(ID_COL).agg(agg_logic).reset_index()

# 6. Final cleanup: Fill remaining NaNs with "Unknown" for the ML model
df_unique = df_unique.fillna("Unknown")

# 7. Quality Check
print(f"Original Row Count: {len(df)}")
print(f"Unique Patient Count: {len(df_unique)}")

# Display an example of a patient who had multiple treatments
# (This helps you verify that the 'treatments.treatment_type' column now contains multiple values)
print("\nExample of aggregated treatments for one patient:")
print(df_unique[[ID_COL, 'treatments.treatment_type']].head())

# 8. Save the organized file
df_unique.to_csv('organized_clinical_data.csv', index=False)
print("\nFile 'organized_clinical_data.csv' created successfully!")