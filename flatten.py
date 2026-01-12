import pandas as pd

# Load the datasets
df_clinical = pd.read_excel('clinicopathological_data.xlsx')
df_exposure = pd.read_excel('exposure.xlsx')
df_family = pd.read_excel('family_history.xlsx')

# SET THE ID NAME ONCE HERE TO AVOID TYPOS
ID_COL = 'cases.case_id'

# --- STEP 1: Flatten Clinicopathological Data ---
agg_logic = {}
for col in df_clinical.columns:
    if col == ID_COL:
        continue
    elif df_clinical[col].dtype == 'object': # Text data (Gender, Stage, etc.)
        agg_logic[col] = 'last' 
    else: # Numerical data (Age, BMI, etc.)
        agg_logic[col] = 'mean'

# Create the unique patient dataframe - Using the ID_COL variable
df_clinical_unique = df_clinical.groupby(ID_COL).agg(agg_logic).reset_index()

print(f"Original clinical rows: {len(df_clinical)}")
print(f"Unique patient rows: {len(df_clinical_unique)}")

# --- STEP 2: Ensure IDs are unique in other files ---
df_exposure_unique = df_exposure.drop_duplicates(subset=[ID_COL])
df_family_unique = df_family.drop_duplicates(subset=[ID_COL])

# --- STEP 3: Merge all files ---
# We merge everything onto the clinical data using the correct ID column
final_df = pd.merge(df_clinical_unique, df_exposure_unique, on=ID_COL, how='left')
final_df = pd.merge(final_df, df_family_unique, on=ID_COL, how='left')

# --- STEP 4: Handle Missing Values ---
# If a patient exists in Clinical but not in Exposure, fill with "Unknown"
final_df = final_df.fillna("Unknown") 

# Save the master dataset
final_df.to_csv('master_colon_cancer_data.csv', index=False)

print("--- SUMMARY ---")
print(f"Final columns: {list(final_df.columns)}")
print(f"Final dataset shape: {final_df.shape}")
print("Master dataset created successfully!")