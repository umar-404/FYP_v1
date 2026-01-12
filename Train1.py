import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Load the data
df = pd.read_csv('FINAL_COLON_CANCER_DATASET.csv')

# 2. CREATE THE TARGET (What we want to predict)
# We will predict "Advanced Stage" (Stage III/IV) vs "Early Stage" (Stage I/II)
# We filter out rows where stage is unknown to keep the model accurate
df = df[~df['diagnoses.ajcc_pathologic_stage'].isin(['Unknown', '0', 'Not Reported'])]

def categorize_stage(stage):
    stage = str(stage)
    if 'III' in stage or 'IV' in stage:
        return 1 # Advanced
    return 0 # Early

df['target_label'] = df['diagnoses.ajcc_pathologic_stage'].apply(categorize_stage)

# 3. SELECT FEATURES (X) and TARGET (y)
# We drop columns that are "cheating" (like the stage itself or follow-up days)
cols_to_drop = [
    'cases.case_id', 'target_label', 'diagnoses.ajcc_pathologic_stage', 
    'diagnoses.ajcc_clinical_stage', 'diagnoses.ajcc_pathologic_t', 
    'diagnoses.ajcc_pathologic_n', 'diagnoses.ajcc_pathologic_m',
    'diagnoses.days_to_last_follow_up', 'diagnoses.days_to_recurrence'
]

X = df.drop(columns=cols_to_drop)
y = df['target_label']

# 4. ENCODE TREATMENTS (Multi-label)
# This turns "Chemo; Surgery" into separate 0/1 columns
treatments = X['treatments.treatment_type'].str.get_dummies(sep='; ')
X = pd.concat([X.drop(columns=['treatments.treatment_type']), treatments], axis=1)

# 5. ENCODE ALL OTHER TEXT COLUMNS (One-Hot Encoding)
# This turns "Male/Female" or "White/Black" into numbers
X = pd.get_dummies(X)

# 6. SPLIT DATA (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. TRAIN THE MODEL
print("Training the Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. EVALUATE
y_pred = model.predict(X_test)
print("\n--- MODEL PERFORMANCE ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# 9. FEATURE IMPORTANCE (Which factors matter most?)
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n--- TOP 5 PREDICTORS FOR COLON CANCER STAGE ---")
print(importances.head(5))