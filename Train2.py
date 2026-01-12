import pandas as pd
from sklearn.model_selection import train_test_split
# Preparation: Standardized Preprocessing

# 1. Load and Clean Target
df = pd.read_csv('FINAL_COLON_CANCER_DATASET.csv')
df = df[~df['diagnoses.ajcc_pathologic_stage'].isin(['Unknown', '0', 'Not Reported'])].copy()
df['target'] = df['diagnoses.ajcc_pathologic_stage'].apply(lambda x: 1 if any(s in str(x) for s in ['III', 'IV']) else 0)

# 2. Basic Feature Selection & Encoding
cols_to_drop = ['cases.case_id', 'target', 'diagnoses.ajcc_pathologic_stage', 'diagnoses.ajcc_clinical_stage', 
                'diagnoses.ajcc_pathologic_t', 'diagnoses.ajcc_pathologic_n', 'diagnoses.ajcc_pathologic_m']

X = df.drop(columns=cols_to_drop)
y = df['target']

# Convert treatments and all text to numbers
treatments = X['treatments.treatment_type'].str.get_dummies(sep='; ')
X = pd.concat([X.drop(columns=['treatments.treatment_type']), treatments], axis=1)
X = pd.get_dummies(X)

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Models
# Model 1: Logistic Regression (The Industry Baseline)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Initialize and Train
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Predict
log_preds = log_model.predict(X_test)

print("--- LOGISTIC REGRESSION RESULTS ---")
print(f"Accuracy: {accuracy_score(y_test, log_preds):.2%}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, log_preds))




# Model 2: Decision Tree (The "Human-Readable" Model)
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Initialize and Train (we limit depth so the tree isn't too messy)
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

# Predict
tree_preds = tree_model.predict(X_test)

print("\n--- DECISION TREE RESULTS ---")
print(f"Accuracy: {accuracy_score(y_test, tree_preds):.2%}")

# VISUALIZE THE TREE (Great for your thesis!)
plt.figure(figsize=(20,10))
plot_tree(tree_model, feature_names=X.columns, class_names=['Early', 'Advanced'], filled=True, rounded=True)
plt.title("Decision Tree Flowchart for Colon Cancer Staging")
plt.show()



