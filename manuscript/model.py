import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import numpy as np

# Load dataset
file_path = '/Users/debang/Documents/NIJ_s_Recidivism_Challenge_Full_Dataset_20241001.csv'
data = pd.read_csv(file_path)

# Data cleaning and preparation
columns_to_drop = ['ID', 'Recidivism_Arrest_Year1', 'Recidivism_Arrest_Year2', 
                   'Recidivism_Arrest_Year3', 'Training_Sample']
data_cleaned = data.drop(columns=columns_to_drop)

# Convert categorical fields
categorical_columns = ['Gender', 'Race', 'Age_at_Release', 'Residence_PUMA', 
                       'Gang_Affiliated', 'Supervision_Level_First', 'Education_Level', 
                       'Dependents', 'Prison_Offense']
for col in categorical_columns:
    data_cleaned[col] = data_cleaned[col].astype('category')

# Fill missing values
for column in data_cleaned.select_dtypes(include=['float64', 'int64']).columns:
    data_cleaned[column] = data_cleaned[column].fillna(data_cleaned[column].mean())

for column in data_cleaned.select_dtypes(include=['category']).columns:
    data_cleaned[column] = data_cleaned[column].fillna(data_cleaned[column].mode()[0])

# Convert boolean columns to binary
data_cleaned = data_cleaned.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)

# Create dummy variables
data_cleaned = pd.get_dummies(data_cleaned, drop_first=True)

# Separate features and target variable
X = data_cleaned.drop('Recidivism_Within_3years', axis=1)
y = data_cleaned['Recidivism_Within_3years']

# Apply scaling to features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add interaction terms for logistic regression (will probably overfit)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interactions = poly.fit_transform(X_scaled)

# Monte Carlo Cross-Validation with 3 splits (too few, will increase later)
n_splits = 3
mc_split = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)

# Initialize
logistic_model = LogisticRegression(max_iter=2000, random_state=42)
lasso_model = Lasso(alpha=0.01, max_iter=1000, random_state=42)
random_forest_model = RandomForestClassifier(random_state=42)
results = {'Model': [], 'Mean Accuracy': [], 'Mean ROC AUC': [], 'FPR': [], 'FNR': []}

# Monte Carlo Cross-Validation
for model_name, model in zip(['Logistic Regression with Interactions', 'LASSO', 'Random Forest'],
                             [logistic_model, lasso_model, random_forest_model]):
    accuracies, aucs, fprs, fnrs = [], [], [], []
    
    for train_idx, test_idx in mc_split.split(X_interactions):
        # Split
        X_train_mc, X_test_mc = X_interactions[train_idx], X_interactions[test_idx]
        y_train_mc, y_test_mc = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train
        model.fit(X_train_mc, y_train_mc)
        
        # Predict
        if model_name == 'LASSO':
            y_pred = model.predict(X_test_mc).round()  # round predictions for binary
        else:
            y_pred = model.predict(X_test_mc)
        
        # Calculate Accuracy and AUC
        accuracies.append(accuracy_score(y_test_mc, y_pred))
        if model_name != 'LASSO':
            aucs.append(roc_auc_score(y_test_mc, model.predict_proba(X_test_mc)[:, 1]))
        else:
            aucs.append(roc_auc_score(y_test_mc, y_pred))
        
        # Calculate confusion matrix and FPR, FNR
        tn, fp, fn, tp = confusion_matrix(y_test_mc, y_pred).ravel()
        fprs.append(fp / (fp + tn))  # FPR
        fnrs.append(fn / (fn + tp))  # FNR
    
    # Store mean results
    results['Model'].append(model_name)
    results['Mean Accuracy'].append(np.mean(accuracies))
    results['Mean ROC AUC'].append(np.mean(aucs))
    results['FPR'].append(np.mean(fprs))
    results['FNR'].append(np.mean(fnrs))

# Display results
results_df = pd.DataFrame(results)
print(results_df)