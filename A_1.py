import numpy as np 
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm 
import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope 
from hyperopt.pyll.stochastic import sample
import pickle
from sklearn.metrics import confusion_matrix


##importing dataset
data = pd.read_csv("Data/globalterrorismdb_0522dist.csv", encoding='ISO-8859-1', low_memory=False
)
##getting an overview of the dataset

print(data.head(22))
print(data.columns)

##subsetting data for the year 2020
data_2020 = data[data['iyear'] == 2020]

## testing to ensure the subset worked 
print(data_2020.head())

## meaningful verification: check if we have any data
print(f"Number of records in 2020: {len(data_2020)}")

##variable analysis:
print(data_2020["targtype1"].describe()) ## 14
print(data_2020["country"].describe()) ## 155
print(data_2020["crit1"].describe()) ## 1
print(data_2020["crit2"].describe()) ## 1
print(data_2020["crit3"].describe()) ## 1
print(data_2020["region"].describe()) ## 10
print(data_2020["doubtterr"].describe()) ## 1
print(data_2020["vicinity"].describe()) ## 0  
print(data_2020["attacktype1"].describe()) ## 6 
print(data_2020["success"].describe())  ##1
print(data_2020["weaptype1"].describe()) ##8


## plotting of variable analysis (target type):
rel_freq = data_2020["targtype1_txt"].value_counts(normalize=True)
print(rel_freq)
rel_freq.plot(kind="barh", color = "steelblue") # 'barh' makes it horizontal
plt.xlabel("Relative Frequency") # Swapped labeling
plt.ylabel("Target Type")
plt.title("Relative Frequency of Terrorist Attack Target Types (2020)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.grid(True)
plt.show()


##one hot encoding:
cols_to_encode = ['attacktype1_txt', 'weaptype1_txt', 'targtype1_txt', "region_txt", "country_txt"]
prefixes = ['AttackType', 'Weapon', "Target", "Region", "Country"] ##renaming new columns 
# Perform One-Hot Encoding on ALL of them at once
data_2020_encoded = pd.get_dummies(data_2020, columns=cols_to_encode, prefix=prefixes)
# Convert all the new boolean (True/False) columns to float (1.0/0.0)
# We select any column that starts with one of our prefixes
new_columns = [col for col in data_2020_encoded.columns if any(col.startswith(p + "_") for p in prefixes)]
data_2020_encoded[new_columns] = data_2020_encoded[new_columns].astype('float64')
##printing to confirm it works and getting shape of data
print(f"Total columns after encoding: {len(data_2020_encoded.columns)}")
print(f"New encoded columns count: {len(new_columns)}")
print(data_2020_encoded[new_columns].head())

##cleaning column names - removing spaces and symbols so they can be used as features
data_2020_encoded.columns = (data_2020_encoded.columns
    .str.replace(' & ', '_', regex=False)
    .str.replace(' ', '_', regex=False)
    .str.replace('/', '_', regex=False)
    .str.replace('-', '_', regex=False)
    .str.replace('(', '', regex=False)
    .str.replace(')', '', regex=False)
)

# Verify the new names - tests the code above
print([c for c in data_2020_encoded.columns if 'Target' in c][:5])


##Random Forests:
##splitting data into training and testing sets (80/20 split)
df_train, df_test = train_test_split(data_2020_encoded, test_size =0.2, random_state=42)
print(f"Training set: {df_train.shape}")
print(f"Testing set: {df_test.shape}")

##defining features and target
features = ["Region_Western_Europe", 
"Region_Middle_East_North_Africa", 
"Region_South_Asia", "Region_Southeast_Asia", 
"Region_Sub_Saharan_Africa", "Weapon_Incendiary", 
"Weapon_Biological", "Weapon_Chemical", "Weapon_Explosives", 
"Weapon_Firearms", "Weapon_Fake_Weapons", "AttackType_Hostage_Taking_Kidnapping",
"AttackType_Hijacking", "AttackType_Hostage_Taking_Barricade_Incident", 
"AttackType_Unarmed_Assault",
"AttackType_Bombing_Explosion", "AttackType_Armed_Assault", 
"AttackType_Assassination", "vicinity", "success", "doubtterr", "crit2", "crit3", "crit1"]
y = "Target_Private_Citizens_Property"

print(f"Feature: {features}")
print(f"Target: {y}")

##fitting the model
clf = RandomForestClassifier(n_estimators = 100, criterion="entropy", 
random_state= 42)
clf.fit(df_train[features], df_train[y])
print(f"Number of trees: {clf.n_estimators}")
print(f"Number of features: {len(features)}")

##important features:
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
sorted_features = [features[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

##plotting important features 
plt.figure(figsize=(10,6))
plt.barh(sorted_features, sorted_importances, color = "lightgreen")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forests Important Features")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()

##model performance 
y_pred = clf.predict(df_test[features])

accuracy = accuracy_score(df_test[y], y_pred)
print(f"Accuracy: {accuracy:.4f}") ##4 decimal places

precision = precision_score(df_test[y], y_pred)
print(f"Precision: {precision:.4f}")

recall = recall_score(df_test[y], y_pred)
print(f"Recall: {recall:.4f}")

f1 = f1_score(df_test[y], y_pred)
print(f"F1 Score: {f1:.4f}")

##fitting a confusion matrix 
cm = confusion_matrix(df_test[y], y_pred)

# plotting confusion matrix in a heatmap 
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Citizens', 'Private Citizens'], 
            yticklabels=['Non-Citizens', 'Private Citizens'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Random Forest Confusion Matrix: Terrorist Attack Target Prediction')
plt.show()

print(f"True Negatives (correctly predicted non-citizens): {cm[0,0]}")
print(f"False Positives (predicted private citizens but were non-citizens): {cm[0,1]}")
print(f"False Negatives (predicted non-citizens but were private citizens): {cm[1,0]}")
print(f"True Positives (correctly predicted private citizens): {cm[1,1]}")

##XGBoost:

##splitting data into training and testing sets 
X_train, X_test = train_test_split(data_2020_encoded, test_size=0.2, random_state=42)
print(X_train.head())

##setting up XGBoost model
xgb_model = xgb.XGBClassifier(objective="binary:logistic", learning_rate = 0.5)
xgb_model.fit(X_train[features], X_train[y])

##plotting important features 
xgb.plot_importance(xgb_model, importance_type='weight', show_values = False)
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()

##performance:
X_pred = xgb_model.predict(X_test[features])
accuracy_XGB = accuracy_score(X_test[y], X_pred)
print(f"Accuracy: {accuracy_XGB:.4f}")
precision_XGB = precision_score(X_test[y], X_pred)
print(f"Precision is {precision_XGB}")
recall_XGB = recall_score(X_test[y], X_pred)
print(f"Recall is {recall_XGB}")
f1_XGB = f1_score(X_test[y], X_pred)
print(f"F1 score is {f1_XGB}")

##plotting comparison of Random Forst and XGBoost performance metrics 
metrics = ["Accuracy", "Precision", "Recall", "F1"]
rf_scores = [accuracy, precision, recall, f1]
xgb_scores = [accuracy_XGB, precision_XGB, recall_XGB, f1_XGB]
bar_width = 0.35 
x = np.arange(len(metrics)) 
bar1 = plt.bar(x - bar_width/2, rf_scores, bar_width, label = "Random Forest", color = "lightgreen")
bar2 = plt.bar(x + bar_width/2, xgb_scores, bar_width, label = "XGBoost", color = "skyblue")
plt.bar_label(bar1, fmt='%.3f', padding=3) ##adding the values to the bars
plt.bar_label(bar2, fmt='%.3f', padding=3)
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.xticks(x, metrics)
plt.legend(title = "Classification Models")
plt.tight_layout()
plt.show()

##confusion matrix for XGBoost
xgcm = confusion_matrix(X_test[y], X_pred)

##plotting confusion matrix in a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(xgcm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Citizens', 'Private Citizens'], 
            yticklabels=['Non-Citizens', 'Private Citizens'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(' XGBoostConfusion Matrix: Terrorist Attack Target Prediction')
plt.show()

print(f"True Negatives (correctly predicted non-citizens): {xgcm[0,0]}")
print(f"False Positives (predicted private citizens but were non-citizens): {xgcm[0,1]}")
print(f"False Negatives (predicted non-citizens but were private citizens): {xgcm[1,0]}")
print(f"True Positives (correctly predicted private citizens): {xgcm[1,1]}")

##bayesian optimisation:
##dictionary of hyperparameters to optimise:
space = {
    'max_depth': scope.int(hp.quniform('max_depth', 1, 15, 1)),  # Integer values
    'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 15, 1)),  # Integer values
    'learning_rate': hp.loguniform('learning_rate', -5, -2),  # Float values
    'subsample': hp.uniform('subsample', 0.5, 1),  # Float values
    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),  # Float values
    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 1))  # Integer values - quniform makes it into an integer- makes its countable 
}

##porbability ditributions:
def sample_hyperparameter(hyperparameter, n_samples):
    """
    Generate samples from a given hyperparameter distribution.

    Args:
        hyperparameter: A Hyperopt distribution (e.g., hp.uniform, hp.loguniform).
        n_samples: Number of samples to generate.

    Returns:
        A list of samples from the distribution.
    """
    return [sample(hyperparameter) for _ in range(n_samples)]

# Define the hyperparameter space
space = {
    'max_depth': hp.quniform('max_depth', 1, 15, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 15, 1),
    'learning_rate': hp.loguniform('learning_rate', -5, -2),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1)
}

# Number of samples to generate
n_samples = 10000

# Sample each hyperparameter using the function
samples = {
    'max_depth': [int(x) for x in sample_hyperparameter(space['max_depth'], n_samples)],
    'min_child_weight': [int(x) for x in sample_hyperparameter(space['min_child_weight'], n_samples)],
    'learning_rate': sample_hyperparameter(space['learning_rate'], n_samples),
    'subsample': sample_hyperparameter(space['subsample'], n_samples),
    'colsample_bytree': sample_hyperparameter(space['colsample_bytree'], n_samples),
    'n_estimators': [int(x) for x in sample_hyperparameter(space['n_estimators'], n_samples)],
}

##plotting the distributions
for param, values in samples.items():
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of {param}")
    plt.xlabel(param)
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

##objective function:
def objective(params): 
    params['max_depth'] = int(params['max_depth'])
    params['min_child_weight'] = int(params['min_child_weight'])
    params['n_estimators'] = int(params['n_estimators'])
    # defines parametres
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        **params
    )
    # fitting model to training set 
    xgb_model.fit(X_train[features], X_train[y])
    # making predictions 
    X_pred = xgb_model.predict(X_test[features])
    # calculate the F1 score 
    score = f1_score(X_test[y], X_pred) 
    return {'loss': -score, 'status': STATUS_OK}  # minises loss
trials = Trials()

# running the hyperopt optimisation
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)
print("Best set of hyperparameters: ", best_params)
for key in best_params:
    if key in ['max_depth', 'min_child_weight', 'n_estimators']:
        best_params[key] = int(best_params[key])
print("Best set of hyperparameters: ", best_params)


##fitting best parametres model after bayesian optimisation
xgb_model2 = xgb.XGBClassifier(objective='binary:logistic', **best_params)

xgb_model2.fit(X_train[features], X_train[y]) ##fitting model to training set 

y_pred2 = xgb_model2.predict(X_test[features]) ##making predictions 

precision2 = precision_score(X_test[y], y_pred2) ##calculating precision
print(f'Precision is {precision2}')

recall2 = recall_score(X_test[y], y_pred2) ##calculating recall
print(f'Recall is {recall2}')

f12 = f1_score(X_test[y], y_pred2) ##calculating F1 score
print(f'F1 score is {f12}')

##plotting graph before bayesian optimisation vs. after
metrics = ["Precision", "Recall", "F1"]
before = [precision_XGB, recall_XGB, f1_XGB]
after = [precision2, recall2, f12]
bar_width = 0.35
x = np.arange(len(metrics))
bar1 = plt.bar(x - bar_width/2, before, bar_width, label = "Before", color = "skyblue")
bar2 = plt.bar(x + bar_width/2, after, bar_width, label = "After", color = "darkblue")
plt.bar_label(bar1, fmt='%.3f', padding=3) # Add labels on top of the bars
plt.bar_label(bar2, fmt='%.3f', padding=3)
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.title("XGBoost Model Performance Comparison")
plt.xticks(x, metrics)
plt.legend(title = "XGBoost Models", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

##creating final model:
clf = xgb.XGBClassifier(objective='binary:logistic', **best_params)
cfit = clf.fit(data_2020_encoded[features], data_2020_encoded[y])
pickle.dump(cfit, open('xgb_model.pkl', 'wb')) ##saving the model
loaded_model = pickle.load(open("xgb_model.pkl", "rb"))

##model including all features:
new_obs = {"Region_Western_Europe":0,  
"Region_Middle_East_North_Africa":0, 
"Region_South_Asia":0, 
"Region_Southeast_Asia":0, 
"Region_Sub_Saharan_Africa":1, 

"Weapon_Incendiary":0, 
"Weapon_Biological":0, 
"Weapon_Chemical":0, 
"Weapon_Explosives":0, 
"Weapon_Firearms":1, 
"Weapon_Fake_Weapons":0, 

"AttackType_Hostage_Taking_Kidnapping":1,
"AttackType_Hijacking":0, 
"AttackType_Hostage_Taking_Barricade_Incident":0, 
"AttackType_Unarmed_Assault":0,
"AttackType_Bombing_Explosion":0, 
"AttackType_Armed_Assault":0, 
"AttackType_Assassination":0, 

"vicinity":1,
"success":1, 
"doubtterr":0, 
"crit1":1,
"crit2":0, 
"crit3":1}

##creating probability from new observations:
df_new_obs = pd.DataFrame([new_obs])
df_new_obs = df_new_obs[features]
prob = loaded_model.predict_proba(df_new_obs[features])
print(loaded_model.classes_)
print("Probability of attack: ", prob[0][1])

##for baseline reference:
print(data_2020_encoded[y].mean())

