
import shap
import json
import pickle
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


MAPPING_PARAMS = {
    'lightGBM': {
        'num_leaves': [10]
    }
}

df_input = pd.read_csv("input.csv")
print(f"df_input.shape : {df_input.shape}")

features = [_col for _col in df_input.columns if _col != "TARGET" and _col != "SK_ID_CURR"]
df = df_input.dropna(how="any", axis="index", subset=features+["TARGET"])
print(df.TARGET.value_counts(normalize=True))

X = df[features].values
y = df.TARGET.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
print(f"X_train.shape : {X_train.shape}")
print(f"X_test.shape : {X_test.shape}")
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)    
X_test_std = scaler.transform(X_test)  

over = SMOTE(sampling_strategy=0.20)    
estimator = LGBMClassifier(objective='binary', n_estimators=100)
pipeline = Pipeline(steps=[('over', over), ('estimator', estimator)], verbose=True)
pipeline.fit(X_train_std, y_train)

# print("Cross validation...") 
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
# params = {f"estimator__{hyperparam}": values for hyperparam, values in MAPPING_PARAMS['lightGBM'].items()}
# params.update({"over__k_neighbors": [5]})
# grid = GridSearchCV(pipeline, params, cv=cv, scoring="roc_auc", verbose=2)
# grid.fit(X_train_std, y_train)
# print(f"{grid.best_params_} : ROC AUC = {grid.best_score_}")
explainer = shap.TreeExplainer(model=estimator, model_output="probability", data=X_train_std, feature_names=features)
shap_values = explainer.shap_values(X_train_std)
df_shap = pd.DataFrame(shap_values, columns=features)
print(df_shap.head())
# trained_model = pickle.load(open("model.pickle", "r"))
# y_classes = trained_model.predict(X_test_std)
# print("Classes : ")
# print(pd.DataFrame(y_classes).value_counts()) 
# print(f"ROC AUC Test Score for lightGB: {roc_auc_score(y_test, y_classes)}")

# X = df[features].values
# y = df.TARGET.values
# X_std = scaler.transform(X)
# y_probas_df = pd.DataFrame(trained_model.predict_proba(X_std))
# y_probas_df["target"] = y
# print(y_probas_df.head())
# print(y_probas_df.shape)
# print(y_probas_df.describe())
# y_probas_df.to_csv("test_probas.csv")