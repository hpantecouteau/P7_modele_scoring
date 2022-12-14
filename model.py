from pathlib import Path
import pickle
from typing import Dict, List
import click
import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score, cross_validate, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier
import yaml
import shap
warnings.simplefilter(action='ignore', category=FutureWarning)

OUTPUT_FILE: str = "full_raw_dataset.csv"
MAPPING_MODELS: Dict = {
        'logistic': LogisticRegression(dual=False, max_iter=200),
        'svm': LinearSVC(dual = False),
        'kernel_svm': SVC(kernel="rbf", probability=True),
        'forest': RandomForestClassifier(criterion='gini', n_estimators=400, min_samples_leaf=10, max_features=0.33, min_samples_split=100),
        'lightGBM': LGBMClassifier(objective='binary', n_estimators=400)
    }

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('./data/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('./data/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df, cat_cols


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('./data/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('./data/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg, bb_cat+bureau_cat


# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('./data/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg, cat_cols


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('./data/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg, cat_cols
    

# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('./data/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg, cat_cols


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('./data/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg, cat_cols

def select_features(data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    return data[features]

def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna(how="any", axis="index")
    return data

def compute_params_grid(estimator_str: str, params: Dict) -> Dict[str, List]:
    if list(params[estimator_str].keys())[0] == 0:
        return None
    else:
        params_grid = {}
        for _param, _dict_param in params[estimator_str].items():
            params_grid[f"estimator__{_param}"] = np.arange(start=_dict_param["min_value"], stop=_dict_param["max_value"], step=_dict_param["step"])
        return params_grid


def train(data: pd.DataFrame, estimator_str: str, cv_on: bool = False):
    features = [_col for _col in data.columns if _col != "TARGET" and _col != "SK_ID_CURR"]
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(data[features+["SK_ID_CURR"]], data.TARGET, test_size=0.2, stratify=data.TARGET, random_state=0)
    X_train = df_X_train.drop(columns=["SK_ID_CURR"]).values
    X_test = df_X_test.drop(columns=["SK_ID_CURR"]).values
    y_train = df_y_train.drop(columns=["SK_ID_CURR"]).values
    y_test = df_y_test.drop(columns=["SK_ID_CURR"]).values
    scaling = StandardScaler()
    over = SMOTE(sampling_strategy=0.20, k_neighbors=5)
    under = RandomUnderSampler(sampling_strategy=0.50)
    estimator = MAPPING_MODELS[estimator_str]
    pipeline = Pipeline(steps=[('scaling', scaling), ('over', over), ('under', under), ('estimator', estimator)])    
    if cv_on:
        dummy_model = DummyClassifier(strategy='most_frequent')
        pipeline_dummy = Pipeline(steps=[('scaling', scaling), ('over', over), ('under', under), ('dummy_model', dummy_model)])
        pipeline_dummy.fit(X_train, y_train)
        dummy_y_pred = pipeline_dummy.predict(X_test)
        print(f"Test with dummy classifier: {roc_auc_score(y_test, dummy_y_pred)}")            
        print("Cross validation...") 
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
        scores = cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=cv, verbose=1)
        return scores                  
    print("Training...")
    pipeline.fit(X_train, y_train)
    print("Predicting...")         
    y_classes = pipeline.predict(X_test)
    print("Classes : ")
    print(pd.DataFrame(pipeline.predict(X_test)).value_counts()) 
    print(f"ROC AUC Test Score for {estimator_str}: {roc_auc_score(y_test, y_classes)}")
    print("Recording model in a pickle...")
    pickle.dump(estimator, open("model.pickle", "wb"))


def validate(data: pd.DataFrame, estimator_str: str):
    scores = train(data, estimator_str, cv_on=True)
    print(scores)
    print(f"CV - ROC AUC train score = {np.mean(scores):.3f} (std {np.std(scores):.3f})") 

def predict(data: pd.DataFrame, trained_model) -> pd.DataFrame:
    X = data[[_col for _col in data.columns if _col != "TARGET" and _col != "SK_ID_CURR"]].values
    y = data.TARGET.values
    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(X)
    y_probas_df = pd.DataFrame(trained_model.predict_proba(X_std)).rename(columns={0:'approved', 1:"rejected"})
    y_probas_df['target'] = y
    y_probas_df["SK_ID_CURR"] = data.SK_ID_CURR
    return y_probas_df

def interpet(data: pd.DataFrame, trained_model, n_samples: int = None):   
    features = [_col for _col in data.columns if _col != "TARGET" and _col != "SK_ID_CURR"]
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(data[features+["SK_ID_CURR"]], data.TARGET, test_size=0.2, stratify=data.TARGET, random_state=0)
    if n_samples is not None:
        df_X_train = df_X_train.sample(n_samples, random_state=0)
    print(df_X_train["SK_ID_CURR"])  
    X_train = df_X_train.drop(columns=["SK_ID_CURR"]).values
    scaler = StandardScaler()     
    X_train_std = scaler.fit_transform(X_train)     
    explainer = shap.KernelExplainer(model=trained_model.predict_proba, data=X_train_std)
    print(f"SHAP expected value : {explainer.expected_value}")
    print(f"Model mean value : {trained_model.predict_proba(X_train).mean()}")
    print(f"Recording SHAP Explainer in a pickle...")
    pickle.dump(explainer, open("explainer.pickle", "wb"))
    print("Computing all SHAP values...")
    shap_values_all_classes = explainer.shap_values(X_train_std)
    shap_values_accepted = shap_values_all_classes[0]
    df_shap = pd.DataFrame(shap_values_accepted, columns=features)
    df_shap["SK_ID_CURR"] = df_X_train["SK_ID_CURR"].values
    print(df_shap["SK_ID_CURR"])
    print(f"Recording SHAP values in a CSV...")
    df_shap.to_csv("df_shap.csv", index=False)
    
@click.command()
@click.option('--source',
    help='The source for data as CSV file',
    required=False,
    type=str
)
@click.option('--model',
    help='The model to apply on data source',
    required=False,
    type=str
)
@click.option("--debug", is_flag=True, show_default=True, default=False, help="Debug.")
@click.option("--cv", is_flag=True, show_default=True, default=False, help="Cross Validation ON.")
@click.option("--load", required=False, type=str, help="Load trained model")
@click.option("--interpretation", required=False, type=str, help="Interpret model output for a specific trained model.")
def main(cv: bool, debug = False, source: str= None, model: str = None, load: str = None, interpretation: str = None):
    if source is None:
        num_rows = 1000 if debug else None
        all_cat_cols = []
        df, cat_cols = application_train_test(num_rows)
        all_cat_cols.extend(cat_cols)
        with timer("Process bureau and bureau_balance"):
            bureau, cat_cols = bureau_and_balance(num_rows)
            all_cat_cols.extend(cat_cols)
            print("Bureau df shape:", bureau.shape)
            df = df.join(bureau, how='left', on='SK_ID_CURR')
            del bureau
            gc.collect()
        with timer("Process previous_applications"):
            prev, cat_cols = previous_applications(num_rows)
            all_cat_cols.extend(cat_cols)
            print("Previous applications df shape:", prev.shape)
            df = df.join(prev, how='left', on='SK_ID_CURR')
            del prev
            gc.collect()
        with timer("Process POS-CASH balance"):
            pos, cat_cols = pos_cash(num_rows)
            all_cat_cols.extend(cat_cols)
            print("Pos-cash balance df shape:", pos.shape)
            df = df.join(pos, how='left', on='SK_ID_CURR')
            del pos
            gc.collect()
        with timer("Process installments payments"):
            ins, cat_cols = installments_payments(num_rows)
            all_cat_cols.extend(cat_cols)
            print("Installments payments df shape:", ins.shape)
            df = df.join(ins, how='left', on='SK_ID_CURR')
            del ins
            gc.collect()
        with timer("Process credit card balance"):
            cc, cat_cols = credit_card_balance(num_rows)
            all_cat_cols.extend(cat_cols)
            print("Credit card balance df shape:", cc.shape)
            df = df.join(cc, how='left', on='SK_ID_CURR')
            del cc
            gc.collect()
        json.dump(all_cat_cols, open("categ_features.json","w"))
        print(f"Writing dataset in {OUTPUT_FILE}.")
        df.to_csv(OUTPUT_FILE, index=False)
        print("Done.")
    else:
        print(f"Reading {source}...")
        data = pd.read_csv(source)        
        print(data.TARGET.value_counts(normalize=True))
        data = clean_dataset(data)
        print(data.head())        
        print(data.shape)
        if load is not None:
            trained_model = pickle.load(open(load, "rb"))
            df_probas = predict(data, trained_model)
            print("Recording probabilities in a CSV...")
            df_probas.to_csv("df_probas.csv", index=False)
        if model is not None:
            print(f"Modelizing with {model}...")
            train(data, model, cv_on=cv)
        if interpretation is not None:                     
            print("Interpeting model outputs...")
            trained_model = pickle.load(open(interpretation, "rb"))
            interpet(data, trained_model, 100)
        print("Done.")

if __name__ == "__main__":
    with timer("Full model run"):
        main()