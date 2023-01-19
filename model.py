from pathlib import Path
import pickle
from typing import Dict, List
from preprocessing import build_features, clean_dataset, timer
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
from imblearn.over_sampling import SMOTENC, SMOTE
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
        'forest': RandomForestClassifier(criterion='gini', n_estimators=400, min_samples_leaf=10, max_features=0.33),
        'lightGBM': LGBMClassifier(objective='binary', n_estimators=400)
    }
MAPPING_PARAMS: Dict = {
    'forest': {
        'min_samples_split': [2,10,20]
    },
    'lightGBM': {
        'num_leaves': [10]
    }
}

def modelize(data: pd.DataFrame, estimator_str: str, steps: List[str] = ["cv", "predict"], model = None):
    initial_cat_cols: List[str] = json.load(open("cat_cols.json", "r"))    
    features = [_col for _col in data.columns if _col != "TARGET" and _col != "SK_ID_CURR" and not _col in initial_cat_cols]
    # cat_cols = [col for col in initial_cat_cols if col in features]
    data = clean_dataset(data, features+["TARGET"])
    click.echo(data.TARGET.value_counts(normalize=True))
    ids = data.SK_ID_CURR.values
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(data[features+["SK_ID_CURR"]], data.TARGET, test_size=0.2, stratify=data.TARGET, random_state=0) 
    X_train = df_X_train.drop(columns=["SK_ID_CURR"]).values
    X_test = df_X_test.drop(columns=["SK_ID_CURR"]).values
    y_train = df_y_train.values
    y_test = df_y_test.values
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)    
    X_test_std = scaler.transform(X_test)  
    # over = SMOTENC(sampling_strategy=0.20, categorical_features=df_X_train.columns.get_indexer(cat_cols))
    over = SMOTE(sampling_strategy=0.20)    
    estimator = MAPPING_MODELS[estimator_str]        
    if "cv" in steps:
        pipeline = Pipeline(steps=[('over', over), ('estimator', estimator)], verbose=True)
        dummy_model = DummyClassifier(strategy='most_frequent')
        pipeline_dummy = Pipeline(steps=[('over', over), ('dummy_model', dummy_model)])
        pipeline_dummy.fit(X_train_std, y_train)
        dummy_y_pred = pipeline_dummy.predict(X_test)
        click.echo(f"Test with dummy classifier: {roc_auc_score(y_test, dummy_y_pred)}")            
        click.echo("Cross validation...") 
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)
        params = {f"estimator__{hyperparam}": values for hyperparam, values in MAPPING_PARAMS[estimator_str].items()}
        params.update({"over__k_neighbors": [5]})
        grid = GridSearchCV(pipeline, params, cv=cv, scoring="roc_auc", verbose=2, refit=False)
        grid.fit(X_train_std, y_train)
        print(f"{grid.best_params_} : ROC AUC = {grid.best_score_}")
        best_params = {param.split("__")[1]: value for param,value in grid.best_params_.items() if param.split("__")[0] == "estimator"}
        best_estimator = estimator.set_params(**best_params)
        best_estimator.fit(X_train_std, y_train)
        y_classes = best_estimator.predict(X_test_std)
        print("Classes : ")
        print(pd.DataFrame(y_classes).value_counts()) 
        click.echo(f"ROC AUC Test Score for {estimator_str}: {roc_auc_score(y_test, y_classes)}") 
        click.echo("Recording model in a pickle...")
        pickle.dump(best_estimator, open("model.pickle", "wb"))
        pickle.dump(scaler, open("scaler.pickle", "wb"))
        # df_X_train.to_csv("df_train.csv", index=False)
        trained_model = best_estimator


    if "predict" in steps:
        if not "cv" in steps and model is not None:
            trained_model = pickle.load(open("model.pickle", "rb"))        
        click.echo("Predicting...")
        X = data[features].values
        y = data.TARGET.values
        X_std = scaler.transform(X)      
        y_probas_df = pd.DataFrame(trained_model.predict_proba(X_std))
        y_probas_df["target"] = y
        y_probas_df["SK_ID_CURR"] = ids
        print(y_probas_df.head())
        print(y_probas_df.shape)
        print(y_probas_df.describe())
        y_probas_df.to_csv("df_probas.csv", index=False) 

    if "explain" in steps:
        if not "cv" in steps and model is not None:
            trained_model = pickle.load(open("model.pickle", "rb"))
        explainer = shap.TreeExplainer(model=estimator, model_output="probability", data=X_train_std, feature_names=features)
        click.echo(f"SHAP expected value : {explainer.expected_value}")
        click.echo(f"Model mean value : {trained_model.predict_proba(X_train).mean()}")
        click.echo(f"Recording SHAP Explainer in a pickle...")
        pickle.dump(explainer, open("explainer.pickle", "wb"))
        click.echo("Computing all SHAP values...")
        shap_values = explainer.shap_values(X_train_std)
        print(shap_values.shape)
        df_shap = pd.DataFrame(shap_values, columns=features)
        df_shap["SK_ID_CURR"] = df_X_train.SK_ID_CURR.values
        click.echo(f"Recording SHAP values in a CSV...")
        df_shap.to_csv("df_shap.csv", index=False)
        plt.figure(figsize=[25,10])
        shap.summary_plot(df_shap.drop(columns=["SK_ID_CURR"]), df_X_train.drop(columns=["SK_ID_CURR"]), feature_names=features, plot_type="bar", max_display=10, show=False)
        plt.savefig("global_summay_plot.jpg")


def _interpret(data: pd.DataFrame, trained_model, n_samples: int = None):   
    initial_cat_cols: List[str] = json.load(open("cat_cols.json", "r"))    
    # features = [_col for _col in data.columns if _col != "TARGET" and _col != "SK_ID_CURR" and not _col in initial_cat_cols]
    features = [_col for _col in data.columns if _col != "TARGET" and _col != "SK_ID_CURR"]
    data = clean_dataset(data, features+["TARGET"])
    X = data[features].values
    y = data.TARGET.values
    ids = data.SK_ID_CURR.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)     
    explainer = shap.TreeExplainer(model=trained_model, model_output="probability", data=X_train_std, feature_names=features)
    click.echo(f"SHAP expected value : {explainer.expected_value}")
    click.echo(f"Model mean value : {trained_model.predict_proba(X_train).mean()}")
    click.echo(f"Recording SHAP Explainer in a pickle...")
    pickle.dump(explainer, open("explainer.pickle", "wb"))
    click.echo("Computing all SHAP values...")
    shap_values = explainer.shap_values(X_train_std)
    print(shap_values.shape)
    df_shap = pd.DataFrame(shap_values, columns=features)
    df_shap["SK_ID_CURR"] = ids
    click.echo(f"Recording SHAP values in a CSV...")
    df_shap.to_csv("df_shap.csv", index=False)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--debug", is_flag=True, show_default=True, default=False, help="Debug.")
def preprocess(debug):
    ''' Build new features and aggregations from the raw data files.'''
    build_features(OUTPUT_FILE, debug)


@cli.command()
@click.option('--input', help='The source for data as CSV file', required=True, type=str)
@click.option('--features', help='The list of features names to keep (JSON)', required=True, type=str)
def select(features: str, input: str):
    ''' Filter a dataset to keep the requested features.'''
    click.echo("Loading features names to keep...")
    features_names = json.load(open(features, "r"))
    click.echo("Loading raw dataset...")
    df = pd.read_csv(input)
    click.echo("Recording new dataset...")
    df[features_names].to_csv("input.csv", index=False)
    click.echo("Dataset has been recorded in input.csv")


@cli.command()
@click.option('--input', help='The source for data as CSV file', required=True, type=str)
@click.option('--estimator', help='The estimator to use', required=True, type=str)
def optimize(input: str, estimator: str):
    ''' Execute a cross-validation and display scores.'''
    df = pd.read_csv(input)
    modelize(df, estimator, cv_on=True, only_train=False)    


@cli.command("modelize")
@click.option('--input', help='The source for data as CSV file', required=True, type=str)
@click.option('--estimator', help='The estimator to use', required=True, type=str)
@click.option('--model', help='The path of the pretrained model to use for the predictions.', required=False, type=str)
@click.option('--step', help='The steps to execute', required=False, multiple=True, type=str)
def train_and_save(input: str, estimator: str, model: str = None, step: List = None):
    ''' Train model and save it as a pickle.'''
    df = pd.read_csv(input)
    if step is not None:
        modelize(df, estimator, steps=step, model=model)
    else:
        modelize(df, estimator)


@cli.command("explain")
@click.option('--input', help='The source for data as CSV file', required=True, type=str)
@click.option('--model', help='The path of the pretrained model to use for the predictions.', required=True, type=str)
@click.option('--n_samples', help='The number of samples used for SHAP interpretation', required=False, type=int)
def interpret_with_SHAP(model: str, input: str, n_samples: int=None):
    click.echo("Interpeting model outputs...")
    trained_model = pickle.load(open(model, "rb"))
    df = pd.read_csv(input)
    if n_samples is not None:
        _interpret(df, trained_model, n_samples)
    else:
        _interpret(df, trained_model)


if __name__ == "__main__":
    cli()