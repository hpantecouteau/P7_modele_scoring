import pickle
from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import shap
from model import SEUIL_CLASSIF

app = Flask(__name__)
df_all = pd.read_csv("full_clean_dataset.csv")
features = [_col for _col in df_all.columns if _col != "TARGET"]
X = df_all.loc[:, features].values
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)
trained_model = pickle.load(open("model.pickle", "rb"))
explainer = pickle.load(open("explainer.pickle", "rb"))
shap_values = pickle.load(open("shap_values.pickle", "rb"))[0]

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Distant Reading Customers DB</h1>
    <p>A prototype API for distant reading of Customers DataBase.</p>'''


@app.route('/api/customers/all', methods=['GET'])
def api_all():    
    df = df_all[:100]  
    dict_all = df.to_dict(orient="index")  
    return jsonify(dict_all)

@app.route('/api/customers/', methods=['GET'])
def show_customer_info():
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."
    return jsonify(df_all.loc[id].to_dict())

@app.route('/api/customers/proba/', methods=['GET'])
def get_proba():
    if 'id' in request.args:
        id = int(request.args.get('id', ''))
    else:
        return "Error: No id field provided. Please specify an id."
    proba = trained_model.predict_proba(X_std[id, :].reshape(1, -1))
    response = {
        "P_OK": round(proba[0][0],2),
        "P_NOT_OK": round(proba[0][1],2),
    }
    return jsonify(response)

@app.route('/api/customers/proba/stats/', methods=['GET'])
def get_stats():
    proba_ok = trained_model.predict_proba(X_std)[:,0]
    return jsonify({
        "min": np.min(proba_ok),
        "mean": np.mean(proba_ok),
        "median": np.median(proba_ok),
        "q1": np.quantile(proba_ok, 0.25),
        "q3": np.quantile(proba_ok, 0.75),
        "max": np.max(proba_ok)
    })

@app.route('/api/model/params', methods=['GET'])
def get_model_params():
    dict_params = trained_model.get_params(deep=False)
    dict_params["seuil_classif"] = SEUIL_CLASSIF
    dict_params["expected_value"] = explainer.expected_value[0]
    dict_params["features"] = features
    return jsonify(dict_params)

@app.route('/api/customers/interpretability/', methods=['GET'])
def get_shap_values():
    if 'id' in request.args:
        id = int(request.args.get('id', ''))
        df = pd.DataFrame(shap_values[id,:]).to_dict()
    else:
        df = pd.DataFrame(shap_values).to_dict()
    return jsonify(df)
