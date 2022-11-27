import pickle
from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import shap
from model import SEUIL_CLASSIF

app = Flask(__name__)
df_input = pd.read_csv("full_clean_dataset.csv")
df_customers_1 = pd.read_csv("./data/application_test.csv")
df_customers_2 = pd.read_csv("./data/application_train.csv")
df_customers = pd.concat([df_customers_1, df_customers_2])
features_names = [_col for _col in df_input.columns if _col != "TARGET" and _col != "SK_ID_CURR"]
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(df_input.loc[:,features_names])
all_customers_features_std = pd.DataFrame(X_std, columns=features_names)
all_customers_features_std["SK_ID_CURR"] = df_input["SK_ID_CURR"]
trained_model = pickle.load(open("model.pickle", "rb"))
explainer = pickle.load(open("explainer.pickle", "rb"))
df_shap = pd.read_csv("df_shap.csv")

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Distant Reading Customers DB</h1>
    <p>A prototype API for distant reading of Customers DataBase.</p>'''


@app.route('/api/customers/all', methods=['GET'])
def api_all():    
    dict_all = df_customers.to_dict(orient="index")  
    return jsonify(dict_all)

@app.route('/api/customers/', methods=['GET'])
def show_customer_info():
    if 'id' in request.args:
        id = int(request.args['id'])
        df = df_customers.loc[df_customers.SK_ID_CURR == id]
        df = df.fillna("nan")     
    else:
        return "Error: No id field provided. Please specify an id."
    return jsonify(df.to_dict())

@app.route('/api/customers/proba/', methods=['GET'])
def get_proba():
    if 'id' in request.args:
        id = int(request.args.get('id', ''))
    else:
        return "Error: No id field provided. Please specify an id."
    df = all_customers_features_std.loc[all_customers_features_std.SK_ID_CURR == id,features_names]
    if not df.empty:
        customer_features = df.values
        proba = trained_model.predict_proba(customer_features)
        response = {
            "P_OK": round(proba[0][0],2),
            "P_NOT_OK": round(proba[0][1],2),
        }
    else:
        response = {
            "P_OK": "nan",
            "P_NOT_OK": "nan",
        }
    return jsonify(response)

# @app.route('/api/customers/proba/stats/', methods=['GET'])
# def get_stats():
#     proba_ok = trained_model.predict_proba(X_std)[:,0]
#     return jsonify({
#         "min": np.min(proba_ok),
#         "mean": np.mean(proba_ok),
#         "median": np.median(proba_ok),
#         "q1": np.quantile(proba_ok, 0.25),
#         "q3": np.quantile(proba_ok, 0.75),
#         "max": np.max(proba_ok)
#     })

@app.route('/api/model/params', methods=['GET'])
def get_model_params():
    dict_params = trained_model.get_params(deep=False)
    dict_params["seuil_classif"] = SEUIL_CLASSIF
    dict_params["expected_value"] = explainer.expected_value[0]
    dict_params["features"] = features_names
    return jsonify(dict_params)

@app.route('/api/customers/interpretability/', methods=['GET'])
def get_shap_values():
    if 'id' in request.args:
        id = int(request.args.get('id', ''))
        response = df_shap.loc[df_shap.SK_ID_CURR == id,:].to_dict()
    else:
        response = df_shap.to_dict()
    return jsonify(response)
