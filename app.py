# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from pathlib import Path
import tempfile
from dash import Dash, html, dcc, Input, Output, State, dash_table
import plotly.express as px
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
import base64

app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Tableau de bord - Crédit', style={'textAlign': 'center'}),
    html.Div(children='''
        Ce tableau de bord permet d'afficher les informations relatives à une demande de crédit d'un client.
    '''),

    html.H2("Recherche par identifiant client"),
    html.H3("Informations client et probabilité de remboursement"),
    html.Div(children=[
        html.Label('ID du client :'),
        dcc.Input(id="customer_id", value='0', type='number'),
        html.Button(id="submit-customer-id", n_clicks=0, children="Chercher")
    ]),    
    html.Div(id="display_data_df"),

    html.H3("Détails de la modélisation client"),
    html.Div(children=[html.Img(id='shap-waterfall-graph')]),
    html.Div(children=[html.Img(id='shap-force-graph')])
])


@app.callback(
    Output(component_id='display_data_df', component_property='children'),
    Input(component_id='submit-customer-id', component_property='n_clicks'),
    State(component_id='customer_id', component_property='value')
)
def display_customer_data(n_clicks, customer_id):
    if n_clicks > 0:
        r = requests.get(f'http://127.0.0.1:5000/api/customers?id={customer_id}').json()      
        df = pd.DataFrame.from_dict(data=r, orient="index").reset_index().rename(columns={0:"Valeur", "index": "Info"})
        df = df.loc[df.Info.isin(["CODE_GENDER", "DAYS_BIRTH", "AMT_CREDIT"]),:]
        r = requests.get(f'http://127.0.0.1:5000/api/customers/proba?id={customer_id}').json()
        df = df.append({"Info": "Probabilité de remboursement", "Valeur": f"{round(r['P_OK']*100,1)} %"}, ignore_index=True)
        return dash_table.DataTable(data=df.to_dict("records"), style_table={'height': '300px', 'overflowY': 'auto'})

     
def get_seuil_classif():
    r = requests.get(f'http://127.0.0.1:5000/api/model/params').json()
    return float(r['seuil_classif'])


@app.callback(
        Output(component_id="shap-waterfall-graph", component_property="src"),
        Input(component_id='submit-customer-id', component_property='n_clicks'),
        State(component_id='customer_id', component_property='value')
)        
def draw_waterfall_plot(n_clicks, customer_id):
    if n_clicks > 0:
        r = requests.get(f'http://127.0.0.1:5000/api/customers/interpretability?id={customer_id}').json()
        shap_values = pd.DataFrame(r).values[:,0]
        params = requests.get(f'http://127.0.0.1:5000/api/model/params').json()
        explanation = shap.Explanation(values = shap_values, base_values=params["expected_value"], feature_names=params["features"])
        with tempfile.TemporaryDirectory() as temp_dir:
            plt.figure()
            shap.plots.waterfall(explanation, show=False)
            plt.savefig(Path(temp_dir, "waterfall_plot_html.png"), bbox_inches="tight")
            plt.close()
            return b64_image(Path(temp_dir, "waterfall_plot_html.png")) 


@app.callback(
        Output(component_id='shap-force-graph', component_property="src"),
        Input(component_id='submit-customer-id', component_property='n_clicks'),
        State(component_id='customer_id', component_property='value')
)   
def draw_force_plot(n_clicks, customer_id):
    if n_clicks > 0:
        r = requests.get(f'http://127.0.0.1:5000/api/customers/interpretability?id={customer_id}').json()
        shap_values = pd.DataFrame(r).values[:,0]
        params = requests.get(f'http://127.0.0.1:5000/api/model/params').json()
        r = requests.get(f'http://127.0.0.1:5000/api/customers?id={customer_id}').json()
        data = pd.DataFrame.from_dict(r, orient="index").T[params["features"]].values
        with tempfile.TemporaryDirectory() as temp_dir:
            plt.figure()
            shap.plots.force(base_value=params["expected_value"], shap_values=shap_values, features=data, feature_names=params["features"], show=False, matplotlib=True)
            plt.savefig(Path(temp_dir, "force_plot_html.png"), bbox_inches="tight")
            plt.close()
            return b64_image(Path(temp_dir, "force_plot_html.png"))


def b64_image(image_filepath: Path) -> str:
    with open(str(image_filepath), 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')


if __name__ == '__main__':
    app.run_server(debug=True)