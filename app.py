# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output, State, dash_table
import plotly.express as px
import pandas as pd
import requests
import shap

app = Dash(__name__)

# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

# markdown_text = '''
# ### Dash and Markdown

# Dash apps can be written in Markdown.
# Dash uses the [CommonMark](http://commonmark.org/)
# specification of Markdown.
# Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
# if this is your first introduction to Markdown!
# '''

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Tableau de bord - Crédit', style={'textAlign': 'center'}),

    html.Div(children='''
        Ce tableau de bord permet d'afficher les informations relatives à une demande de crédit d'un client.
    '''),

    # html.Div(dcc.Markdown(children=markdown_text)),

    html.Div(children=[
        html.Label('ID du client :'),
        dcc.Input(id="customer_id", value='0', type='number'),
        html.Button(id="submit-customer-id", n_clicks=0, children="Envoyer")
    ]),
    
    # dash_table.DataTable(id='display_data_df'),
    html.Div(id="display_data_df"),

    html.Div(id="customer_data"),

    html.Div(id="display_proba"),

    html.Div(id="display_seuil"),

    dcc.Graph(
        id='shap-graph'
    ),
])

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


@app.callback(
    Output(component_id='display_data_df', component_property='children'),
    Input(component_id='submit-customer-id', component_property='n_clicks'),
    State(component_id='customer_id', component_property='value')
)
def display_customer_data(n_clicks, customer_id):
    if n_clicks > 0:
        r = requests.get(f'http://127.0.0.1:5000/api/customers?id={customer_id}').json()
        df = pd.DataFrame.from_dict(data=r, orient="index").reset_index().rename(columns={0:"Value", "index": "Info"})
        df = df[["GENDER", "DAYS_BIRTH", "CREDIT_AMT"]]
        return dash_table.DataTable(data=df.to_dict("records"), style_table={'height': '300px', 'overflowY': 'auto'})

     
@app.callback(
    Output(component_id='display_proba', component_property='children'),
    Input(component_id='submit-customer-id', component_property='n_clicks'),
    State(component_id='customer_id', component_property='value')
)
def display_customer_proba(n_clicks, customer_id):
    if n_clicks > 0:
        r = requests.get(f'http://127.0.0.1:5000/api/customers/proba?id={customer_id}').json()
        return f"Probabilité de remboursement : {r['P_OK']}"


@app.callback(
    Output(component_id="display_seuil", component_property="children"),
    Input(component_id="submit-customer-id", component_property="n_clicks")
)
def display_seuil_classif(n_clicks):
    r = requests.get(f'http://127.0.0.1:5000/api/model/params').json()
    return f"Seuil de classification : {r['seuil_classif']}"

# @app.callback(
#     Output(component_id="shap-graph", component_property="figure"),
#     Input(component_id='submit-customer-id', component_property='n_clicks'),
#     State(component_id='customer_id', component_property='value')
# )
# def display_interpetabilty_plot(n_clicks, customer_id):
#     if n_clicks > 0:
#         r = requests.get(f'http://127.0.0.1:5000/api/customers/interpretability?id={customer_id}').json()
#         shap_values = pd.DataFrame(data=r).values
#         print(shap_values)
#         params = requests.get(f'http://127.0.0.1:5000/api/model/params').json()
#         explanation = shap.Explanation(values = shap_values, base_values=params["expected_value"], feature_names=params["features"])
#         return None
#         # return shap.plots.waterfall(explanation, matplotlib=True)

if __name__ == '__main__':
    app.run_server(debug=True)