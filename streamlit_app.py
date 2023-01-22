import base64
from pathlib import Path
from typing import List, Tuple
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st
from shillelagh.backends.apsw.db import connect


# sns.set_theme()
st.set_page_config(
    page_title="Tableau de bord - Crédit",
    layout="wide",
    menu_items={}
)
st.set_option('deprecation.showPyplotGlobalUse', False)


def b64_image(image_filepath: Path) -> str:
    with open(str(image_filepath), 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')


def build_data_df(df_stats: pd.DataFrame):
    r = get_customer_info(st.session_state.customer_id)
    df = pd.DataFrame.from_dict(data=r, orient="index")
    df.columns=["Client"]
    if not df.empty:
        st.session_state.df_customer_data = (
            pd.merge(df, df_stats, how="left", left_index=True, right_index=True)
            .rename(columns={
                "0":"Client",
                "mean": "Moyenne des clients",
                "std": "Dispersion",
                "50%": "Médiane des clients",
                "min": "Minimum",
                "max": "Maximum",
                "freq": "Fréquence",
                "top": "Plus fréquent",
                "unique": "Nb de val. uniques"})
            .drop(columns=["count", "25%", "75%"], index=["SK_ID_CURR", "TARGET"])
        )
    else:
        st.session_state.df_customer_data = pd.DataFrame(data=None)


@st.experimental_memo
def build_df_shap_customer(customer_id: int):
    r = get_customer_shap(customer_id)
    return pd.DataFrame.from_dict(data=r, orient="index")


@st.experimental_memo
def get_customer_info(customer_id: int):
    r = requests.get(f'https://hpanteco.pythonanywhere.com/api/customers?id={customer_id}')
    if r:
        return r.json()
    else:
        return {}
    

@st.experimental_memo
def get_customer_proba(customer_id: int):
    r = requests.get(f'https://hpanteco.pythonanywhere.com/api/customers/proba?id={customer_id}', timeout=300)
    if r:
        return r.json()
    else:
        return {
                "P_OK": "nan",
                "P_NOT_OK": "nan",
            }


@st.experimental_memo
def get_customer_shap(customer_id: int):
    r = requests.get(f'https://hpanteco.pythonanywhere.com/api/customers/interpretability?id={customer_id}')
    if r:
        return r.json()
    else:
        return {}


@st.experimental_memo
def decision_attribution(proba_perc):
    if isinstance(proba_perc, float) and pd.notnull(proba_perc):
        if proba_perc >= st.session_state.r_params["seuil_classif"]*100:
            return "Accordé"
        else:
            return "Refusé"
    else:
        return np.nan


@st.experimental_memo
def get_customers_data() -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    connection = connect(":memory:", adapters=["gsheetsapi"])
    cursor = connection.cursor()
    sheet_url = st.secrets["public_gsheets_url_input"]
    query = f'SELECT * FROM "{sheet_url}"'
    response = cursor.execute(query)
    headers: List[str] = [item[0] for item in response.description]
    all_rows: List[Tuple] = response.fetchall()
    df = pd.DataFrame(all_rows, columns=headers)
    cat_cols = st.session_state.r_params["cat_cols"]
    df = df.astype(float)
    df.SK_ID_CURR = df.SK_ID_CURR.astype(int)
    for _col in cat_cols:
        df[_col] = df[_col].astype("category")
    customers_ids = df.get("SK_ID_CURR", np.arange(0,df.shape[0],1))
    stats = df.describe(include="all").T
    return customers_ids, stats, df


@st.experimental_memo
def get_all_shap_values() -> Tuple[pd.Series, pd.DataFrame]:
    connection = connect(":memory:", adapters=["gsheetsapi"])
    cursor = connection.cursor()
    sheet_url = st.secrets["public_gsheets_url_shap"]
    query = f'SELECT * FROM "{sheet_url}"'
    response = cursor.execute(query)
    headers: List[str] = [item[0] for item in response.description]
    all_rows: List[Tuple] = response.fetchall()
    return pd.DataFrame(all_rows, columns=headers)


@st.experimental_memo
def get_default_proba_values():
    connection = connect(":memory:", adapters=["gsheetsapi"])
    cursor = connection.cursor()
    sheet_url = st.secrets["public_gsheets_url_proba"]
    query = f'SELECT * FROM "{sheet_url}"'
    response = cursor.execute(query)
    headers: List[str] = [item[0] for item in response.description]
    all_rows: List[Tuple] = response.fetchall()
    return pd.DataFrame(all_rows, columns=headers)
    

@st.experimental_memo
def draw_bivariate_plot(data: pd.DataFrame, x_var: str, y_var: str, customer_id: int):
    assert isinstance(data, pd.DataFrame) and "SK_ID_CURR" in data.columns
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(
        data.loc[:, x_var].values,
        data.loc[:, y_var].values,
        marker='+',
        s=5,
        alpha=0.25,
        color='forestgreen'
        )
    ax.scatter(
        data.loc[data.SK_ID_CURR == customer_id, x_var].values,
        data.loc[data.SK_ID_CURR == customer_id, y_var].values,
        marker='+',
        s=10,
        alpha=1,
        color="firebrick"
    )
    ax.hlines(
        y=data.loc[data.SK_ID_CURR == customer_id, y_var].values,
        xmin=data.loc[:, x_var].min(),
        xmax=data.loc[data.SK_ID_CURR == customer_id, x_var].values,
        linestyle='--',
        color="firebrick",
        alpha=0.8
    )
    ax.vlines(
        x=data.loc[data.SK_ID_CURR == customer_id, x_var].values,
        ymin=data.loc[:, y_var].min(),
        ymax=data.loc[data.SK_ID_CURR == customer_id, y_var].values,
        linestyle='--',
        color="firebrick",
        alpha=0.8
    )
    ax.set(xlabel=x_var, ylabel=y_var)
    ax.margins(0)        
    return fig


@st.experimental_memo
def draw_univariate_plot(data: pd.DataFrame, x_var: str, customer_id: int):
    assert isinstance(data, pd.DataFrame) and "SK_ID_CURR" in data.columns
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(
        data.loc[:, x_var].values,
        alpha=0.8
        )      
    ax.axvline(
        x=data.loc[data.SK_ID_CURR == customer_id, x_var].values,        
        color="firebrick",
        alpha=1.0,
        linestyle='--'
    )
    ax.set(xlabel=x_var, ylabel="Effectif")
    return fig
    

@st.experimental_memo
def show_filtered_dataframe(data: pd.DataFrame, additional_vars: List[str]):
    minimal_vars_to_show= ["AMT_ANNUITY", "AMT_CREDIT", "AMT_INCOME_TOTAL", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    return data.loc[minimal_vars_to_show+additional_vars,:]


with st.spinner("Chargement..."):
    st.session_state.r_params = requests.get(f'https://hpanteco.pythonanywhere.com/api/model/params').json()   
    customers_ids, stats, df_customers = get_customers_data()
    df_shap = get_all_shap_values()
    df_probas = get_default_proba_values()

st.title("Tableau de bord - Crédit")
st.write("Ce tableau de bord permet d'afficher les informations relatives à une demande de crédit d'un client.")
with st.form("get_data", clear_on_submit=False):
    customer_id = st.selectbox("Recherche par identifiant client :", options=customers_ids, key="customer_id")
    clicked = st.form_submit_button("Chercher", on_click=build_data_df, args=(stats,))


st.markdown(f"## Résultats et critères prépondérants dans la modélisation du client n°{st.session_state.customer_id}")
df_shap_customer = build_df_shap_customer(st.session_state.customer_id)
if not df_shap_customer.empty:
    explanation = shap.Explanation(values = df_shap_customer.drop(columns=["SK_ID_CURR"]).values[0], base_values=st.session_state.r_params["expected_value"], feature_names=st.session_state.r_params["features"])    
    col_left, col_right = st.columns(2)
    with col_left:
        proba = get_customer_proba(st.session_state.customer_id)["P_OK"]
        default_proba = float(df_probas.loc[df_probas.SK_ID_CURR == st.session_state.customer_id, "0"].values[0])
        if isinstance(proba, float) and pd.notnull(proba):
            proba_to_show = round(proba*100,1)
        else:
            proba_to_show = round(default_proba*100,1)
        st.metric("Décision conseillée pour l'attribution du prêt", decision_attribution(proba_to_show))
        fig = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = proba_to_show,
            mode = "gauge+number",
            title = {'text': "Probabilité de remboursement"},
            gauge = {'axis': {'range': [0, 100]},
                    'bar': {'color': "gray"},
                    'steps' : [
                        {'range': [0, st.session_state.r_params["seuil_classif"]*100], 'color': "coral"},
                        {'range': [st.session_state.r_params["seuil_classif"]*100, 100], 'color': "lightgreen"}],
                    'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': proba_to_show}})
        )
        st.plotly_chart(fig)          
    with col_right:
        st.slider("Afficher les x critères plus importants :", min_value=1, max_value=15, value=6, step=1, key="nb_customer_var_to_show")
        plt.figure()
        waterfall_plot = shap.plots.waterfall(explanation, show=True, max_display=st.session_state.nb_customer_var_to_show) 
        plt.title("Importance des variables pour le client demandé")         
        st.pyplot(waterfall_plot)
        plt.close()

st.markdown("## Informations client et visualisations")
if "df_customer_data" in st.session_state:
    additional_var_to_show = st.multiselect("Information à afficher :", options=st.session_state.df_customer_data.index, key="var_to_show", default="AMT_CREDIT")  
    st.dataframe(st.session_state.df_customer_data.head(5))      
    st.dataframe(show_filtered_dataframe(st.session_state.df_customer_data, additional_var_to_show))
    numerical_vars = [var for var in df_customers.select_dtypes('number').columns if var != "SK_ID_CURR" and "FLAG" not in var]
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### Situation du client pour une variable donnée :")
        st.selectbox("Variable à afficher : ", options=numerical_vars, key="x_var_uni", index=0)
        st.markdown("# ")
        st.markdown("# ")
        st.markdown("# ")
        univariate_plot = draw_univariate_plot(df_customers, st.session_state.x_var_uni, st.session_state.customer_id)
        st.pyplot(univariate_plot)
    with col_right:
        st.markdown("### Situation du client dans un nuage de points :")
        st.selectbox("Axe des abscisses (en bas) : ", options=numerical_vars, key="x_var", index=0)
        st.selectbox("Axe des ordonnées (à gauche) : ", options=numerical_vars, key="y_var", index=1)
        bivariate_plot = draw_bivariate_plot(df_customers, st.session_state.x_var, st.session_state.y_var, st.session_state.customer_id)
        st.pyplot(bivariate_plot)


st.markdown("## Critères prépondérants dans la modélisation générale")
col_left, col_right = st.columns(2)
with col_left:
    img = b64_image(Path("./global_summary_plot.jpg"))
    st.write(f'<img src="{img}" />', unsafe_allow_html=True)
