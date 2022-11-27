import base64
import tempfile
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st

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
    print(df)
    st.session_state.df_customer_data = (
        pd.merge(df, df_stats, how="left", left_index=True, right_index=True)
        .rename(columns={
            "0":"Client",
            "mean": "Moyenne des clients",
            "std": "Dispersion autour de la moyenne",
            "50%": "Médiane des clients",
            "min": "Minimum des clients",
            "max": "Maximum des clients",})
        .drop(columns=["count", "25%", "75%"], index=["SK_ID_CURR", "TARGET"])
    )


@st.experimental_memo
def get_customer_info(customer_id: int):
    r = requests.get(f'http://127.0.0.1:5000/api/customers?id={customer_id}').json()
    return r
    # st.session_state.r_proba = requests.get(f'http://127.0.0.1:5000/api/customers/proba?id={customer_id}').json()
    # st.session_state.r_shap_customer = requests.get(f'http://127.0.0.1:5000/api/customers/interpretability?id={customer_id}').json()


@st.experimental_memo
def decision_attribution(proba):
    if proba >= st.session_state.r_params["seuil_classif"]:
        return "Accordé"
    else:
        return "Refusé"


@st.experimental_memo
def get_customers_data_stats(data_dir: str) -> Tuple[pd.Series, pd.DataFrame]:
    df_customers_1 = pd.read_csv(f"{data_dir}/application_test.csv")
    df_customers_2 = pd.read_csv(f"{data_dir}/application_train.csv")
    df_customers = pd.concat([df_customers_1, df_customers_2])
    customers_ids = df_customers.SK_ID_CURR
    stats = df_customers.describe().T
    return customers_ids, stats

with st.spinner("Chargement..."):
    customers_ids, stats = get_customers_data_stats("./data")
    st.session_state.r_params = requests.get(f'http://127.0.0.1:5000/api/model/params').json()
# st.session_state.r_stats = requests.get(f'http://127.0.0.1:5000/api/customers/proba/stats/').json()
# st.session_state.r_shap = requests.get(f'http://127.0.0.1:5000/api/customers/interpretability/').json()


st.title("Tableau de bord - Crédit")
st.write("Ce tableau de bord permet d'afficher les informations relatives à une demande de crédit d'un client.")
st.header("Recherche par identifiant client")
with st.form("get_data", clear_on_submit=False):
    customer_id = st.number_input("ID du client :", min_value=customers_ids.min(), max_value=customers_ids.max(), key="customer_id")
    clicked = st.form_submit_button("Chercher", on_click=build_data_df, args=(stats,))

st.subheader("Informations client et conseil de décision")
if "df_customer_data" in st.session_state:
    var_to_show = st.selectbox("Information à afficher :", options=st.session_state.df_customer_data.index, key="var_to_show")
    print(st.session_state.customer_id)
    st.dataframe(st.session_state.df_customer_data.loc[var_to_show,:])
    # df = df.loc[df.Info.isin(["CODE_GENDER", "DAYS_BIRTH", "AMT_CREDIT"]),:]
    # df = df.append({"Info": "Probabilité de remboursement", "Valeur": f"{round(st.session_state.r_proba['P_OK']*100,1)} %"}, ignore_index=True)
    
    # col_left, col_right = st.columns(2)
    # with col_left:        
    #     st.dataframe(df)
    # with col_right:
    #     st.metric("Décision conseillée pour l'attribution du prêt", decision_attribution(st.session_state.r_proba['P_OK']))
    #     st.metric("Probabilité de remboursement", f"{round(st.session_state.r_proba['P_OK']*100,1)} %")
    # st.subheader("Détails de la modélisation client")
    # shap_values = pd.DataFrame(st.session_state.r_shap_customer).values[:,0]
    # params = requests.get(f'http://127.0.0.1:5000/api/model/params').json()
    # explanation = shap.Explanation(values = shap_values, base_values=params["expected_value"], feature_names=params["features"])    
    # data = pd.DataFrame.from_dict(st.session_state.r_data, orient="index").T[params["features"]].values    
    # col_left, col_right = st.columns(2)
    # with col_left:  
    #     waterfall_plot = shap.plots.waterfall(explanation, max_display=6, show=True) 
    #     plt.title("Importance relative des variables pour le client demandé")         
    #     st.pyplot(waterfall_plot)
    #     plt.close()
    # with col_right:
    #     fig, ax = plt.subplots(figsize=[4,2])
    #     ax.bar(x=[str(st.session_state.customer_id)], height=1.0, alpha=0.0)
    #     ax.axhline(y=st.session_state.r_stats["q3"], linestyle="-.", label="3ème quantile (75 %)", color="forestgreen")
    #     ax.axhline(y=st.session_state.r_stats["mean"], linestyle="--", label="Moyenne", color="royalblue")
    #     ax.axhline(y=st.session_state.r_stats["q1"], linestyle="-.", label="1er quantile (25 %)", color="forestgreen")
    #     ax.axhline(y=st.session_state.r_proba['P_OK'], linestyle="-.", label="Client", color="firebrick")
    #     ax.legend(bbox_to_anchor=(1.1, 1.05))
    #     ax.set(ylabel="Probabilité", xlabel="Identifiant client", title="Situation du client dans la base clients")
    #     st.pyplot(fig)
    #     plt.close()
    # force_plot = shap.plots.force(base_value=params["expected_value"], shap_values=shap_values, features=data, feature_names=params["features"], show=False, matplotlib=True)
    # st.pyplot(force_plot)
    # plt.close()
    
