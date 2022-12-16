import base64
from pathlib import Path
from typing import List, Tuple
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st
from gsheetsdb import connect


@st.cache(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows


sns.set_theme()
st.set_page_config(
    page_title="Tableau de bord - Crédit",
    layout="wide",
    menu_items={}
)
st.set_option('deprecation.showPyplotGlobalUse', False)


conn = connect()
sheet_url = st.secrets["public_gsheets_url"]
df_train = run_query(f'SELECT * FROM "{sheet_url}"')


def b64_image(image_filepath: Path) -> str:
    with open(str(image_filepath), 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')


def build_data_df(df_stats: pd.DataFrame):
    r = get_customer_info(st.session_state.customer_id)
    df = pd.DataFrame.from_dict(data=r, orient="index")
    df.columns=["Client"]
    st.session_state.df_customer_data = (
        pd.merge(df, df_stats, how="left", left_index=True, right_index=True)
        .rename(columns={
            "0":"Client",
            "mean": "Moyenne des clients",
            "std": "Dispersion",
            "50%": "Médiane des clients",
            "min": "Minimum",
            "max": "Maximum",})
        .drop(columns=["count", "25%", "75%"], index=["SK_ID_CURR", "TARGET"])
    )


@st.experimental_memo
def build_df_shap_customer(customer_id: int):
    r = get_customer_shap(customer_id)
    return pd.DataFrame.from_dict(data=r, orient="index")


@st.experimental_memo
def get_customer_info(customer_id: int):
    r = requests.get(f'http://127.0.0.1:5000/api/customers?id={customer_id}').json()
    return r
    

@st.experimental_memo
def get_customer_proba(customer_id: int):
    r = requests.get(f'http://127.0.0.1:5000/api/customers/proba?id={customer_id}').json()
    return r


@st.experimental_memo
def get_customer_shap(customer_id: int):
    r = requests.get(f'http://127.0.0.1:5000/api/customers/interpretability?id={customer_id}').json()
    return r


@st.experimental_memo
def decision_attribution(proba):
    if isinstance(proba, float):
        if proba >= st.session_state.r_params["seuil_classif"]:
            return "Accordé"
        else:
            return "Refusé"
    else:
        return np.nan


@st.experimental_memo
def get_all_train_data(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(f"{data_dir}/df_train.csv") 


with st.spinner("Chargement..."):
    st.session_state.r_params = requests.get(f'http://127.0.0.1:5000/api/model/params').json()
    st.write(df_train[1])




