from typing import List, Tuple
import pandas as pd
import streamlit as st
from gsheetsdb import connect


@st.cache(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows


st.set_page_config(
    page_title="Tableau de bord - Crédit",
    layout="wide",
    menu_items={}
)
st.set_option('deprecation.showPyplotGlobalUse', False)


conn = connect()
sheet_url = st.secrets["public_gsheets_url"]
X_train = run_query(f'SELECT * FROM "{sheet_url}"')
df_train = pd.DataFrame(X_train)

st.write(X_train[1])
st.dataframe(df_train.head())




