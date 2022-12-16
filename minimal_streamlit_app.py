from typing import List, Tuple
import pandas as pd
import streamlit as st
from shillelagh.backends.apsw.db import connect


st.set_page_config(
    page_title="Tableau de bord - Crédit",
    layout="wide",
    menu_items={}
)
st.set_option('deprecation.showPyplotGlobalUse', False)

connection = connect(":memory:")
cursor = connection.cursor()
sheet_url = st.secrets["public_gsheets_url"]
query = f'SELECT * FROM "{sheet_url}"'
rows = cursor.execute(query)
df_train = pd.DataFrame(rows)

st.write(rows[1])
st.dataframe(df_train.head())




