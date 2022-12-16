from typing import List, Tuple
import pandas as pd
import streamlit as st
from shillelagh.backends.apsw.db import connect
from shillelagh.adapters.registry import registry


st.set_page_config(
    page_title="Tableau de bord - Crédit",
    layout="wide",
    menu_items={}
)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write(registry.loaders.keys())
connection = connect(":memory:", adapters=["gsheetsapi"])
cursor = connection.cursor()
sheet_url = st.secrets["public_gsheets_url"]
query = f'SELECT * FROM "{sheet_url}"'
response = cursor.execute(query)
all_rows: List[Tuple] = response.fetchall()
st.dataframe(pd.DataFrame(all_rows))

