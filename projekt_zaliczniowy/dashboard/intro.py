import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.sidebar.write('Milena Kustroń')

dane_przed = pd.read_csv('../dane/dane_wyczyszczone.csv')
dane_po = pd.read_csv('../dane/messy_data.csv')

st.title('Dane wejściowe')
st.text('W ponizszej tabeli przedstawione są dane źródłowe pochodzące z pliku messy_data.csv przed częścią EDA.')

st.dataframe(dane_po.head(), width=4000)

st.text("""W tej tabeli natomiast przedstawione są dane będące efektem EDA, w tym: 
        - usunięcie spacji przed nazwami kolumn,
        - zamiana pustych wartości "" na  null w celu dalszego uzupełnienia
        - zmiana typów danych
        - weryfikacja istnienia duplikatów
        - wprowadzenie jednolitości wielkości liter w przypadku wartości tekstowych
        - usunięcie rekordu z nadmierną ilością nulli ze względu na małą wartość informacyjną
        - wyłapanie usunięcie outliersów według kryterium 1,5IQR ze względu na wraliwość regresji liniowej na outliersy
        - uzupełnie brakujących wartości przy pomocy algorytmu KNN""")
st.dataframe(dane_przed.head(), width=4000)


