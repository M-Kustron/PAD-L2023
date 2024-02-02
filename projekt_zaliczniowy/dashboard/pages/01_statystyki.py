import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.sidebar.write('Milena Kustroń')

data = pd.read_csv('../dane/dane_wyczyszczone.csv')

col1, col2, col3 = st.columns(3)

with col1:
    st.title('Rozkłady zmiennych')
    zmienne = st.multiselect('Wybierz zmienną:', data.select_dtypes(include=['number']).columns, key = 'zmienne')


    for zmienna in zmienne:
        fig, ax = plt.subplots()
        ax.hist(data[zmienna], bins=30, edgecolor='white', color = 'lightcoral', density=True)
        sns.kdeplot(data[zmienna], ax=ax, color="grey", linewidth=2)

        ax.set_title(f'Wykres rozkładu dla {zmienna}')
        ax.set_xlabel('wartości zmiennej')
        ax.set_ylabel('częstotliwość występowania')
        st.pyplot(fig, use_container_width=True)

with col2:
    st.title('Zależność zmiennych od ceny')
    
    zmienne2 = st.multiselect('Wybierz drugą zmienną:', data.drop('price', axis=1).columns, key='zmienne2')

    for zmienna2 in zmienne2:
        fig, ax = plt.subplots()

        if zmienna2 in ['clarity', 'color', 'cut']:
            ax.bar( data[zmienna2], data['price'], color = 'lightcoral')
        else:
            ax.scatter(data[zmienna2], data['price'], color = 'lightcoral')

        ax.set_title(f'Zalezność ceny od {zmienna2}')
        ax.set_xlabel(zmienna2)
        ax.set_ylabel('cena')
        st.pyplot(fig)

with col3:
    st.title('Liczebność kategorii')
    kategorie = st.multiselect('Wybierz kategorię:', data.select_dtypes(include=['object', 'category']).columns, key='kategorie')

    for kategoria in kategorie:
        fig, ax = plt.subplots()
        sns.countplot(x=data[kategoria], color='lightcoral')
        ax.set_title(f'Liczebność kategorii {kategoria}')
        st.pyplot(fig)

