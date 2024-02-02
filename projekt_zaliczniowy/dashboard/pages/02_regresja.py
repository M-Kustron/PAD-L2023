import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns

st.sidebar.write('Milena Kustroń')

data = pd.read_csv(('../dane/dane_wyczyszczone.csv'))

st.title('Postać modelu: price ~ carat + clarity + cut + z dimension + z dimension*carat')
st.title('Wyniki modelu:')
st. text('''Dep. Variable:                  price   R-squared:                       0.969
Model:                            OLS   Adj. R-squared:                  0.967
Method:                 Least Squares   F-statistic:                     466.8
Date:                Sun, 28 Jan 2024   Prob (F-statistic):          1.41e-127
Time:                        14:38:07   Log-Likelihood:                -1475.3
No. Observations:                 191   AIC:                             2977.
Df Residuals:                     178   BIC:                             3019.
Df Model:                          12                                         
Covariance Type:            nonrobust                                         
==========================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------
Intercept              -3588.5206   1017.933     -3.525      0.001   -5597.289   -1579.752
clarity[T. if]          1566.9722    168.071      9.323      0.000    1235.304    1898.640
clarity[T. si1]          824.4004    157.096      5.248      0.000     514.390    1134.411
clarity[T. si2]          675.2431    142.111      4.752      0.000     394.805     955.682
clarity[T. vvs1]        1363.6484    147.174      9.266      0.000    1073.219    1654.078
clarity[T. vvs2]        1307.7826    147.522      8.865      0.000    1016.666    1598.900
cut[T. good]            -287.5049    131.827     -2.181      0.031    -547.650     -27.360
cut[T. ideal]           -130.9979    149.812     -0.874      0.383    -426.633     164.637
cut[T. premium]          253.1023    137.986      1.834      0.068     -19.197     525.402
cut[T. very good]        -49.2980    141.696     -0.348      0.728    -328.918     230.322
carat                   3765.5137   1391.220      2.707      0.007    1020.106    6510.921
Q('z dimension')         424.2077    310.437      1.366      0.174    -188.403    1036.818
Q('z dimension'):carat  1213.9246    345.612      3.512      0.001     531.900    1895.949
==============================================================================
Omnibus:                        4.674   Durbin-Watson:                   1.888
Prob(Omnibus):                  0.097   Jarque-Bera (JB):                5.101
Skew:                           0.200   Prob(JB):                       0.0780
Kurtosis:                       3.693   Cond. No.                         200.
==============================================================================''')



model6 = smf.ols(formula="price ~ carat + clarity + cut + Q('z dimension') + Q('z dimension')*carat", data=data).fit()

col1, col2 = st.columns([1, 1]) 

with col1:
    st.title('Załozenia regresji:')
    with st.expander('Liniowość związku między odpowiedzią a zmiennymi objaśniającymi'):
        for zmienna in ('carat', 'clarity', 'cut', 'x dimension'):
            fig, ax = plt.subplots()

            if zmienna in ['clarity', 'cut']:
                ax.bar( data[zmienna], data['price'], color = 'lightcoral')
            else:
                ax.scatter(data[zmienna], data['price'], color = 'lightcoral')

            ax.set_title(f'Zalezność ceny od {zmienna}')
            ax.set_xlabel(zmienna)
            ax.set_ylabel('cena')
            st.pyplot(fig)


    with st.expander('Niezależność statystyczna reszt'):
        
        data["residuals"] = model6.resid
        for zmienna in ('carat', 'clarity', 'cut', 'x dimension'):
            plt.figure()
            plt.scatter(data[zmienna], data.residuals, color = 'lightcoral')
            plt.title(f'{zmienna} vs reszty')
            st.pyplot(plt)

    with st.expander('Homoskedastyczność'):

        plt.figure()
        plt.scatter(model6.fittedvalues, model6.resid, color = 'lightcoral')
        plt.axhline(y=0, color='grey')
        plt.xlabel('Przewidywane wartości')
        plt.ylabel('Reszty')
        plt.title('Wykres reszt')
        st.pyplot(plt)

    with st.expander('Normalność rozkładu błędów'):

        fig = sm.qqplot(model6.resid, fit=True, line='45')
        plt.title("QQ wykres reszt")
        st.pyplot(plt)

    st.title('Wizualizacja skuteczności:')

    data['fitted'] = model6.fittedvalues
    plt.figure(figsize=(8,8))
    plt.scatter(data['fitted'], data['price'], color = 'lightcoral') 
    plt.plot([0, 12000], [0, 12000], color='grey')
    plt.xlabel('Predykcja modelu')
    plt.ylabel('Rzeczywisty price')
    plt.title('Predykcja vs. rzeczywiste wartości')
    st.pyplot(plt)
        