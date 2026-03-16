import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM


# IMPORT DATA

df = pd.read_excel('Brasil_data.xlsx')

df['Date'] = pd.to_datetime(df['Date'], format='%m/%Y')
df = df.set_index('Date')
df = df.sort_index()
df.index.freq = 'MS'

df = df[['RER', 'Reserva_real', 'Commod_real']]
df['RER'] = df['RER'].str.replace(',', '.').astype(float)

print(df.describe().round(2))


# ADF TEST

for col in df.columns:
    adf = adfuller(df[col], autolag='AIC')
    kpss_stat, kpss_p, _, _ = kpss(df[col], regression='c', nlags='auto')
    print(f"{col} — ADF p = {adf[1]:.4f} | KPSS p = {kpss_p:.4f}")


# LAG SELECTION

lag_test = VAR(df)
lag_selection = lag_test.select_order(maxlags=12)
print(lag_selection.summary())

p = max(2, lag_selection.selected_orders['bic'])
print(f"Lag escolhido: {p}")

# JOHANSEN TEST

johansen = coint_johansen(df, det_order=0, k_ar_diff=p-1)
print(johansen.lr1)
print(johansen.cvt)


# VECM

vecm = VECM(df, k_ar_diff=p-1, coint_rank=1, deterministic='co')
vecm_fit = vecm.fit()
print(vecm_fit.summary())


#IRF E FEVD

var_model = VAR(df)
var_fit = var_model.fit(p)

irf = var_fit.irf(24)
irf.plot(orth=True, signif=0.05)
plt.suptitle('Funções Impulso-Resposta — Brasil', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('irf.png', dpi=150, bbox_inches='tight')
plt.show()

fevd = var_fit.fevd(24)
print(fevd.summary())