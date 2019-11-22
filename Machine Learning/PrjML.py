import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas_profiling

df = pd.read_csv('BaseANP.csv', header=0, skiprows=15)
# profile = df.profile_report(title="ANP_antes")
# profile.to_file(output_file=Path("./RelatorioAntes.html"))
# print(df.corr())
df['MES'], df['ANO'] = df['MÊS'].str.split('-').str
df = df.drop(columns='MÊS')
# print(df)
df.rename(
    columns={
        "MES": "mes",
        "ANO": "ano",
        "REGIÃO": "regiao",
        "ESTADO": "estado",
        "PRODUTO": "produto",
        "NÚMERO DE POSTOS PESQUISADOS": "num_postos",
        "UNIDADE DE MEDIDA": "uni_medida",
        "PREÇO MÉDIO REVENDA": "val_med_revenda",
        "DESVIO PADRÃO REVENDA": "desvio_padrao_revenda",
        "PREÇO MÍNIMO REVENDA": "val_min_revenda",
        "PREÇO MÁXIMO REVENDA": "val_max_revenda",
        "MARGEM MÉDIA REVENDA": "mrg_med_revenda",
        "COEF DE VARIAÇÃO REVENDA": "coef_var_revenda",
        "COEF DE VARIAÇÃO DISTRIBUIÇÃO": "coef_var_dist",
        "PREÇO MÁXIMO DISTRIBUIÇÃO": "val_max_dist",
        "PREÇO MÍNIMO DISTRIBUIÇÃO": "val_min_dist",
        "DESVIO PADRÃO DISTRIBUIÇÃO": "desvio_padrao_dist",
        "PREÇO MÉDIO DISTRIBUIÇÃO": "val_med_dist",
    },
    inplace=True
)
print(df.dtypes)
for col in ['mrg_med_revenda', 'val_med_dist', 'desvio_padrao_dist', 'val_min_dist', 'val_max_dist', 'coef_var_dist', 'ano']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print(df.dtypes)
# profile = df.profile_report(title="ANP_depois")
# profile.to_file(output_file=Path("./RelatorioDepois.html"))
# depois do relatorio, vi que essas colunas não são relevantes para conseguir os dados do valor médio dos combustíveis nos estados e regiões
df = df.drop(columns=['desvio_padrao_revenda','val_min_revenda','val_max_revenda', 'desvio_padrao_revenda', 'mrg_med_revenda', 'coef_var_revenda', 'coef_var_dist', 'val_max_dist', 'val_min_dist', 'desvio_padrao_dist', 'val_med_dist'])
# profile = df.profile_report(title="ANP_depoisDeDropar")
# profile.to_file(output_file=Path("./RelatorioDepoisDeDropar.html"))
print(df['ano'].unique().tolist())
print(df['mes'].unique().tolist())
print(df['produto'].unique().tolist())
print(df['val_med_revenda'].unique().tolist())
# plot Preço médio de revenda da gasolina e etanol agrupados por ano e região
fig, ax = plt.subplots(figsize=(25,10))
QryStr = 'ano!=19 & produto in ["GASOLINA COMUM","ETANOL HIDRATADO"]' 
QryGb  = ['ano','regiao']
df.query(QryStr).groupby(QryGb).sum()['val_med_revenda'].unstack().plot(ax=ax)
plt.grid(b=bool,axis='both')
plt.show()
# fig, ax = plt.subplots(figsize=(25,10))
# df.query(QryStr).groupby(QryGb).sum()['val_med_revenda'].pct_change().unstack().plot(ax=ax)
# plt.grid(b=bool,axis='both')
# plt.show()
# plot Preço médio de revenda da gasolina e etanol agrupados por ano e estado da regiao nordeste
QryStr = 'ano!=19 & produto in ["GASOLINA COMUM","ETANOL HIDRATADO"] & regiao in ["NORDESTE"]'
QryGb  = ['ano','estado']
fig, ax = plt.subplots(figsize=(25,10))
df.query(QryStr).groupby(QryGb).sum()['val_med_revenda'].unstack().plot(ax=ax)
plt.grid(b=bool,axis='both')
plt.show()
# fig, ax = plt.subplots(figsize=(25,10))
# df.query(QryStr).groupby(QryGb).sum()['val_med_revenda'].pct_change().unstack().plot(ax=ax)
# plt.grid(b=bool,axis='both')
# plt.show()
