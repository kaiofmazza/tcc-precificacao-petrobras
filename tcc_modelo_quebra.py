# -*- coding: utf-8 -*-
"""
TCC – Análise de Preços dos Combustíveis com Quebra Estrutural
Autor: Kaio
Descrição:
Modelo de intervenção em séries temporais para avaliar os efeitos da mudança
na política de preços da Petrobras em maio de 2023, com controle por Brent
(convertido para R$), taxa de câmbio, tendência temporal e interação.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


# =========================================================
# FUNÇÃO AUXILIAR – TABELA COMO FIGURA
# =========================================================
def mostrar_tabela_como_plot(df_tabela):
    fig, ax = plt.subplots(figsize=(12, 0.6 + 0.4 * len(df_tabela)))
    ax.axis('off')
    tabela_plot = ax.table(
        cellText=df_tabela.values,
        colLabels=df_tabela.columns,
        rowLabels=df_tabela.index,
        loc='center',
        cellLoc='center'
    )
    tabela_plot.scale(1, 1.5)
    plt.tight_layout()
    plt.show()


# =========================================================
# 1. LEITURA DOS DADOS
# =========================================================
arquivo = 'dados_tcc_historico.xlsx'
df = pd.read_excel(arquivo)

df['data'] = pd.to_datetime(df['data'])
df = df.sort_values('data').reset_index(drop=True)

colunas_necessarias = {
    'data',
    'preco_diesel',
    'preco_gasolina',
    'preco_brent',
    'preco_dolar'
}
faltando = colunas_necessarias - set(df.columns)
if faltando:
    raise ValueError(f"Colunas ausentes no arquivo de dados: {faltando}")


# =========================================================
# 2. CRIAÇÃO DE VARIÁVEIS
# =========================================================
df['tempo'] = range(1, len(df) + 1)
df['pos_2023'] = (df['data'] >= '2023-05-01').astype(int)
df['tempo_pos'] = df['tempo'] * df['pos_2023']

# Brent convertido para reais (R$/barril)
df['brent_rs'] = df['preco_brent'] * df['preco_dolar']


# =========================================================
# 3. MODELOS ECONOMÉTRICOS
# =========================================================
modelo_gasolina = smf.ols(
    'preco_gasolina ~ tempo + pos_2023 + tempo_pos + brent_rs + preco_dolar',
    data=df
).fit()

modelo_diesel = smf.ols(
    'preco_diesel ~ tempo + pos_2023 + tempo_pos + brent_rs + preco_dolar',
    data=df
).fit()


# =========================================================
# 4. FIGURA 1 – SÉRIES TEMPORAIS EM NÍVEL
# =========================================================
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.set_xlabel('Data')
ax1.set_ylabel('R$/L ou R$/US$')

ax1.plot(df['data'], df['preco_diesel'], label='Diesel S10 (R$/L)', color='blue')
ax1.plot(df['data'], df['preco_gasolina'], label='Gasolina A (R$/L)', color='orange')
ax1.plot(df['data'], df['preco_dolar'], label='Câmbio (R$/US$)', color='red')
ax1.axvline(
    pd.to_datetime('2023-05-01'),
    color='black',
    linestyle='--',
    label='Mudança de Política (Mai/23)'
)
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel('Brent (R$/barril)')
ax2.plot(df['data'], df['brent_rs'], label='Brent (R$/barril)', color='green')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()


# =========================================================
# 5. FIGURA 2 – SÉRIES NORMALIZADAS (0–1)
# =========================================================
df_norm = df.copy()
for col in ['preco_diesel', 'preco_gasolina', 'brent_rs', 'preco_dolar']:
    df_norm[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_norm, x='data', y='preco_diesel_norm', label='Diesel S10 (R$/L)')
sns.lineplot(data=df_norm, x='data', y='preco_gasolina_norm', label='Gasolina A (R$/L)')
sns.lineplot(data=df_norm, x='data', y='brent_rs_norm', label='Brent (R$/barril)')
sns.lineplot(data=df_norm, x='data', y='preco_dolar_norm', label='Câmbio (R$/US$)')
plt.axvline(
    pd.to_datetime('2023-05-01'),
    color='red',
    linestyle='--',
    label='Mudança de Política (Mai/23)'
)
plt.xlabel('Data')
plt.ylabel('Valor normalizado (0–1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================================================
# 6. FIGURA 3 – BOXPLOT DO DIESEL
# =========================================================
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='pos_2023', y='preco_diesel')
plt.xticks([0, 1], ['Antes de Mai/23', 'Depois de Mai/23'])
plt.xlabel('Período')
plt.ylabel('Preço do Diesel (R$/litro)')
plt.tight_layout()
plt.show()


# =========================================================
# 7. FIGURA 4 – BOXPLOT DA GASOLINA
# =========================================================
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='pos_2023', y='preco_gasolina')
plt.xticks([0, 1], ['Antes de Mai/23', 'Depois de Mai/23'])
plt.xlabel('Período')
plt.ylabel('Preço da Gasolina (R$/litro)')
plt.tight_layout()
plt.show()


# =========================================================
# 8. TABELAS (1 a 3) – DESCRITIVAS E CORRELAÇÕES
# =========================================================
tabela1 = (
    df.groupby('pos_2023')[['preco_diesel', 'preco_gasolina', 'brent_rs', 'preco_dolar']]
    .agg(['mean', 'std'])
    .round(3)
)

tabela2 = (
    df[df['pos_2023'] == 0][['preco_diesel', 'preco_gasolina', 'brent_rs', 'preco_dolar']]
    .corr()
    .round(3)
)

tabela3 = (
    df[df['pos_2023'] == 1][['preco_diesel', 'preco_gasolina', 'brent_rs', 'preco_dolar']]
    .corr()
    .round(3)
)

mostrar_tabela_como_plot(tabela1)
mostrar_tabela_como_plot(tabela2)
mostrar_tabela_como_plot(tabela3)


# =========================================================
# 9. TABELAS (4 e 5) – REGRESSÕES COM R² AJUSTADO
# =========================================================
tabela4 = modelo_gasolina.summary2().tables[1].round(4)
tabela5 = modelo_diesel.summary2().tables[1].round(4)


def adicionar_r2_ajustado(tabela, r2_adj):
    tabela_out = tabela.copy()
    nova_linha = {col: '' for col in tabela_out.columns}
    nova_linha[tabela_out.columns[-1]] = round(r2_adj, 4)
    tabela_out.loc['R² ajustado'] = nova_linha
    return tabela_out


tabela4 = adicionar_r2_ajustado(tabela4, modelo_gasolina.rsquared_adj)
tabela5 = adicionar_r2_ajustado(tabela5, modelo_diesel.rsquared_adj)

mostrar_tabela_como_plot(tabela4)
mostrar_tabela_como_plot(tabela5)
