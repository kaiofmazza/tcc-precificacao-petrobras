# -*- coding: utf-8 -*-
"""
TCC — Análise de Preços dos Combustíveis com Quebra Estrutural (Intervenção)
============================================================================

Objetivo
--------
Estimar o impacto da mudança na política de preços da Petrobras (mai/2023) sobre
os preços de Diesel S10 e Gasolina A no Brasil, controlando por:
- Brent (convertido para R$ via câmbio),
- taxa de câmbio (R$/US$),
- tendência temporal e inclinação pós-intervenção.

Modelo (OLS)
------------
y_t = β0 + β1*tempo + β2*pos_2023 + β3*(tempo*pos_2023) + β4*brent_rs + β5*cambio + ε_t

Saídas
------
- Figuras 1–4 (SVG): séries em nível, séries normalizadas, boxplots pré vs pós
- Tabelas 1–5 (SVG): descritivas, correlações e regressões (inclui R² ajustado)

Autor: Kaio
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import statsmodels.formula.api as smf


# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================
DATA_BREAK = "2023-05-01"
COR_LINHA_POLITICA = "black"

TAMANHO_FONTE = 11
CAMINHO_ARIAL = r"C:\Windows\Fonts\arial.ttf"  # Windows/Spyder: ajuste se necessário

ARQUIVO_DADOS = Path("dados_tcc_historico.xlsx")

PASTA_SAIDA = Path("outputs")
PASTA_FIGURAS = PASTA_SAIDA / "figuras"
PASTA_TABELAS = PASTA_SAIDA / "tabelas"


# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================
def preparar_pastas() -> None:
    """Cria as pastas de saída (figuras e tabelas)."""
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)


def configurar_estilo() -> str:
    """
    Configura estilo global (Arial 11, sem grid) e exportação SVG.
    Retorna o nome da fonte efetivamente utilizada.
    """
    # Forçar Arial no Spyder/Anaconda (Windows)
    if Path(CAMINHO_ARIAL).exists():
        fm.fontManager.addfont(CAMINHO_ARIAL)
        fonte_padrao = fm.FontProperties(fname=CAMINHO_ARIAL).get_name()
    else:
        # Fallback seguro se o Arial não for encontrado
        fonte_padrao = "DejaVu Sans"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [fonte_padrao],
        "font.size": TAMANHO_FONTE,
        "axes.titlesize": TAMANHO_FONTE,
        "axes.labelsize": TAMANHO_FONTE,
        "xtick.labelsize": TAMANHO_FONTE,
        "ytick.labelsize": TAMANHO_FONTE,
        "legend.fontsize": TAMANHO_FONTE,
        "figure.dpi": 120,
    })

    # Mantém texto como texto no SVG (editável e com fonte preservada quando possível)
    plt.rcParams["svg.fonttype"] = "none"

    # Seaborn: fundo branco e sem grades
    sns.set_theme(style="white", rc={
        "font.family": "sans-serif",
        "font.sans-serif": [fonte_padrao],
        "font.size": TAMANHO_FONTE,
        "axes.grid": False,
    })

    return fonte_padrao


def salvar_svg(path_saida: Path) -> None:
    """Salva a figura atual como SVG com recorte justo."""
    plt.savefig(path_saida, format="svg", bbox_inches="tight")


def validar_colunas(df: pd.DataFrame, colunas: set[str]) -> None:
    """Valida se as colunas necessárias existem no DataFrame."""
    faltando = colunas - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas ausentes no arquivo de dados: {faltando}")


def criar_variaveis(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis de tempo, dummy pós-2023 e Brent convertido para R$."""
    df = df.copy()
    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data").reset_index(drop=True)

    df["tempo"] = range(1, len(df) + 1)
    df["pos_2023"] = (df["data"] >= DATA_BREAK).astype(int)
    df["tempo_pos"] = df["tempo"] * df["pos_2023"]

    # Brent convertido para reais (R$/barril)
    df["brent_rs"] = df["preco_brent"] * df["preco_dolar"]
    return df


def normalizar_0_1(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cria colunas normalizadas (0–1) para uma lista de variáveis."""
    df_norm = df.copy()
    for col in cols:
        denom = (df[col].max() - df[col].min())
        df_norm[f"{col}_norm"] = (df[col] - df[col].min()) / denom if denom != 0 else 0.0
    return df_norm


def tabela_como_figura_svg(df_tabela: pd.DataFrame, path_saida: Path, fonte: str) -> None:
    """Renderiza tabela como figura e salva em SVG."""
    fig, ax = plt.subplots(figsize=(12, 0.6 + 0.4 * len(df_tabela)))
    ax.axis("off")

    tabela_plot = ax.table(
        cellText=df_tabela.values,
        colLabels=df_tabela.columns,
        rowLabels=df_tabela.index,
        loc="center",
        cellLoc="center",
    )
    tabela_plot.scale(1, 1.5)

    for _, cell in tabela_plot.get_celld().items():
        cell.get_text().set_fontname(fonte)
        cell.get_text().set_fontsize(TAMANHO_FONTE)

    plt.tight_layout()
    plt.savefig(path_saida, format="svg", bbox_inches="tight")
    plt.close(fig)


def adicionar_r2_ajustado(tabela: pd.DataFrame, r2_adj: float) -> pd.DataFrame:
    """Adiciona uma linha de R² ajustado à tabela de coeficientes."""
    out = tabela.copy()
    nova_linha = {col: "" for col in out.columns}
    nova_linha[out.columns[-1]] = round(r2_adj, 4)
    out.loc["R² ajustado"] = nova_linha
    return out


# =============================================================================
# PLOTS
# =============================================================================
def plot_figura_1_series_nivel(df: pd.DataFrame) -> None:
    """Figura 1 — Séries temporais em nível (diesel, gasolina, câmbio e brent em eixo secundário)."""
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_xlabel("Data")
    ax1.set_ylabel("R$/L ou R$/US$")

    ax1.plot(df["data"], df["preco_diesel"], label="Diesel S10 (R$/L)", color="blue")
    ax1.plot(df["data"], df["preco_gasolina"], label="Gasolina A (R$/L)", color="orange")
    ax1.plot(df["data"], df["preco_dolar"], label="Câmbio (R$/US$)", color="red")

    ax1.axvline(
        pd.to_datetime(DATA_BREAK),
        color=COR_LINHA_POLITICA,
        linestyle="--",
        label="Mudança de Política (Mai/23)",
    )

    ax1.grid(False)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Brent (R$/barril)")
    ax2.plot(df["data"], df["brent_rs"], label="Brent (R$/barril)", color="green")
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    salvar_svg(PASTA_FIGURAS / "Figura_1_series_nivel.svg")
    plt.close(fig)


def plot_figura_2_series_normalizadas(df: pd.DataFrame) -> None:
    """Figura 2 — Séries normalizadas (0–1)."""
    df_norm = normalizar_0_1(df, ["preco_diesel", "preco_gasolina", "brent_rs", "preco_dolar"])

    fig = plt.figure(figsize=(14, 6))
    sns.lineplot(data=df_norm, x="data", y="preco_diesel_norm", label="Diesel S10 (R$/L)")
    sns.lineplot(data=df_norm, x="data", y="preco_gasolina_norm", label="Gasolina A (R$/L)")
    sns.lineplot(data=df_norm, x="data", y="brent_rs_norm", label="Brent (R$/barril)")
    sns.lineplot(data=df_norm, x="data", y="preco_dolar_norm", label="Câmbio (R$/US$)")

    plt.axvline(
        pd.to_datetime(DATA_BREAK),
        color=COR_LINHA_POLITICA,
        linestyle="--",
        label="Mudança de Política (Mai/23)",
    )

    plt.xlabel("Data")
    plt.ylabel("Valor normalizado (0–1)")
    plt.grid(False)
    plt.legend()

    plt.tight_layout()
    salvar_svg(PASTA_FIGURAS / "Figura_2_series_normalizadas.svg")
    plt.close(fig)


def plot_figura_3_boxplot_diesel(df: pd.DataFrame) -> None:
    """Figura 3 — Boxplot do Diesel (pré vs pós)."""
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="pos_2023", y="preco_diesel")
    plt.xticks([0, 1], ["Antes de Mai/23", "Depois de Mai/23"])
    plt.xlabel("Período")
    plt.ylabel("Preço do Diesel (R$/litro)")
    plt.grid(False)

    plt.tight_layout()
    salvar_svg(PASTA_FIGURAS / "Figura_3_boxplot_diesel.svg")
    plt.close(fig)


def plot_figura_4_boxplot_gasolina(df: pd.DataFrame) -> None:
    """Figura 4 — Boxplot da Gasolina (pré vs pós)."""
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="pos_2023", y="preco_gasolina")
    plt.xticks([0, 1], ["Antes de Mai/23", "Depois de Mai/23"])
    plt.xlabel("Período")
    plt.ylabel("Preço da Gasolina (R$/litro)")
    plt.grid(False)

    plt.tight_layout()
    salvar_svg(PASTA_FIGURAS / "Figura_4_boxplot_gasolina.svg")
    plt.close(fig)


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================
def main() -> None:
    preparar_pastas()
    fonte_usada = configurar_estilo()

    df = pd.read_excel(ARQUIVO_DADOS)

    validar_colunas(df, {
        "data",
        "preco_diesel",
        "preco_gasolina",
        "preco_brent",
        "preco_dolar",
    })

    df = criar_variaveis(df)

    # -------------------------
    # Modelos econométricos (OLS)
    # -------------------------
    modelo_gasolina = smf.ols(
        "preco_gasolina ~ tempo + pos_2023 + tempo_pos + brent_rs + preco_dolar",
        data=df,
    ).fit()

    modelo_diesel = smf.ols(
        "preco_diesel ~ tempo + pos_2023 + tempo_pos + brent_rs + preco_dolar",
        data=df,
    ).fit()

    # -------------------------
    # Figuras (SVG)
    # -------------------------
    plot_figura_1_series_nivel(df)
    plot_figura_2_series_normalizadas(df)
    plot_figura_3_boxplot_diesel(df)
    plot_figura_4_boxplot_gasolina(df)

    # -------------------------
    # Tabelas 1–3 (SVG)
    # -------------------------
    tabela1 = (
        df.groupby("pos_2023")[["preco_diesel", "preco_gasolina", "brent_rs", "preco_dolar"]]
        .agg(["mean", "std"])
        .round(3)
    )

    tabela2 = (
        df[df["pos_2023"] == 0][["preco_diesel", "preco_gasolina", "brent_rs", "preco_dolar"]]
        .corr()
        .round(3)
    )

    tabela3 = (
        df[df["pos_2023"] == 1][["preco_diesel", "preco_gasolina", "brent_rs", "preco_dolar"]]
        .corr()
        .round(3)
    )

    tabela_como_figura_svg(tabela1, PASTA_TABELAS / "Tabela_1_estatisticas_descritivas.svg", fonte_usada)
    tabela_como_figura_svg(tabela2, PASTA_TABELAS / "Tabela_2_correlacoes_pre.svg", fonte_usada)
    tabela_como_figura_svg(tabela3, PASTA_TABELAS / "Tabela_3_correlacoes_pos.svg", fonte_usada)

    # -------------------------
    # Tabelas 4–5 (Regressões + R² ajustado) (SVG)
    # -------------------------
    tabela4 = modelo_gasolina.summary2().tables[1].round(4)
    tabela5 = modelo_diesel.summary2().tables[1].round(4)

    tabela4 = adicionar_r2_ajustado(tabela4, modelo_gasolina.rsquared_adj)
    tabela5 = adicionar_r2_ajustado(tabela5, modelo_diesel.rsquared_adj)

    tabela_como_figura_svg(tabela4, PASTA_TABELAS / "Tabela_4_regressao_gasolina.svg", fonte_usada)
    tabela_como_figura_svg(tabela5, PASTA_TABELAS / "Tabela_5_regressao_diesel.svg", fonte_usada)

    # Feedback simples no console
    print("Exportação concluída.")
    print(f"- Figuras: {PASTA_FIGURAS.resolve()}")
    print(f"- Tabelas: {PASTA_TABELAS.resolve()}")
    print(f"- Fonte utilizada: {fonte_usada}")


if __name__ == "__main__":
    main()
