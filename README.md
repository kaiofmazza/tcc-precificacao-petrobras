# Código do TCC – Precificação de Combustíveis (Petrobras)

Este repositório contém o código utilizado no Trabalho de Conclusão de Curso (TCC) do MBA em Data Science & Analytics (USP/ESALQ), cujo objetivo é analisar os efeitos da mudança na política de preços da Petrobras, ocorrida em maio de 2023, sobre os preços da Gasolina A e do Diesel S10 nas refinarias.

A análise emprega modelos de séries temporais com intervenção e quebra estrutural, controlando para os efeitos do preço internacional do petróleo Brent e da taxa de câmbio.

## Conteúdo do repositório
- `tcc_modelo_quebra.py`: script principal com a estimação dos modelos econométricos, geração de tabelas e figuras.

## Requisitos
- Python 3.11 ou superior
- Bibliotecas: pandas, statsmodels, matplotlib, seaborn

## Execução
O script pode ser executado diretamente após a instalação das dependências mencionadas, assumindo que os dados já tenham sido previamente organizados conforme descrito no trabalho.

## Observação sobre os dados
Os dados utilizados na pesquisa foram construídos a partir de fontes públicas e institucionais (Petrobras, Federal Reserve Bank of St. Louis – FRED, e IPEADATA - IPEA), conforme detalhado no TCC. Por questões de organização e reprodutibilidade conceitual, os arquivos de dados não estão incluídos neste repositório.
