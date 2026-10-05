import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#GRAFICO DE INTERTEMPORADAS

# Lista exata das temporadas correspondentes às 14 janelas
temporadas = [
    "2016-17", "2017-18", "2018-19", "2019-20", 
    "2020-21", "2021-22", "2022-23", "2023-24"
]

# --- 1. DADOS DO EXPERIMENTO (XGBOOST) ---
dados = {
    "Temporada": temporadas,
    "Acuracia": [
        0.60,
        0.62,
        0.62,
        0.58,
        0.57,
        0.61,
        0.59,
        0.62,
    ],
    "F1_Score": [
        0.59,
        0.60,
        0.61,
        0.56,
        0.56,
        0.60,
        0.58,
        0.61,
    ],
}

df = pd.DataFrame(dados)

# --- 2. CONFIGURAÇÕES DO PLOT ---
plt.figure(figsize=(14, 7))

x_indices = np.arange(len(df["Temporada"]))

# 1. Plotar a linha de Acurácia - AZUL DESTAQUE
plt.plot(
    x_indices,
    df["Acuracia"],
    label="Acurácia XGBOOST",
    color="#1f77b4",
    linewidth=3.0,
    marker="o",
    zorder=3,
)

# 2. Plotar a linha de F1-Score - VERMELHO CLARO OPCO (Fundo)
plt.plot(
    x_indices,
    df["F1_Score"],
    label="F1-Score XGBOOST (Tendência)",
    color="indianred",
    linewidth=1.8,
    alpha=0.25,  # Deixa a linha bem suave e em segundo plano
    zorder=2,
)

# --- 3. AJUSTES DE EIXO E LIMITES ---
plt.xticks(x_indices, df["Temporada"], rotation=45)
plt.xlim(-0.5, len(df["Temporada"]) - 0.5)
plt.ylim(0.40, 1.0)

# Customização do gráfico
plt.title(
    "Evolução de Desempenho: Inter-temporadas - XGBOOST",
    fontsize=14,
    pad=15,
)
plt.xlabel("Temporadas de Teste (Crescentes em Volume de Treino)", fontsize=12)
plt.ylabel("Índice da Métrica", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.5)

# Legenda formatada
plt.legend(loc="lower left", fontsize=10, shadow=True, frameon=True)

# --- 4. ANOTAÇÕES DE DESTAQUE ATUALIZADAS ---
# Temporada 2011-2012 (Maior Pico - índice 1)
plt.annotate(
    "Pico Inicial: 0.75",
    xy=(1, 0.75),
    xytext=(1.5, 0.80),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="black"),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
)

# Temporada 2022-2023 (Estabilização - índice 11)
plt.annotate(
    "Estabilização: 0.72",
    xy=(11, 0.72),
    xytext=(11.5, 0.80),
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=-0.2", color="black"
    ),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
)

plt.tight_layout()

# --- 5. SALVAMENTO ---
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
plot_dir = os.path.join(base_path, "results", "graficos", "graficos nba", "graficos intertemporadas")
os.makedirs(plot_dir, exist_ok=True)

plot_filename = os.path.join(plot_dir, "grafico_intertemporadas_xgboost.png")
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
print(f">> Gráfico limpo salvo com sucesso em: {plot_filename}")

plt.close()