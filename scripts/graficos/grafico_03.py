import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#GRAFICO ACUMULADA X INTERTEMPORADAS

# --- 1. CONFIGURAÇÃO DE CAMINHOS E TEMPORADAS ---
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))

# Lista de temporadas testadas (serão usadas diretamente como RÓTULOS do eixo X)
temporadas_acumulada = [
    "2016-17", "2017-18", "2018-19", "2019-20", 
    "2020-21", "2021-22", "2022-23", "2023-24"
]

f1_inter_temporadas = []
f1_acuracia_acumulada = []

# --- 2. LEITURA AUTOMÁTICA DOS DADOS ---

# A) Extração do Experimento Inter-temporadas (Experimento 04)
path_inter = os.path.join(base_path, "results", "nba" ,"nba_experimento_04_minuto", "xgboost_experimento_04_15_minuto.csv")

if os.path.exists(path_inter):
    df_inter = pd.read_csv(path_inter)
    # Filtra apenas as 14 janelas (descarta 'Média Geral')
    df_janelas = df_inter[df_inter["Janela"].str.startswith("janela_")].copy()
    f1_inter_temporadas = df_janelas["F1-Score"].astype(float).tolist()
    print(f">> F1-Score Inter-temporadas lido com sucesso ({len(f1_inter_temporadas)} janelas).")
else:
    print(f"[!] Arquivo não encontrado: {path_inter}")

# B) Extração do Experimento de Acurácia Acumulada (Experimento 03_2)
dir_acumulada = os.path.join(base_path, "results","nba","nba_experimento_acuracia_acumulada_min")

for temp in temporadas_acumulada:
    arquivo_temp = os.path.join(dir_acumulada, f"evolucao_xgboost_{temp}_K1.csv")
    
    if os.path.exists(arquivo_temp):
        df_temp = pd.read_csv(arquivo_temp)
        # Pega o valor da última linha na coluna de F1-Score
        if "F1_Score_Acumulado" in df_temp.columns:
            ultimo_f1 = df_temp["F1_Score_Acumulado"].iloc[-1]
        else:
            ultimo_f1 = df_temp.iloc[-1, -1]
            
        f1_acuracia_acumulada.append(float(ultimo_f1))
    else:
        print(f"[!] Arquivo de evolução não encontrado para {temp}: {arquivo_temp}")
        f1_acuracia_acumulada.append(0.0)

# --- 3. CONFIGURAÇÕES DO GRÁFICO DE BARRAS ---
labels = temporadas_acumulada  # RÓTULOS AGORA SÃO AS TEMPORADAS
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(15, 7))

# Barras 1: Janela Acurácia Acumulada (ESQUERDA - Laranja)
rects1 = ax.bar(
    x - width / 2,
    f1_acuracia_acumulada,
    width,
    label="Janela Acurácia Acumulada (F1-Score)",
    color="#ff7f0e",
)

# Barras 2: Janela Inter-temporadas (DIREITA - Azul)
rects2 = ax.bar(
    x + width / 2,
    f1_inter_temporadas,
    width,
    label="Janela Inter-temporadas (F1-Score)",
    color="#1f77b4",
)

# --- 4. CUSTOMIZAÇÃO VISUAL ---
ax.set_title(
    "Comparação de F1-Score por Temporada de Teste: Acumulada vs Inter-temporadas (XGBOOST)",
    fontsize=14,
    pad=15,
)
ax.set_xlabel("Temporadas de Teste Cronológicas", fontsize=12)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.set_ylim(0.40, 0.85)
ax.grid(True, linestyle=":", alpha=0.5, axis="y")

ax.legend(loc="upper right", fontsize=11, shadow=True)


# Rótulos de valor em cima de cada barra
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.4f}", # Alterado para 3 casas decimais
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
        )


autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

# --- 5. SALVAMENTO AUTOMÁTICO ---
plot_dir = os.path.join(base_path, "results", "graficos", "graficos nba", "graficos comparacao")
os.makedirs(plot_dir, exist_ok=True)

plot_filename = os.path.join(plot_dir, "comparacao_f1_score_xgboost.png")
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
print(f"\n>> Gráfico comparativo de F1-Score com temporadas salvo em:\n   {plot_filename}")

plt.close()