import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURAÇÃO DE CAMINHOS E MODELO ---
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))

# Defina qual modelo você quer plotar ("xgboost", "regressao_logistica", "svm", etc)
modelo_nome = "regressao_logistica"

# Defina em qual temporada o gráfico deve parar
temporada_limite = "2022-2023"

# Lista de temporadas de teste (Rótulos do Eixo X)
temporadas = [
    "2009-2010", "2010-2011", "2011-2012", "2012-2013", "2013-2014",
    "2014-2015", "2015-2016", "2016-2017", "2017-2018", "2018-2019",
    "2019-2020", "2020-2021", "2021-2022", "2022-2023"
]

# Corta a lista de temporadas no limite definido para não plotar o que está vazio
if temporada_limite in temporadas:
    idx_limite = temporadas.index(temporada_limite)
    temporadas = temporadas[:idx_limite + 1]

f1_inter_temporadas = []
f1_combinatoria = []
f1_acumulada = []

# --- 2. LEITURA AUTOMÁTICA DOS DADOS ---

# A) Experimento 04: Janela Inter-temporadas
path_inter = os.path.join(base_path, "results", "nba_experimento_04_minuto", f"{modelo_nome}_experimento_04_15_minuto.csv")

if os.path.exists(path_inter):
    df_inter = pd.read_csv(path_inter)
    df_janelas = df_inter[df_inter["Janela"].str.startswith("janela_")].copy()
    valores_salvos = df_janelas["F1-Score"].astype(float).tolist()
    
    # Preenche com 0.0 as janelas faltantes para não quebrar o gráfico
    for i in range(len(temporadas)):
        if i < len(valores_salvos):
            f1_inter_temporadas.append(valores_salvos[i])
        else:
            f1_inter_temporadas.append(0.0)
            
    print(f">> Experimento 04 lido: {len(valores_salvos)} janelas reais (Preenchido até {len(temporadas)}).")
else:
    print(f"[!] Arquivo do Experimento 04 não encontrado: {path_inter}")
    f1_inter_temporadas = [0.0] * len(temporadas)

# B) Experimento 05: Janela Combinatória (Inter-temporadas + Incremental)
dir_exp05 = os.path.join(base_path, "results", "nba_experimento_05_minuto")

for temp in temporadas:
    temp_arquivo = temp[:5] + temp[-2:]
    arquivo_temp = os.path.join(dir_exp05, f"evolucao_{modelo_nome}_{temp_arquivo}_K1_min.csv")
    
    if os.path.exists(arquivo_temp):
        df_temp = pd.read_csv(arquivo_temp)
        if "F1_Score_Acumulado" in df_temp.columns:
            ultimo_f1 = df_temp["F1_Score_Acumulado"].iloc[-1]
        else:
            ultimo_f1 = df_temp.iloc[-1, -1]
            
        f1_combinatoria.append(float(ultimo_f1))
    else:
        print(f"[!] Exp 05 não encontrado para {temp}: {arquivo_temp}")
        f1_combinatoria.append(0.0)

# C) Experimento 03_2: Acurácia Acumulada
dir_exp03 = os.path.join(base_path, "results", "nba_experimento_03_minuto")

for temp in temporadas:
    temp_arquivo = temp[:5] + temp[-2:]
    arquivo_temp = os.path.join(dir_exp03, f"evolucao_{modelo_nome}_{temp_arquivo}_K1.csv")
    
    if os.path.exists(arquivo_temp):
        df_temp = pd.read_csv(arquivo_temp)
        if "F1_Score_Acumulado" in df_temp.columns:
            ultimo_f1 = df_temp["F1_Score_Acumulado"].iloc[-1]
        else:
            ultimo_f1 = df_temp.iloc[-1, -1]
            
        f1_acumulada.append(float(ultimo_f1))
    else:
        print(f"[!] Exp 03_2 não encontrado para {temp}: {arquivo_temp}")
        f1_acumulada.append(0.0)

# --- 3. CONFIGURAÇÕES DO GRÁFICO DE BARRAS ---

# Dicionário mapeando a temporada para a quantidade total de jogos
qtd_jogos_nba = {
    "2009-2010": 1230,
    "2010-2011": 1230,
    "2011-2012": 990,
    "2012-2013": 1229,
    "2013-2014": 1230,
    "2014-2015": 1230,
    "2015-2016": 1230,
    "2016-2017": 1230,
    "2017-2018": 1230,
    "2018-2019": 1230,
    "2019-2020": 1059,
    "2020-2021": 1080,
    "2021-2022": 1230,
    "2022-2023": 1230,
    "2023-2024": 1230
}

# Cria os rótulos unindo o nome da temporada, quebra de linha (\n) e a quantidade de jogos
labels = [f"{temp}\n({qtd_jogos_nba.get(temp, '?')} jogos)" for temp in temporadas]

x = np.arange(len(labels))
width = 0.25 

fig, ax = plt.subplots(figsize=(18, 8))

# Barras 1: Experimento 03_2 - Acurácia Acumulada (ESQUERDA - Verde)
rects1 = ax.bar(x - width, f1_acumulada, width, label="Acurácia Acumulada (Exp. 03)", color="#2ca02c")
# Barras 2: Experimento 04 - Inter-temporadas (CENTRO - Laranja)
rects2 = ax.bar(x, f1_inter_temporadas, width, label="Janela Inter-temporadas (Exp. 04)", color="#ff7f0e")
# Barras 3: Experimento 05 - Janela Combinatória (DIREITA - Azul)
rects3 = ax.bar(x + width, f1_combinatoria, width, label="Janela Combinatória (Exp. 05)", color="#1f77b4")

# --- 4. CUSTOMIZAÇÃO VISUAL ---
ax.set_title(
    f"Comparação de F1-Score: Acurácia Acumulada vs Inter-temporadas vs Combinatória ({modelo_nome.upper()})",
    fontsize=15,
    pad=15,
)
ax.set_xlabel("Temporadas de Teste Cronológicas", fontsize=12)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
# MUDANÇA: Limite máximo do eixo Y reduzido para 0.75
ax.set_ylim(0.40, 0.75)
ax.grid(True, linestyle=":", alpha=0.5, axis="y")

ax.legend(loc="upper right", fontsize=11, shadow=True)

# Rótulos com os valores formatados em cima de cada barra
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # Não exibe o texto se o valor for 0.0 (para as barras faltantes)
        if height > 0:
            ax.annotate(
                f"{height:.3f}", 
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), 
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8, 
                rotation=45 
            )

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()

# --- 5. SALVAMENTO AUTOMÁTICO ---
plot_dir = os.path.join(base_path, "results", "graficos_combinatorio", f"{modelo_nome}")
os.makedirs(plot_dir, exist_ok=True)

plot_filename = os.path.join(plot_dir, f"comparacao_f1_score_exp03_vs_04_vs_05_{modelo_nome}.png")
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
print(f"\n>> Gráfico comparativo triplo salvo com sucesso em:\n   {plot_filename}")

plt.close()