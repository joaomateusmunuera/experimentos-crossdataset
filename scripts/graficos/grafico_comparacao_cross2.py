import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. CONFIGURAÇÃO DE CAMINHOS E TEMPORADAS ---
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Modelos que serão gerados
modelos = ["xgboost", "regressao_logistica"]

# Rótulos do eixo X (Temporadas de Teste Cronológicas)
temporadas_teste = [
    "2009-2010", "2011-2012", "2012-2013", "2013-2014", 
    "2014-2015", "2015-2016", "2016-2017", "2018-2019", 
    "2019-2020", "2020-2021", "2021-2022", "2022-2023", 
    "2023-2024", "2024-2025"
]

# Diretório de salvamento das imagens
plot_dir = os.path.join(base_path, "results", "graficos", "graficos_cross_dataset")
os.makedirs(plot_dir, exist_ok=True)

for modelo in modelos:
    print(f"\n>> Processando gráfico triplo para o modelo: {modelo.upper()}")
    
    # --- 2. LEITURA DOS DADOS ---
    
    # A) Treino NBA -> Teste NBB (Centro - Laranja)
    path_nba_nbb = os.path.join(base_path, "results", "cross_dataset_minuto_nba_nbb", f"{modelo}_cross_dataset_15_minuto.csv")
    
    # B) Treino NBB -> Teste NBA (Direita - Azul)
    path_nbb_nba = os.path.join(base_path, "results", "cross_dataset_minuto_nbb_nba", f"{modelo}_cross_dataset_15_minuto.csv")
    
    # C) Treino NBB -> Teste NBB (Esquerda - Verde)
    path_nbb_nbb = os.path.join(base_path, "results", "nbb", "intertemporadas", f"{modelo}_experimento_04_15_minuto.csv")

    def extrair_f1(caminho):
        if os.path.exists(caminho):
            df = pd.read_csv(caminho)
            df_jan = df[df["Janela"].str.startswith("janela_")].copy()
            return df_jan["F1-Score"].astype(float).tolist()
        else:
            print(f"  [!] Arquivo não encontrado: {caminho}")
            return []

    f1_nba_nbb = extrair_f1(path_nba_nbb)
    f1_nbb_nba = extrair_f1(path_nbb_nba)
    f1_nbb_nbb = extrair_f1(path_nbb_nbb)

    tamanhos = [len(f1_nba_nbb), len(f1_nbb_nba), len(f1_nbb_nbb)]
    if 0 in tamanhos:
        print("   [X] Faltam dados em uma das pastas para gerar o gráfico. Pulando modelo.")
        continue
        
    tamanho_minimo = min(tamanhos)
    f1_nba_nbb = f1_nba_nbb[:tamanho_minimo]
    f1_nbb_nba = f1_nbb_nba[:tamanho_minimo]
    f1_nbb_nbb = f1_nbb_nbb[:tamanho_minimo]

    # --- 3. CONFIGURAÇÕES DO GRÁFICO DE BARRAS ---
    labels = temporadas_teste[:tamanho_minimo]
    x = np.arange(len(labels))
    width = 0.25  

    # Tamanho igual ao do script fornecido para maior espaçamento
    fig, ax = plt.subplots(figsize=(18, 8))

    # Barra 1 (Esquerda - Verde): Treino NBB -> Teste NBB
    rects1 = ax.bar(x - width, f1_nbb_nbb, width, label="Treino NBB e Teste NBB (Controle)", color="#2ca02c")
    
    # Barra 2 (Centro - Laranja): Treino NBA -> Teste NBB
    rects2 = ax.bar(x, f1_nba_nbb, width, label="Treino NBA e Teste NBB", color="#ff7f0e")
    
    # Barra 3 (Direita - Azul): Treino NBB -> Teste NBA
    rects3 = ax.bar(x + width, f1_nbb_nba, width, label="Treino NBB e Teste NBA", color="#1f77b4")

    # --- 4. CUSTOMIZAÇÃO VISUAL ---
    ax.set_title(
        f"Comparação crossdataset e configuração padrão ({modelo.upper()})",
        fontsize=15,
        pad=15,
    )
    ax.set_xlabel("Temporadas de Teste Cronológicas", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_xticks(x)
    
    # ha="right" adicionado para alinhar perfeitamente com a inclinação de 45 graus
    ax.set_xticklabels(labels, rotation=45, ha="right")
    
    # Limites ajustados para dar espaço aos números no topo
    ax.set_ylim(0.20, 0.90) 
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    ax.legend(loc="upper right", fontsize=11, shadow=True)

    # --- 5. RÓTULOS FORMATADOS ---
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
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

    # --- 6. SALVAMENTO AUTOMÁTICO ---
    modelo_limpo = modelo.strip().replace('\n', '').replace('\r', '')
    plot_filename = os.path.join(plot_dir, f"comparacao_tripla_{modelo_limpo}.png")
    
    time.sleep(1.0) 
    
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"   [OK] Gráfico salvo em: {plot_filename}")
    except OSError:
        print(f"   [AVISO] Erro com bbox_inches='tight'. Tentando salvamento padrão...")
        plt.savefig(plot_filename, dpi=300)
        print(f"   [OK] Gráfico salvo em: {plot_filename}")

    plt.close()

print("\n>> Geração de gráficos triplos finalizada com sucesso!")