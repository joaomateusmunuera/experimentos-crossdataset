import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURAÇÃO DE CAMINHOS E TEMPORADAS ---
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Modelos que serão gerados
modelos = ["xgboost", "regressao_logistica"]

# Rótulos do eixo X (Temporadas de Teste Cronológicas)
# Nota: Ignoramos 2008-2009 pois ela foi o Treino 01. O teste começa em 2009-2010.
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
    print(f"\n>> Processando gráfico para o modelo: {modelo.upper()}")
    
    # --- 2. LEITURA DOS DADOS ---
    
    # A) Treino NBA -> Teste NBB (Laranja)
    path_nba_nbb = os.path.join(
        base_path, "results", "cross_dataset_minuto_nba_nbb", 
        f"{modelo}_cross_dataset_15_minuto.csv"
    )
    
    # B) Treino NBB -> Teste NBA (Azul)
    path_nbb_nba = os.path.join(
        base_path, "results", "cross_dataset_minuto_nbb_nba", 
        f"{modelo}_cross_dataset_15_minuto.csv"
    )

    f1_nba_nbb = []
    f1_nbb_nba = []

    # Leitura do Experimento NBA -> NBB
    if os.path.exists(path_nba_nbb):
        df_nba_nbb = pd.read_csv(path_nba_nbb)
        df_janelas = df_nba_nbb[df_nba_nbb["Janela"].str.startswith("janela_")].copy()
        f1_nba_nbb = df_janelas["F1-Score"].astype(float).tolist()
    else:
        print(f"[!] Arquivo não encontrado: {path_nba_nbb}")

    # Leitura do Experimento NBB -> NBA
    if os.path.exists(path_nbb_nba):
        df_nbb_nba = pd.read_csv(path_nbb_nba)
        df_janelas2 = df_nbb_nba[df_nbb_nba["Janela"].str.startswith("janela_")].copy()
        f1_nbb_nba = df_janelas2["F1-Score"].astype(float).tolist()
    else:
        print(f"[!] Arquivo não encontrado: {path_nbb_nba}")

    if not f1_nba_nbb or not f1_nbb_nba:
        print("   [X] Dados insuficientes para gerar o gráfico. Pulando modelo.")
        continue

    # --- 3. CONFIGURAÇÕES DO GRÁFICO DE BARRAS ---
    labels = temporadas_teste[:len(f1_nba_nbb)] # Garante que as labels batem com o número de janelas
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 7))

    # Barras 1: Treino NBA -> Teste NBB (ESQUERDA - Laranja)
    rects1 = ax.bar(
        x - width / 2,
        f1_nba_nbb,
        width,
        label="Treino NBA -> Teste NBB (F1-Score)",
        color="#ff7f0e",
    )

    # Barras 2: Treino NBB -> Teste NBA (DIREITA - Azul)
    rects2 = ax.bar(
        x + width / 2,
        f1_nbb_nba,
        width,
        label="Treino NBB -> Teste NBA (F1-Score)",
        color="#1f77b4",
    )

    # --- 4. CUSTOMIZAÇÃO VISUAL (Idêntica ao anexo original) ---
    ax.set_title(
        f"Comparação F1-Score Experimento Intertemporadas Cross-dataset ({modelo.upper()})",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("Temporadas de Teste Cronológicas", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylim(0.20, 0.85) # Congela o eixo Y igual ao seu exemplo
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    ax.legend(loc="upper right", fontsize=11, shadow=True)

    # Rótulos de valor em cima de cada barra (Horizontais e com 4 casas decimais)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()

    # --- 5. SALVAMENTO AUTOMÁTICO (COM BLINDAGEM CONTRA ERROS) ---
    
    # 1. Higieniza a string para evitar qualquer caractere oculto (ex: \n, \r)
    modelo_limpo = modelo.strip().replace('\n', '').replace('\r', '')
    plot_filename = os.path.join(plot_dir, f"comparacao_cross_dataset_{modelo_limpo}.png")
    
    # 2. Pausa curta de 1 segundo para evitar bloqueio do OneDrive/Antivírus
    import time
    time.sleep(1.0) 
    
    try:
        # Tentativa primária com o corte de margens ativado
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        print(f"   [OK] Gráfico salvo em: {plot_filename}")
    except OSError as e:
        # Se o Windows ou o bbox_inches recusarem, tenta o salvamento padrão seguro
        print(f"   [AVISO] Erro com bbox_inches='tight'. Tentando salvamento padrão...")
        plt.savefig(plot_filename, dpi=300)
        print(f"   [OK] Gráfico salvo (plano de contingência) em: {plot_filename}")

    plt.close()

    print("\n>> Geração de gráficos finalizada com sucesso!")