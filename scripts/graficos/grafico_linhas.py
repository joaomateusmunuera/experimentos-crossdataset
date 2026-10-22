import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# 1. Mapeamento Cronológico das Temporadas
mapa_temporadas = {
    'janela_01': '2009-2010', 'janela_02': '2011-2012', 'janela_03': '2012-2013',
    'janela_04': '2013-2014', 'janela_05': '2014-2015', 'janela_06': '2015-2016',
    'janela_07': '2016-2017', 'janela_08': '2018-2019', 'janela_09': '2019-2020',
    'janela_10': '2020-2021', 'janela_11': '2021-2022', 'janela_12': '2022-2023',
    'janela_13': '2023-2024', 'janela_14': '2024-2025'
}

# 2. Configurações Visuais e Temáticas
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# A estrutura agora recebe a subpasta exata onde os CSVs estão localizados
config_experimentos = {
 #   'nba_baseline': {
 #       'pasta': os.path.join('nba', 'nba_experimento_04_minuto'),
 #       'label': 'Treino NBA e Teste NBA', 'cor': '#0047AB', 'alpha': 0.35, 'lw': 1.5, 'marker': 'o' # Azul Cobalto
#    },
#    'nbb_baseline': {
#        'pasta': os.path.join('nbb', 'intertemporadas'),
#        'label': 'Treino NBB e Teste NBB', 'cor': '#008000', 'alpha': 0.35, 'lw': 1.5, 'marker': 'o' # Verde Bandeira
#    },
    'cross_nba_nbb': {
        'pasta': 'treina_nba_testa_nbb',
        'label': 'Treino NBA e Teste NBB', 'cor': '#4a8520', 'alpha': 1.0, 'lw': 1.5, 'marker': 'o' # Vermelho Carmesim
    },
    'cross_nbb_nba': {
        'pasta': 'treina_nbb_testa_nba',
        'label': 'Treino NBB e Teste NBA', 'cor': '#6A0DAD', 'alpha': 1.0, 'lw': 1.5, 'marker': 'o' # Roxo
    },
    'misto_nba': {
        'pasta': 'treino_misto_nba',
        'label': 'NBA + NBB e Teste NBA', 'cor': '#17A2B8', 'alpha': 1.0, 'lw': 1.5, 'marker': 's' # Teal / Ciano Escuro
    },
    'misto_nbb': {
        'pasta': 'treino_misto_nbb',
        'label': 'NBA + NBB e Teste NBB', 'cor': '#FF8C00', 'alpha': 1.0, 'lw': 1.5, 'marker': 's' # Laranja Escuro
    },
    'fine_tuning': {
        'pasta': 'fine_tuning',
        'label': 'Fine-Tuning (Teste NBB)', 'cor': '#000000', 'alpha': 1.0, 'lw': 2.5, 'marker': 'D' # Preto Sólido
    }
}

# 3. Definição de Caminhos Relativos
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
results_dir = os.path.join(project_root, 'results')
output_dir = os.path.join(results_dir, 'graficos', 'grafico_linhas')
os.makedirs(output_dir, exist_ok=True)

classificadores = ['random_forest']

def gerar_grafico_linhas(classificador):
    fig, ax = plt.subplots(figsize=(14, 7))
    linhas_plotadas = 0
    
    for key, config in config_experimentos.items():
        caminho_pasta = os.path.join(results_dir, config['pasta'])
        
        if not os.path.exists(caminho_pasta):
            print(f"Diretório não encontrado: {caminho_pasta}")
            continue
            
        padrao_busca = os.path.join(caminho_pasta, f"*{classificador}*.csv")
        arquivos = glob.glob(padrao_busca)
        
        if not arquivos:
            continue
            
        arquivo_csv = arquivos[0]
        try:
            df = pd.read_csv(arquivo_csv)
            
            if 'Janela' not in df.columns or 'F1-Score' not in df.columns:
                continue
                
            df_limpo = df[df['Janela'] != 'Média Geral'].copy()
            df_limpo['F1-Score'] = pd.to_numeric(df_limpo['F1-Score'].astype(str).str.replace(',', '.'), errors='coerce')
            
            # Aplica o mapeamento das temporadas
            df_limpo['Temporada'] = df_limpo['Janela'].map(mapa_temporadas)
            
            # Remove linhas onde a janela não foi encontrada no mapeamento
            df_limpo = df_limpo.dropna(subset=['Temporada'])
            
            ax.plot(
                df_limpo['Temporada'], 
                df_limpo['F1-Score'], 
                label=config['label'],
                color=config['cor'], 
                alpha=config['alpha'], 
                linewidth=config['lw'], 
                marker=config['marker'],
                markersize=6
            )
            linhas_plotadas += 1
            
        except Exception as e:
            print(f"Erro ao ler {arquivo_csv}: {e}")
            
    if linhas_plotadas == 0:
        print(f"Nenhum dado plotado para o classificador: {classificador.upper()}")
        plt.close(fig)
        return

    # 4. Formatação Estética do Gráfico
    ax.set_title(f"Comparação entre abordagens — {classificador.upper()}", 
                 fontsize=14, pad=15)
    
    ax.set_ylabel('F1-Score', fontsize=10)
    ax.set_xlabel('Temporadas de Teste Cronológicas', fontsize=10, labelpad=10)
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    ax.set_ylim(0.20, 0.95)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(1.0)
    
    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    ax.legend(loc='upper right', shadow=True, fancybox=True, framealpha=1.0, fontsize=9)
    
    plt.tight_layout()
    
    caminho_salvamento = os.path.join(output_dir, f"grafico_linhas_{classificador}.png")
    plt.savefig(caminho_salvamento, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico gerado com sucesso: {caminho_salvamento}")

for clf in classificadores:
    gerar_grafico_linhas(clf)