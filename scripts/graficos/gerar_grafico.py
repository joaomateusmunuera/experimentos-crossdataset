import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

#GRAFICO JANELA ACURACIA ACUMULADA

# --- CONFIGURAÇÕES ---
modelos = ['xgboost']
temporadas = [
    "2016-17", "2017-18", "2018-19", "2019-20", 
    "2020-21", "2021-22", "2022-23", "2023-24"
]

k = 1
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

for modelo in modelos:
    for temporada in temporadas:
        # Caminhos dos arquivos CSV gerados pelo experimento_03_2.py
   #    path_grid = os.path.join(base_path, 'results', 'experimento_03_2', f'evolucao_{modelo}_{temporada}_K{k}.csv')
        path_grid = os.path.join(base_path, 'results', 'nba_experimento_acuracia_acumulada_min', f'evolucao_{modelo}_{temporada}_K{k}.csv')
        path_vanilla = os.path.join(base_path, 'results', 'nba_experimento_acuracia_acumulada_min', f'evolucao_vanilla_{temporada}_K{k}.csv')

        if os.path.exists(path_grid) and os.path.exists(path_vanilla):
            df_grid = pd.read_csv(path_grid)
            df_vanilla = pd.read_csv(path_vanilla)

            plt.figure(figsize=(14, 7))
            
            # 1. Plotar a linha do modelo - AZUL
            plt.plot(df_grid['Jogo_Real_n'], df_grid['F1_Score_Acumulado'], 
                     label=f'F1-Score {modelo.upper()}', 
                     color='#1f77b4', linewidth=2.5)

            # 3. Plotar a linha do F1-Score Vanilla - VERMELHO CLARO PONTILHADO

            if 'F1_Score_Acumulado' in df_vanilla.columns:
                plt.plot(df_vanilla['Jogo_Real_n'], df_vanilla['F1_Score_Acumulado'], 
                         label='F1-Score Vanilla', 
                         color='indianred',linewidth=2, alpha=0.7)

            # --- AJUSTES DE EIXO E LIMITES ---
            max_jogos = df_grid['Jogo_Real_n'].max()
            
            # Saltando de 100 em 100 jogos para limpar a visualização da NBA
            plt.xticks(np.arange(0, max_jogos + 100, 100))
            
            # Expandimos o X em +80 para criar espaço para as caixas de texto à direita
            plt.xlim(-5, max_jogos + 80) 
            plt.ylim(0.20, 1.0) # <--- ALTERADO: Limite alterado de 0.85 para 1.0

            # Customização do gráfico
            plt.title(f'Comparação de Evolução: Vanilla vs {modelo.upper()} - {temporada}', fontsize=14, pad=15)
            plt.xlabel('Número do Jogo na Temporada', fontsize=12)
            plt.ylabel('Métrica Acumulada', fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.6)
            
            # Legenda movida para a esquerda para não obstruir as setas na direita
            plt.legend(loc='lower right', fontsize=10, shadow=True, frameon=True)

            # --- ANOTAÇÕES COM PILHAMENTO FIXO (ORDEM DE ESCADA) ---
            final_jogo = df_grid['Jogo_Real_n'].iloc[-1]
            espaco_lateral = 12

            # 1. Modelo Principal (Sempre no topo da escada)
            final_acc_grid = df_grid['F1_Score_Acumulado'].iloc[-1]
            plt.annotate(f'Final {modelo.upper()}: {final_acc_grid:.4f}', 
                        xy=(final_jogo, final_acc_grid), 
                        xytext=(final_jogo + espaco_lateral, 0.85), # <--- RECOMENDADO: Subi de 0.70 para 0.85 para acompanhar a nova escala do gráfico
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='black'),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # 3. Vanilla F1 (Sempre na base da escada)
            if 'F1_Score_Acumulado' in df_vanilla.columns:
                final_f1_vanilla = df_vanilla['F1_Score_Acumulado'].iloc[-1]
                plt.annotate(f'Final F1 Vanilla: {final_f1_vanilla:.4f}', 
                            xy=(final_jogo, final_f1_vanilla), 
                            xytext=(final_jogo + espaco_lateral, 0.45), # Mantido na base para dar um bom distanciamento visual
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='indianred'),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="indianred", alpha=0.6))

            plt.tight_layout()

            # --- SALVAMENTO ---
            plot_dir = os.path.join(base_path, 'results', 'graficos','graficos nba')
            os.makedirs(plot_dir, exist_ok=True) 

            plot_filename = os.path.join(plot_dir, f'grafico_{modelo}_{temporada}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f">> Gráfico salvo: {plot_filename}")
            
            plt.close()
        else:
            if not os.path.exists(path_grid):
                print(f"[-] Arquivo Grid faltando: {path_grid}")
            if not os.path.exists(path_vanilla):
                print(f"[-] Arquivo Vanilla faltando: {path_vanilla}")

print("\n--- Todos os gráficos foram gerados com sucesso ---")