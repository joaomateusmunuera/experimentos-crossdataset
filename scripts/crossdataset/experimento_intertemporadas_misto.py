import os
import sys
import time
import pandas as pd
import numpy as np

# Adiciona a pasta pai 'scripts/' ao caminho do Python
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(current_dir, '..'))
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# Importações dos classificadores
from xgboost_model import run_model_xgboost
from rede_neural import run_model_rede_neural
from regressao_logistica_model import run_model_logreg
from random_forest import run_model_rf
from experimentos import save_results_csv



# Configurações do Experimento Misto (Joint Training)
modelos = ['random_forest']
jogos_media = '15' 
janelas = [f"janela_{i:02d}" for i in range(1, 15)]

base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
start_time = time.time()

for modelo in modelos:
    results = [] 
    acuracias_do_experimento = []
    f1_scores_do_experimento = []

    for janela in janelas:
        print(f"\nRodando Treinamento Misto: {modelo.upper()} | {janela}")
        
        path_treino_nba = os.path.join(base_path, "data", "cross_dataset_nba_para_nbb", jogos_media, janela, "treino_nba.csv")
        path_treino_nbb = os.path.join(base_path, "data", "cross_dataset_nbb_para_nba", jogos_media, janela, "treino_nbb.csv")
        path_teste_nbb  = os.path.join(base_path, "data", "cross_dataset_nba_para_nbb", jogos_media, janela, "teste_nbb.csv")
     #   path_teste_nbb  = os.path.join(base_path, "data", "cross_dataset_nbb_para_nba", jogos_media, janela, "teste_nba.csv")

        accuracy = None
        f1 = None

        if os.path.exists(path_treino_nba) and os.path.exists(path_treino_nbb) and os.path.exists(path_teste_nbb):
            
            # --- O PULO DO GATO: CONCATENAÇÃO BALANCEADA ---
            df_nba = pd.read_csv(path_treino_nba)
            df_nbb = pd.read_csv(path_treino_nbb)
            
            # Limita a NBA às 4 temporadas mais recentes relativas a esta janela
            # 4 temporadas * ~1230 jogos = 4920 jogos
            limite_jogos_nba = 4920
            
            if len(df_nba) > limite_jogos_nba:
                df_nba = df_nba.tail(limite_jogos_nba) # Pega apenas o final (mais recentes)
                
            # O NBB deixamos inteiro ou limitamos às últimas 10 (se quiser limitar, 10 * ~250 = 2500)
            limite_jogos_nbb = 2500
            if len(df_nbb) > limite_jogos_nbb:
               df_nbb = df_nbb.tail(limite_jogos_nbb)
            
            # Junta as linhas da NBA (reduzida) e do NBB num único DataFrame
            df_misto = pd.concat([df_nba, df_nbb], ignore_index=True)
            
            # Cria uma pasta para os dados mistos e salva o ficheiro
         #   pasta_mista = os.path.join(base_path, "data", "experimento_misto_nba", jogos_media, janela)
            pasta_mista = os.path.join(base_path, "data", "experimento_misto_nbb", jogos_media, janela)
            os.makedirs(pasta_mista, exist_ok=True)
            path_treino_misto = os.path.join(pasta_mista, "treino_misto.csv")
            df_misto.to_csv(path_treino_misto, index=False)
            
            # --- EXECUÇÃO DO MODELO COM OS DADOS MISTOS ---
            if modelo == 'xgboost':
                accuracy, f1, best_params = run_model_xgboost(path_treino_misto, path_teste_nbb, True)
            elif modelo == 'regressao_logistica':
                accuracy, f1, best_params = run_model_logreg(path_treino_misto, path_teste_nbb, True)
            elif modelo == 'rede_neural':
                accuracy, f1, best_params = run_model_rede_neural(path_treino_misto, path_teste_nbb, True)
            elif modelo == 'random_forest':
                accuracy, f1, best_params = run_model_rf(path_treino_misto, path_teste_nbb, True)

            if accuracy is not None and f1 is not None:
                acuracias_do_experimento.append(accuracy)
                f1_scores_do_experimento.append(f1)
                print(f"   [OK] F1-Score: {f1:.4f} | Tamanho do Treino: {len(df_misto)} jogos")

                results.append({
                    'Janela': janela,
                    'Acurácia': f'{accuracy:.4f}',
                    'Desvio Padrão Acurácia': '-',
                    'F1-Score': f'{f1:.4f}',
                    'Desvio Padrão F1-Score': '-',
                })
        else:
            print(f"   [!] Faltam ficheiros de treino/teste para a {janela}.")

    # --- FINALIZAÇÃO E SALVAMENTO ---
    if acuracias_do_experimento:
        results.append({
            'Janela': 'Média Geral',
            'Acurácia': f'{np.mean(acuracias_do_experimento):.4f}',
            'Desvio Padrão Acurácia': f'{np.std(acuracias_do_experimento):.2f}',
            'F1-Score': f'{np.mean(f1_scores_do_experimento):.4f}',
            'Desvio Padrão F1-Score': f'{np.std(f1_scores_do_experimento):.2f}',
        })

 #   output_dir = os.path.join(base_path, 'results', 'treino_misto_nba')
    output_dir = os.path.join(base_path, 'results', 'treino_misto_nbb')
    os.makedirs(output_dir, exist_ok=True)
 #   path_resultados = os.path.join(output_dir, f'{modelo}_treino_misto_nba_{jogos_media}_minuto.csv')
    path_resultados = os.path.join(output_dir, f'{modelo}_treino_misto_nbb_{jogos_media}_minuto.csv')
    
    save_results_csv(path_resultados, results)
    print(f"\n>> Resultados mistos guardados em: {path_resultados}")

end_time = time.time()
print(f"\n--- Experimento de Treino Misto concluído em {end_time - start_time:.2f} segundos ---")