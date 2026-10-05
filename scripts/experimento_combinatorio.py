import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# Seus imports de modelos
from rede_neural import run_model_rede_neural
from svm import run_model_svm 
from random_forest import run_model_rf
from naive_bayes import run_model_naive_bayes
from vanilla import run_model_vanilla
from xgboost_model import run_model_xgboost
from cart_model import run_model_cart
from regressao_logistica_model import run_model_logreg

# Experimento 5 - Janela Combinatoria (acumulada + intertemporadas) PARA A NBA

# Configurações
modelos = ['cart_model','svm','xgboost','regressao_logistica'] 

# MUDANÇA 1: A primeira temporada (2015-16) é usada como treino base histórico.
# Portanto, as temporadas de TESTE iniciam a partir de 2016-17.
temporadas_nba = [
  #'2008-09', 
  #'2009-10', '2010-11', 
  #'2011-12', '2012-13', 
 # '2013-14', 
 # '2014-15', 
 # '2015-16', '2016-17',
  #'2017-18', 
 # '2018-19', 
 #'2019-20'
#   '2020-21','2021-22','2022-23', '2023-24'
    '2024-25'
]

# MUDANÇA 2: Deixamos K=1 porque o salto de 10 jogos JÁ FOI FEITO na extração!
K_espacamento = 1
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
start_time = time.time()

for modelo in modelos:
    for temporada in temporadas_nba:
        # MUDANÇA 3: Aponta diretamente para a pasta do nba_experimento_05
        temporada_dir = os.path.join(base_path, "data", "nba_experimento_05_minuto", temporada)
        
        if not os.path.exists(temporada_dir):
            print(f"[!] Diretório não encontrado: {temporada_dir}")
            continue

        janelas = [d for d in os.listdir(temporada_dir) if os.path.isdir(os.path.join(temporada_dir, d))]
        # Ordena as pastas corretamente (ex: 15-1, 25-1, 35-1)
        janelas.sort(key=lambda x: int(x.split('-')[0]))

        print(f"\n>> Processando NBA: {modelo.upper()} | Temporada {temporada} (Inter + Incremental)")

        num_sequencial = 1
        acertos_acumulados = 0
        total_jogos_testados = 0
        evolucao_metricas = []
        
        historico_hiperparametros = []
        y_true_acumulado = []
        y_pred_acumulado = []

        for i, pasta_janela in enumerate(janelas):
            if i % K_espacamento != 0:
                num_sequencial += 1
                continue

            treino_path = os.path.join(temporada_dir, pasta_janela, f"treino_{num_sequencial}.csv")
            teste_path = os.path.join(temporada_dir, pasta_janela, f"teste_{num_sequencial}.csv")

            if os.path.exists(treino_path) and os.path.exists(teste_path):
                try:
                    from experimentos import read_dados
                    _, _, _, y_test_val = read_dados(treino_path, teste_path)
                    
                    accuracy = None
                    best_params = None  
                    
                    if modelo == 'naive_bayes':
                        accuracy, _, best_params = run_model_naive_bayes(treino_path, teste_path)
                    elif modelo == 'svm':
                        accuracy, _, best_params = run_model_svm(treino_path, teste_path, True)
                    elif modelo == 'vanilla':
                        accuracy, _, best_params = run_model_vanilla(treino_path, teste_path)
                    elif modelo == 'random_forest':
                        accuracy, _, best_params = run_model_rf(treino_path, teste_path, True)
                    elif modelo == 'xgboost':
                        accuracy, _, best_params = run_model_xgboost(treino_path, teste_path, True)
                    elif modelo == 'rede_neural':
                        accuracy, _, best_params = run_model_rede_neural(treino_path, teste_path, True)
                    elif modelo == 'regressao_logistica':
                        accuracy, f1, best_params = run_model_logreg(treino_path, teste_path, True)
                    elif modelo == 'cart_model':
                        accuracy, f1, best_params = run_model_cart(treino_path, teste_path, True)

                    if accuracy is not None and y_test_val is not None:
                        total_jogos_testados += 1
                        
                        y_real = y_test_val[0]
                        
                        if accuracy == 1.0:
                            acertos_acumulados += 1
                            y_pred = y_real
                        else:
                            y_pred = 1 - y_real

                        y_true_acumulado.append(y_real)
                        y_pred_acumulado.append(y_pred)
                        
                        acuracia_prog = acertos_acumulados / total_jogos_testados
                        f1_prog = f1_score(y_true_acumulado, y_pred_acumulado, average='weighted', zero_division=0)
                        
                        registro = {                            
                            'Jogo_Real_n': i + 1,
                            'Amostra_n': total_jogos_testados,
                            'Resultado': "Acerto" if accuracy == 1.0 else "Erro",
                            'Acuracia_Acumulada': round(acuracia_prog, 4),
                            'F1_Score_Acumulado': round(f1_prog, 4)
                        }

                        if best_params:
                            registro_param = {
                                'Jogo_Real_n': i + 1,
                                'Hiperparametros': best_params
                            }
                            historico_hiperparametros.append(registro_param)
                        
                        evolucao_metricas.append(registro)

                except Exception as e:
                    print(f"[!] Erro na janela {pasta_janela}: {e}")

            num_sequencial += 1

        # MUDANÇA 4: Salvamento dos resultados na pasta exclusiva da NBA
        output_dir = os.path.join(base_path, 'results', 'nba_experimento_05_minuto')
        os.makedirs(output_dir, exist_ok=True)

        if evolucao_metricas:
            filename = os.path.join(output_dir, f'evolucao_{modelo}_{temporada}_K{K_espacamento}_min.csv')
            pd.DataFrame(evolucao_metricas).to_csv(filename, index=False)
        
     #   if historico_hiperparametros:
     #       param_filename = os.path.join(output_dir, f'params_{modelo}_{temporada}_K{K_espacamento}_min.csv')
     #       pd.DataFrame(historico_hiperparametros).to_csv(param_filename, index=False)

print(f"\n--- Experimento 05 NBA concluído em {time.time() - start_time:.2f} segundos ---")