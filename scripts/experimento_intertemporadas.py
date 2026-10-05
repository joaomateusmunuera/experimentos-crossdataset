import os
import time
import pandas as pd
import numpy as np

from rede_neural import run_model_rede_neural
from svm import run_model_svm 
from random_forest import run_model_rf
from naive_bayes import run_model_naive_bayes
from vanilla import run_model_vanilla
from xgboost_model import run_model_xgboost
from cart_model import run_model_cart
from regressao_logistica_model import run_model_logreg

from experimentos import save_results_csv

# Experimento 4 - Janela Inter Temporadas (NBA)

modelos = ['svm','regressao_logistica','cart_model'] # Adicione outros se desejar
jogos_media = '15' 

# MUDANÇA: Com 9 temporadas totais na lista (2015 a 2024), teremos 8 janelas (01 a 08)
janelas = [f"janela_{i:02d}" for i in range(1, 15)]

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
start_time = time.time()

for modelo in modelos:
    results = [] 
    acuracias_do_experimento = []
    f1_scores_do_experimento = []
    historico_hiperparametros = []

    for janela in janelas:
        # MUDANÇA: Lendo da pasta nba_experimento_04_minuto
        treino_path = os.path.join(base_path, "data", "nba_experimento_04_minuto", jogos_media, janela, "treino.csv")
        teste_path = os.path.join(base_path, "data", "nba_experimento_04_minuto", jogos_media, janela, "teste.csv")

        print(f"Rodando modelo NBA: {modelo.upper()} | Média: {jogos_media} | {janela}")
        
        accuracy = None
        f1 = None
        best_params = None

        if os.path.exists(treino_path) and os.path.exists(teste_path):
            if modelo == 'naive_bayes':
                accuracy, f1, best_params = run_model_naive_bayes(treino_path, teste_path)
            elif modelo == 'svm':
                accuracy, f1, best_params = run_model_svm(treino_path, teste_path, True)
            elif modelo == 'vanilla':
                accuracy, f1, best_params = run_model_vanilla(treino_path, teste_path)
            elif modelo == 'random_forest':
                accuracy, f1, best_params = run_model_rf(treino_path, teste_path, True)
            elif modelo == 'xgboost':
                accuracy, f1, best_params = run_model_xgboost(treino_path, teste_path, True)
            elif modelo == 'rede_neural':
                accuracy, f1, best_params = run_model_rede_neural(treino_path, teste_path, True)
            elif modelo == 'regressao_logistica':
                accuracy, f1, best_params = run_model_logreg(treino_path, teste_path, True)
            elif modelo == 'cart_model':
                accuracy, f1, best_params = run_model_cart(treino_path, teste_path, True)

            # 2. ADICIONE A VERIFICAÇÃO DO F1 AQUI
            if accuracy is not None and f1 is not None:
                acuracias_do_experimento.append(accuracy)
                f1_scores_do_experimento.append(f1)

                results.append({
                    'Janela': janela,
                    'Acurácia': f'{accuracy:.4f}',
                    'Desvio Padrão Acurácia': '-',
                    'F1-Score': f'{f1:.4f}',
                    'Desvio Padrão F1-Score': '-',
                })

                if best_params:
                    historico_hiperparametros.append({
                        'Janela': janela,
                        'Hiperparametros': best_params
                    })
            else:
                print(f"   [!] Pulando a {janela} pois o modelo falhou e retornou None.")

    if acuracias_do_experimento:
        media_acuracia = np.mean(acuracias_do_experimento)
        media_f1_score = np.mean(f1_scores_do_experimento)
        desvio_padrao_acuracia = np.std(acuracias_do_experimento)
        desvio_padrao_f1_score = np.std(f1_scores_do_experimento)

        results.append({
            'Janela': 'Média Geral',
            'Acurácia': f'{media_acuracia:.4f}',
            'Desvio Padrão Acurácia': f'{desvio_padrao_acuracia:.2f}',
            'F1-Score': f'{media_f1_score:.4f}',
            'Desvio Padrão F1-Score': f'{desvio_padrao_f1_score:.2f}',
        })

    # MUDANÇA: Salvando resultados na pasta nba_experimento_04_minuto
    output_dir = os.path.join(base_path, 'results', 'nba_experimento_04_minuto')
    os.makedirs(output_dir, exist_ok=True)

    path_resultados = os.path.join(output_dir, f'{modelo}_experimento_04_{jogos_media}_minuto.csv')
    save_results_csv(path_resultados, results)

   # if historico_hiperparametros:
     #   path_params = os.path.join(output_dir, f'params_{modelo}_experimento_04_{jogos_media}_minuto.csv')
     #   pd.DataFrame(historico_hiperparametros).to_csv(path_params, index=False)

end_time = time.time()
print(f"\n--- Experimento 04 da NBA concluído em {end_time - start_time:.2f} segundos ---")