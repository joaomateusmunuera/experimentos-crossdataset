import os
import sys
import time
import pandas as pd
import numpy as np

# Adiciona a pasta pai 'scripts/' ao caminho do Python para conseguir importar os modelos
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(current_dir, '..'))
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# Importações dos classificadores (agora funcionam porque o Python sabe onde 'scripts/' está)
from rede_neural import run_model_rede_neural
from svm import run_model_svm 
from random_forest import run_model_rf
from naive_bayes import run_model_naive_bayes
from vanilla import run_model_vanilla
from xgboost_model import run_model_xgboost
from cart_model import run_model_cart
from regressao_logistica_model import run_model_logreg

from experimentos import save_results_csv

# Experimento Cross-Dataset (Treino NBB -> Teste NBA)

modelos = ['svm', 'regressao_logistica', 'cart_model', 'xgboost', 'random_forest']
jogos_media = '15' 
janelas = [f"janela_{i:02d}" for i in range(1, 14)]

# Como o script está em 'nba-nbb/scripts/crossdata/', precisamos subir DOIS níveis para chegar na raiz 'nba-nbb/'
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
start_time = time.time()

for modelo in modelos:
    results = [] 
    acuracias_do_experimento = []
    f1_scores_do_experimento = []
    historico_hiperparametros = []

    for janela in janelas:
        # Aponta para a raiz do projeto -> data/cross_dataset_minuto/...
        treino_path = os.path.join(base_path, "data", "cross_dataset_minuto", jogos_media, janela, "treino_nbb.csv")
        teste_path = os.path.join(base_path, "data", "cross_dataset_minuto", jogos_media, janela, "teste_nba.csv")

        print(f"Rodando Cross-Dataset: {modelo.upper()} | Média: {jogos_media} | {janela}")
        
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
        else:
            print(f"   [!] Arquivos não encontrados para {janela} em {treino_path}")

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

    # Salvando os resultados na pasta results/ da raiz do projeto
    output_dir = os.path.join(base_path, 'results', 'cross_dataset_minuto')
    os.makedirs(output_dir, exist_ok=True)

    path_resultados = os.path.join(output_dir, f'{modelo}_cross_dataset_{jogos_media}_minuto.csv')
    save_results_csv(path_resultados, results)

end_time = time.time()
print(f"\n--- Experimento Cross-Dataset concluído em {end_time - start_time:.2f} segundos ---")