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

#Experimento 3 - Janela Incremental

# Configurações
#jogos_medias = ['15']
modelos = ['svm','regressao_logistica','cart_model']
temporadas = [
    "2008-2009", "2009-2010", "2011-2012", "2012-2013",
    "2013-2014", "2014-2015", "2015-2016", "2016-2017",
    "2018-2019", "2019-2020", "2020-2021", "2021-2022", 
    "2022-2023", "2023-2024","2024-2025"
]

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
start_time = time.time()

for modelo in modelos:
   #for jogos_media in jogos_medias:
        results = [] 
        
        for temporada in temporadas:
            temporada_dir = os.path.join(base_path, "data", "experimento_03", temporada)
            
            if not os.path.exists(temporada_dir):
                print(f"Pulo: {temporada} não encontrada")
                continue

            # Lista pastas como '5-1', '6-1' e ordena pelo primeiro número (o índice do jogo)
            janelas = [d for d in os.listdir(temporada_dir) if os.path.isdir(os.path.join(temporada_dir, d))]
            janelas.sort(key=lambda x: int(x.split('-')[0]))

            acuracias_temporada = []
            y_real_acumulado = []      # resultado verdadeiro do jogo
            y_pred_acumulado = []      # palpite feito pelo modelo para cada jogo

            print(f"\n>> Processando {modelo.upper()} | Temporada {temporada}")

            # 'num_sequencial' controlará o sufixo do arquivo (treino_1, treino_2...)
            num_sequencial = 1 

            for pasta_janela in janelas:
                treino_path = os.path.join(temporada_dir, pasta_janela, f"treino_{num_sequencial}.csv")
                teste_path = os.path.join(temporada_dir, pasta_janela, f"teste_{num_sequencial}.csv")

                if os.path.exists(treino_path) and os.path.exists(teste_path):
                    try:
                        accuracy = None
                        _, _, _, y_test = read_dados(treino_path, teste_path)
                        valor_real = y_test[0]

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

                        # Resultados somente se o modelo retornou valores válidos
                        if accuracy is not None and f1 is not None:
                            acuracias_temporada.append(accuracy)
                            y_real_acumulado.append(valor_real) #resultado real
                            predicao = valor_real if accuracy == 1.0 else 1 - valor_real # matriz de confusao 
                            y_pred_acumulado.append(predicao) #previsao
                        else:
                            print(f"[-] Modelo {modelo} retornou vazio na janela {pasta_janela}")

                    except Exception as e:
                        print(f"[!] Erro ao processar janela {pasta_janela}: {e}")
                
                # Incrementa o sufixo independentemente do sucesso, para manter a sincronia com as pastas
                num_sequencial += 1

            # Consolidação dos resultados da temporada (Média dos N jogos testados)
            if acuracias_temporada:
                mean_acc = np.mean(acuracias_temporada)
                real_f1 = f1_score(y_real_acumulado, y_pred_acumulado, average='weighted') #
                
                results.append({
                    'Janela Incremental': 15, # O valor de K (5, 10 ou 15)
                    'Temporada': temporada,
                    'Acurácia': f'{mean_acc:.4f}',
                    'F1-Score': f'{real_f1:.4f}'
                })

        # Salva um CSV de resultados para cada combinação de Modelo e K
        output_dir = os.path.join(base_path, 'results', 'experimento_03')
        os.makedirs(output_dir, exist_ok=True)
        
        # Nome do arquivo final
        filename = os.path.join(output_dir, f'{modelo}_K15_final.csv')
        
        df_final = pd.DataFrame(results)
        df_final.to_csv(filename, index=False)
        print(f">> Arquivo salvo em: {filename}")

end_time = time.time()
print(f"\n--- Experimento concluído em {end_time - start_time:.2f} segundos ---")