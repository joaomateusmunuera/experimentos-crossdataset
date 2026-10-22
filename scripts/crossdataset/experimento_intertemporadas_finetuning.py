import os
import sys
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# --- CONFIGURAÇÕES INICIAIS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(current_dir, '..'))
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

from experimentos import save_results_csv

modelos = ['rede_neural']
jogos_media = '15' 
janelas = [f"janela_{i:02d}" for i in range(1, 15)]
base_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

start_time = time.time()

for modelo in modelos:
    print(f"\n{'='*50}")
    print(f" INICIANDO EXPERIMENTO DE FINE-TUNING: {modelo.upper()}")
    print(f"{'='*50}")
    
    results = []
    acuracias = []
    f1_scores = []

    for janela in janelas:
        # 1. Definição de Caminhos
        path_treino_nba = os.path.join(base_path, "data", "cross_dataset_nba_para_nbb", jogos_media, janela, "treino_nba.csv")
        path_treino_nbb = os.path.join(base_path, "data", "cross_dataset_nbb_para_nba", jogos_media, janela, "treino_nbb.csv")
        path_teste_nbb  = os.path.join(base_path, "data", "cross_dataset_nba_para_nbb", jogos_media, janela, "teste_nbb.csv")

        if not (os.path.exists(path_treino_nba) and os.path.exists(path_treino_nbb) and os.path.exists(path_teste_nbb)):
            print(f" [!] Faltam arquivos na {janela}. Pulando...")
            continue

        # 2. Leitura dos Dados
        df_treino_nba = pd.read_csv(path_treino_nba)
        df_treino_nbb = pd.read_csv(path_treino_nbb)
        df_teste_nbb  = pd.read_csv(path_teste_nbb)

        colunas_ignorar = ['placar_casa','placar_visitante','resultado', 'data', 'equipe_casa', 'equipe_visitante', 'ano', 'round', 'stage']
        
        X_train_nba = df_treino_nba.drop(columns=colunas_ignorar, errors='ignore')
        y_train_nba = df_treino_nba['resultado']
        
        X_train_nbb = df_treino_nbb.drop(columns=colunas_ignorar, errors='ignore')
        y_train_nbb = df_treino_nbb['resultado']
        
        X_test_nbb = df_teste_nbb.drop(columns=colunas_ignorar, errors='ignore')
        y_test_nbb = df_teste_nbb['resultado']

        print(f"\n>> Processando {janela}...")

        # --- PRÉ-PROCESSAMENTO ESPECÍFICO PARA REDE NEURAL ---
        if modelo == 'rede_neural':
            scaler = StandardScaler()
            # A NBA define a escala universal (fit_transform)
            X_train_nba = scaler.fit_transform(X_train_nba)
            # O NBB apenas herda essa escala (transform)
            X_train_nbb = scaler.transform(X_train_nbb)
            X_test_nbb = scaler.transform(X_test_nbb)

        # --- EXECUÇÃO DOS MODELOS ---
        if modelo == 'xgboost':
            # Etapa 1: Treino Base NBA
            modelo_base = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, 
                random_state=42, eval_metric='logloss'
            )
            modelo_base.fit(X_train_nba, y_train_nba)
            
            temp_model_path = os.path.join(base_path, "results", f"temp_base_nba_{janela}.json")
            os.makedirs(os.path.dirname(temp_model_path), exist_ok=True)
            modelo_base.save_model(temp_model_path)

            # Etapa 2: Fine-Tuning NBB
            modelo_transfer = xgb.XGBClassifier(
                n_estimators=50, learning_rate=0.05, max_depth=5, 
                random_state=42, eval_metric='logloss'
            )
            modelo_transfer.fit(X_train_nbb, y_train_nbb, xgb_model=temp_model_path)
            
            y_pred = modelo_transfer.predict(X_test_nbb)

            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

        elif modelo == 'rede_neural':
            # Etapa 1: Treino Base NBA com warm_start ativado
            modelo_transfer = MLPClassifier(
                hidden_layer_sizes=(50,25),
                alpha=0.05, 
                max_iter=200, 
                learning_rate_init=0.0005,
                warm_start=True, # Crucial: Não apaga a memória entre os .fit()
                random_state=42
            )
            modelo_transfer.fit(X_train_nba, y_train_nba)
            
            # Etapa 2: Fine-Tuning NBB
            # Aumentamos o limite de épocas para permitir que a rede continue aprendendo com o NBB
            modelo_transfer.max_iter += 50 
            modelo_transfer.fit(X_train_nbb, y_train_nbb)
            
            y_pred = modelo_transfer.predict(X_test_nbb)

        # --- AVALIAÇÃO E SALVAMENTO ---
        acc = accuracy_score(y_test_nbb, y_pred)
        f1 = f1_score(y_test_nbb, y_pred)
        
        acuracias.append(acc)
        f1_scores.append(f1)
        print(f"   [OK] F1-Score: {f1:.4f}")

        results.append({
            'Janela': janela,
            'Acurácia': f'{acc:.4f}',
            'Desvio Padrão Acurácia': '-',
            'F1-Score': f'{f1:.4f}',
            'Desvio Padrão F1-Score': '-',
        })
        
    # --- FINALIZAÇÃO DO MODELO ATUAL ---
    if acuracias:
        results.append({
            'Janela': 'Média Geral',
            'Acurácia': f'{np.mean(acuracias):.4f}',
            'Desvio Padrão Acurácia': f'{np.std(acuracias):.2f}',
            'F1-Score': f'{np.mean(f1_scores):.4f}',
            'Desvio Padrão F1-Score': f'{np.std(f1_scores):.2f}',
        })

    output_dir = os.path.join(base_path, 'results', 'fine_tuning')
    os.makedirs(output_dir, exist_ok=True)
    path_resultados = os.path.join(output_dir, f'{modelo}_fine_tuning_{jogos_media}_minuto.csv')

    save_results_csv(path_resultados, results)
    print(f"\n>> Resultados de {modelo.upper()} guardados em: {path_resultados}")

print(f"\n--- Tempo total de execução: {time.time() - start_time:.2f} segundos ---")