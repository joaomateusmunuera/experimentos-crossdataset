from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from experimentos import read_dados

def run_model_vanilla(treino_path, teste_path):
    # Lê os dados (usando uma função fictícia para simular os dados)
    X_train, X_test, y_train, y_test = read_dados(treino_path, teste_path)

    # Simula as previsões: tudo como True
    y_pred = np.ones_like(y_test)  # Cria um array com os mesmos tamanhos de y_test e preenche com True (ou 1)

    # Avalia o desempenho simulando 100% de previsões como "True"
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Retorna os resultados simulados
    return accuracy, f1, []