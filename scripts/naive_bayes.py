from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

from experimentos import read_dados

def run_model_naive_bayes(treino_path, teste_path, useGridSearch=False):
    X_train, X_test, y_train, y_test = read_dados(treino_path, teste_path)

    # Criar o modelo Naïve Bayes
    model = GaussianNB()

    # O Naïve Bayes não utiliza Grid Search para ajuste de hiperparâmetros, mas mantemos o parâmetro
    # `useGridSearch` para consistência com os outros classificadores.
    best_params = []  # Não há hiperparâmetros ajustáveis aqui

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Fazer previsões com o conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Para lidar com desbalanceamento de classes

    return accuracy, f1, best_params
