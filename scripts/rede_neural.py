from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from experimentos import read_dados

def get_hyper_params_rede_neural(X_train, y_train):
    # Definir os hiperparâmetros para o Grid Search
    # param_grid = {
    #     'max_iter':[10000],
    #     'hidden_layer_sizes': [(50, 50), (100, 50), (100, 100)],
    #     'activation': ['relu', 'tanh'],
    #     'solver': ['adam', 'sgd'],
    #     'learning_rate': ['constant']
    # }

    param_grid = {
        'max_iter':[10000],
        'hidden_layer_sizes': [(50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive']
    }

    # Criar o MLPClassifier
    mlp = MLPClassifier(max_iter=10000, random_state=42)

    # Configurar o Grid Search com validação cruzada
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=2, scoring='f1_weighted', verbose=2, n_jobs=-2)

    # Executar o Grid Search no conjunto de treino
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    return best_params

def run_model_rede_neural(treino_path, teste_path, useGridSearch=True):
    X_train, X_test, y_train, y_test = read_dados(treino_path, teste_path)

    if (useGridSearch):
        best_params = get_hyper_params_rede_neural(X_train, y_train)

        # Criar e treinar o modelo com os melhores hiperparâmetros
        model = MLPClassifier(
            hidden_layer_sizes=best_params['hidden_layer_sizes'],
            activation=best_params['activation'],
            solver=best_params['solver'],
            learning_rate=best_params['learning_rate'],
            max_iter=best_params['max_iter'],
            random_state=42
        )

    else:
        model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=10000, random_state=42)

    # Treinar modelo
    model.fit(X_train, y_train)

    # Fazer previsões com o conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # 'weighted' para lidar com desbalanceamento de classes  
                                                       # Calcula o F1 Score para cada classe individualmente, mas pondera cada valor pela proporção de amostras de cada classe no conjunto de dados.

    return accuracy, f1, best_params