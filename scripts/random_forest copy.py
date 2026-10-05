from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from experimentos import read_dados

def get_hyper_params_rf(X_train, y_train):
    # Definir os hiperparâmetros para o Grid Search
    param_grid = {
        'n_estimators': [50, 100],  # Número de árvores na floresta
        'max_depth': [None, 10],  # Profundidade máxima da árvore
        'min_samples_split': [5, 10],  # Número mínimo de amostras para dividir um nó
        'min_samples_leaf': [1, 4],  # Número mínimo de amostras em uma folha
        'bootstrap': [True]  # Método de amostragem
    }

    # Criar o modelo Random Forest
    rf = RandomForestClassifier(random_state=42)

    # Configurar o Grid Search com validação cruzada
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, scoring='f1_weighted', verbose=2, n_jobs= -2)

    # Executar o Grid Search no conjunto de treino
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    return best_params

def run_model_rf(treino_path, teste_path, useGridSearch=True):
    X_train, X_test, y_train, y_test = read_dados(treino_path, teste_path)

    if useGridSearch:
        # Obter os melhores hiperparâmetros usando Grid Search
        best_params = get_hyper_params_rf(X_train, y_train)

        # Criar e treinar o modelo com os melhores hiperparâmetros
        model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            bootstrap=best_params['bootstrap'],
            random_state=42
        )
    else:
        # Modelo com hiperparâmetros padrão
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        best_params = []

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Fazer previsões com o conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Para lidar com desbalanceamento de classes

    return accuracy, f1, best_params
