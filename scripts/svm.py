from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from experimentos import read_dados

def get_hyper_params_svm(X_train, y_train):
    # Otimização 1: Separar os dicionários evita testar hiperparâmetros inativos.
    # Ex: 'gamma' não tem efeito no kernel 'linear'.
    # O kernel 'poly' foi removido devido à complexidade computacional absurda em grandes datasets.
    param_grid = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10]},
        {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    ]

    # Otimização 2: Aumentar o cache_size de 200MB para 1000MB acelera muito o kernel 'rbf'
    svm = SVC(random_state=42, cache_size=1000)

    # Configurar o Grid Search (n_jobs=-1 usa 100% do processador, verbose=0 tira o spam do terminal)
    grid_search = GridSearchCV(
        estimator=svm, 
        param_grid=param_grid, 
        cv=2, 
        scoring='f1_weighted', 
        verbose=0, 
        n_jobs=-1
    )

    # Otimização 3: Subamostragem para a fase de GridSearch.
    # O SVM tem complexidade de tempo O(n^2) a O(n^3). Buscar parâmetros em 15.000 jogos demoraria dias.
    max_amostras_grid = 5000
    if len(X_train) > max_amostras_grid:
        # Pega as linhas mais recentes para o Grid Search
        if hasattr(X_train, 'iloc'): # Se for Pandas DataFrame
            X_train_grid = X_train.iloc[-max_amostras_grid:]
            y_train_grid = y_train.iloc[-max_amostras_grid:]
        else: # Se for Numpy Array
            X_train_grid = X_train[-max_amostras_grid:]
            y_train_grid = y_train[-max_amostras_grid:]
    else:
        X_train_grid = X_train
        y_train_grid = y_train

    try:
        # Roda o Grid Search apenas na subamostra
        grid_search.fit(X_train_grid, y_train_grid)
        best_params = grid_search.best_params_
    except ValueError as e:
        print(f"Erro durante o Grid Search (SVM): {e}")
        return None

    return best_params

def run_model_svm(treino_path, teste_path, useGridSearch=True):
    X_train, X_test, y_train, y_test = read_dados(treino_path, teste_path)

    if useGridSearch:
        best_params = get_hyper_params_svm(X_train, y_train)

        if best_params is None:
            return None, None, None

        # Criar e treinar o modelo com os melhores hiperparâmetros (mantendo cache alto)
        model = SVC(
            C=best_params['C'],
            kernel=best_params['kernel'],
            gamma=best_params.get('gamma', 'scale'), # Usa .get() para evitar erro no kernel linear
            random_state=42,
            cache_size=1000 
        )
    else:
        # Modelo com hiperparâmetros padrão
        model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42, cache_size=1000)
        best_params = []

    # Treinar o modelo final usando TODA a base de treino
    model.fit(X_train, y_train)

    # Fazer previsões com o conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) 

    return accuracy, f1, best_params