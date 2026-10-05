from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from experimentos import read_dados

def get_hyper_params_cart(X_train, y_train):
    # Grade de hiperparâmetros enxuta para a Árvore de Decisão (CART)
    param_grid = {
        'criterion': ['gini', 'entropy'],      # Métrica de divisão dos nós
        'max_depth': [3, 5, 7, None],          # Profundidade máxima da árvore
        'min_samples_split': [2, 5, 10],       # Amostras mínimas para dividir um nó
        'min_samples_leaf': [1, 2, 4]          # Amostras mínimas em um nó folha
    }

    # Criar o modelo CART (Decision Tree)
    cart = DecisionTreeClassifier(random_state=42)

    # Configurar o Grid Search
    grid_search = GridSearchCV(
        estimator=cart, 
        param_grid=param_grid, 
        cv=2, 
        scoring='f1_weighted', 
        verbose=0,       # Mantido em 0 para não poluir o terminal
        n_jobs=-1
    )

    try:
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
    except ValueError as e:
        print(f"Erro durante o Grid Search (CART): {e}")
        return None

    return best_params

def run_model_cart(treino_path, teste_path, useGridSearch=True):
    X_train, X_test, y_train, y_test = read_dados(treino_path, teste_path)

    if useGridSearch:
        best_params = get_hyper_params_cart(X_train, y_train)

        if best_params is None:
            return None, None, None

        model = DecisionTreeClassifier(
            criterion=best_params['criterion'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=42
        )
    else:
        # Hiperparâmetros base caso o GridSearch esteja desativado
        model = DecisionTreeClassifier(
            criterion='gini',
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        best_params = []

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return accuracy, f1, best_params