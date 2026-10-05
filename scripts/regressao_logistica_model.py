from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from experimentos import read_dados

def get_hyper_params_logreg(X_train, y_train):
    # Grade de hiperparâmetros 100% nativa para Scikit-Learn 1.8+
    # A palavra 'penalty' foi completamente removida.
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.0, 1.0],       # 0.0 = L2 (Suavização) | 1.0 = L1 (Seleção de features)
        'solver': ['liblinear']       # Otimizador perfeito para lidar com a alta correlação do basquete
    }

    # Instância limpa sem conflitos internos
    logreg = LogisticRegression(max_iter=1000, random_state=42)

    grid_search = GridSearchCV(
        estimator=logreg, 
        param_grid=param_grid, 
        cv=3,            
        scoring='f1_weighted', 
        verbose=0,       
        n_jobs=-1
    )

    try:
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
    except ValueError as e:
        print(f"Erro durante o Grid Search (Logistic Regression): {e}")
        return None

    return best_params


def run_model_logreg(treino_path, teste_path, useGridSearch=True):
    X_train, X_test, y_train, y_test = read_dados(treino_path, teste_path)

    # Padronização garantida para evitar o "ConvergenceWarning"
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if useGridSearch:
        best_params = get_hyper_params_logreg(X_train_scaled, y_train)

        if best_params is None:
            return None, None, None

        model = LogisticRegression(
            C=best_params['C'],
            l1_ratio=best_params['l1_ratio'],  # Usando APENAS a nova sintaxe exigida
            solver=best_params['solver'],
            max_iter=1000,
            random_state=42
        )
    else:
        # Fallback conservador e compatível com as novas regras
        model = LogisticRegression(
            C=0.1,
            l1_ratio=0.0,       # 0.0 = L2 padrão
            solver='liblinear',
            max_iter=1000,
            random_state=42
        )
        best_params = []

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, f1, best_params