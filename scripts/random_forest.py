import os
# O silenciador ABSOLUTO: Injeta a regra no Windows antes do Scikit-Learn nascer
os.environ["PYTHONWARNINGS"] = "ignore"

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from experimentos import read_dados

def get_hyper_params_rf(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],       
        'max_depth': [5, 7],              
        'min_samples_split': [10, 15],    
        'min_samples_leaf': [4, 8],       
        'max_features': ['sqrt']          
    }

    # CRÍTICO PARA NÃO TRAVAR: n_jobs=None aqui para evitar o engarrafamento
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3,                  
        scoring='f1_weighted', 
        # MUDANÇA VISUAL: Verbose=3 vai imprimir na tela CADA teste concluído!
        verbose=3,             
        n_jobs=-1  # O paralelismo fica exclusivamente aqui no GridSearch
    )

    try:
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
    except ValueError as e:
        print(f"Erro durante o Grid Search (RF): {e}")
        return None

    return best_params


def run_model_rf(treino_path, teste_path, useGridSearch=True):
    X_train, X_test, y_train, y_test = read_dados(treino_path, teste_path)

    if useGridSearch:
        best_params = get_hyper_params_rf(X_train, y_train)

        if best_params is None:
            return None, None, None

        # Na hora de treinar o modelo FINAL (fora do GridSearch), podemos usar força total
        model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            max_features=best_params['max_features'],
            bootstrap=True,
            random_state=42,
            n_jobs=-1  
        )
    else:
        model = RandomForestClassifier(
            n_estimators=150, 
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        best_params = []

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, f1, best_params