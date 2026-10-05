from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from experimentos import read_dados

def get_hyper_params_xgboost(X_train, y_train):
    # Grade de hiperparâmetros FOCADA EM REGULARIZAÇÃO (Anti-Overfitting para a NBA)
    param_grid = {
        'n_estimators': [100, 150],           # Mais árvores...
        'learning_rate': [0.05, 0.1],         # ...aprendendo mais devagar
        'max_depth': [3, 4],                  # Árvores rasas para não decorar ruídos (zebras)
        'min_child_weight': [3, 5],           # Exige mais "provas" antes de criar uma regra
        'reg_lambda': [1, 5],                 # Regularização L2 (Penaliza pesos extremos)
        'subsample': [0.8],                   
        'colsample_bytree': [0.8]        
    }

    # Criar o modelo XGBoost
    xgb = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-2)

    # Configurar o Grid Search
    grid_search = GridSearchCV(
        estimator=xgb, 
        param_grid=param_grid, 
        cv=3,            # Aumentado para 3 para testes mais rigorosos
        scoring='f1_weighted', 
        verbose=0,       
        n_jobs=-1
    )

    try:
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
    except ValueError as e:
        print(f"Erro durante o Grid Search: {e}")
        return None

    return best_params


def run_model_xgboost(treino_path, teste_path, useGridSearch=True):
    X_train, X_test, y_train, y_test = read_dados(treino_path, teste_path)

    if useGridSearch:
        best_params = get_hyper_params_xgboost(X_train, y_train)

        if best_params is None:
            return None, None, None

        model = XGBClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            min_child_weight=best_params['min_child_weight'], # <-- Novo
            reg_lambda=best_params['reg_lambda'],             # <-- Novo
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            eval_metric='logloss',
            random_state=42
        )
    else:
        # Configuração padrão mais robusta caso o GridSearch esteja desligado
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=3,
            reg_lambda=1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42
        )
        best_params = []

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, f1, best_params