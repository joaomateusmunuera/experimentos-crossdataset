from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from experimentos import read_dados

def get_hyper_params_xgboost(X_train, y_train):
    # Grade de hiperparâmetros OTIMIZADA com base no histórico da Liga
    param_grid = {
        'n_estimators': [50, 100],       # Reduzido de [50, 100, 150]
        'max_depth': [3, 7],             # Reduzido de [3, 5, 7]
        'learning_rate': [0.1, 0.2],     # Reduzido de [0.01, 0.1, 0.2]
        'subsample': [0.8],              # Fixado no valor dominante
        'colsample_bytree': [0.8]        # Fixado no valor dominante
    }

    # Criar o modelo XGBoost
    xgb = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-2)

    # Configurar o Grid Search
    grid_search = GridSearchCV(
        estimator=xgb, 
        param_grid=param_grid, 
        cv=2, 
        scoring='f1_weighted', 
        verbose=0,       # Pode mudar verbose para 0 para não poluir o terminal
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
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            eval_metric='logloss',
            random_state=42
        )
    else:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric='logloss',
            random_state=42
        )
        best_params = []

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, f1, best_params