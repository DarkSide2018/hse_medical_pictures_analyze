import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


def create_best_params(X_train, y_train):
    study = optuna.create_study(direction="maximize")

    multi_roc = make_scorer(roc_auc_score, average='weighted', multi_class='ovr', needs_proba=True)

    def objective_cat_boost(trial):
        max_depth = trial.suggest_int("max_depth", 2, 14)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        n_estimators = trial.suggest_int("n_estimators", 10, 600)

        score = cross_val_score(CatBoostClassifier(max_depth=max_depth,
                                                   learning_rate=learning_rate,
                                                   bagging_temperature=0.1,
                                                   l2_leaf_reg=0.1,
                                                   n_estimators=n_estimators),
                                X_train,
                                y_train,
                                cv=3,
                                scoring=multi_roc,
                                n_jobs=-1).mean()
        return score

    study.optimize(objective_cat_boost, n_trials=30)

    best_params = study.best_params
    return best_params
