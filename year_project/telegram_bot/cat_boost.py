import pickle

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from year_project.telegram_bot.functions import data_dictionary
from year_project.telegram_bot.notify_bot_service import notify_bot

model = CatBoostClassifier(max_depth=12,
                           eval_metric='HammingLoss',
                           early_stopping_rounds=10,
                           random_seed=42,
                           loss_function='MultiClass',
                           bagging_temperature=1.0,
                           classes_count=23,
                           thread_count=-1,
                           l2_leaf_reg=0.1,
                           class_names=list(range(23)),
                           verbose=200)

try:

    dataframe_train = data_dictionary(part="train")
    dataframe_test = data_dictionary(part="test")

    y_train = dataframe_train['target']

    dataframe_train.drop('target', axis=1, inplace=True)

    X_train = dataframe_train

    y_test = dataframe_test['target']

    dataframe_test.drop('target', axis=1, inplace=True)

    X_test = dataframe_test

    params = {
        'n_estimators': [500]
    }

    gs_cb = GridSearchCV(model, params, cv=1, scoring='roc_auc_ovr', verbose=2)

    notify_bot(f"CatBoostClassifier started grid search")

    gs_cb.fit(X_train, y_train)

    gs_cb_best_estimator_ = gs_cb.best_estimator_

    with open('gs_cb_best_estimator_.pickle', 'wb') as f:
        pickle.dump(gs_cb_best_estimator_, f)

    pickled_model = pickle.load(open('gs_cb_best_estimator_.pickle', 'rb'))

    pred_cb = pickled_model.predict_proba(X_test)
    predicted = pickled_model.predict(X_test)

    print(f"best_params cat_boost : {gs_cb.best_params_}")
    print(f"best_score cat_boost : {gs_cb.best_score_}")

    r2_score = r2_score(y_test, predicted)

    print(f"r2_score cat_boost : {r2_score}")

    accuracy = accuracy_score(y_test, predicted)

    print("Accuracy cat_boost:", accuracy)

    roc_auc = roc_auc_score(y_test, pred_cb, multi_class='ovr')

    print(f"roc_auc cat_boost : {roc_auc}")

    report = f'''
        cat_boost report:
        roc_auc_score : {roc_auc}
        Accuracy : {accuracy}
        r2_score : {r2_score}
        best_params : {gs_cb.best_params_}
        best_score : {gs_cb.best_score_}
        '''
    notify_bot(report)

except Exception as error:
    print("An exception occurred during CatBoostClassifier main process:", error)
    notify_bot(f"An exception occurred during CatBoostClassifier main process: {error}")
