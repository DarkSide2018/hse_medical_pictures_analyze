import pickle

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from year_project.telegram_bot.functions import get_skin_problems_dataset
from year_project.telegram_bot.notify_bot_service import notify_bot
from year_project.telegram_bot.optuna_optimizer import create_best_params

try:

    dataframe_train = get_skin_problems_dataset("train")
    print(dataframe_train.columns)
    dataframe_test = get_skin_problems_dataset("test")
    print(dataframe_test)
    y_train = dataframe_train['target']
    dataframe_train.drop('target', axis=1, inplace=True)
    y_test = dataframe_test['target']
    dataframe_test.drop('target', axis=1, inplace=True)

    scaler = StandardScaler()

    dataframe_train_scaled = scaler.fit_transform(dataframe_train)
    dataframe_test_scaled = scaler.fit_transform(dataframe_test)
    X_train = dataframe_train_scaled
    X_test = dataframe_test_scaled

    notify_bot(f"CatBoostClassifier started optuna")

    best_params = create_best_params(X_train, y_train)

    model = CatBoostClassifier(max_depth=best_params['max_depth'],
                               eval_metric='HammingLoss',
                               learning_rate=best_params['learning_rate'],
                               n_estimators=best_params['n_estimators'],
                               early_stopping_rounds=10,
                               random_seed=42,
                               loss_function='MultiClass',
                               bagging_temperature=best_params['bagging_temperature'],
                               classes_count=23,
                               thread_count=-1,
                               l2_leaf_reg=0.1,
                               class_names=list(range(23)),
                               verbose=200
                               )

    model.fit(X_train, y_train)

    with open('cat_boost_model.pickle', 'wb') as f:
        pickle.dump(model, f)

    pickled_model = pickle.load(open('cat_boost_model.pickle', 'rb'))

    pred_cb = pickled_model.predict_proba(X_test)
    predicted = pickled_model.predict(X_test)

    train_pool = Pool(X_train, label=y_train)

    print(f'Xtrain describe: {X_train}')

    feature_importance = pickled_model.get_feature_importance(data=train_pool, type="FeatureImportance")
    prediction_values_change = pickled_model.get_feature_importance(data=train_pool, type="PredictionValuesChange")
    loss_function_change = pickled_model.get_feature_importance(data=train_pool, type="LossFunctionChange")

    print(f"feature_importance: {feature_importance}")

    r2_score = r2_score(y_test, predicted)

    print(f"r2_score cat_boost : {r2_score}")

    accuracy = accuracy_score(y_test, predicted)

    print("Accuracy cat_boost:", accuracy)

    roc_auc = roc_auc_score(y_test, pred_cb, multi_class='ovr')

    print(f"roc_auc cat_boost : {roc_auc}")
    feature_names = dataframe_train.columns
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    loss_function_change = pd.DataFrame({'feature': feature_names, 'importance': loss_function_change})
    prediction_values_change = pd.DataFrame({'feature': feature_names, 'importance': prediction_values_change})
    report = f'''
       
        cat_boost report:
        roc_auc_score : {roc_auc}
        Accuracy : {accuracy}
        r2_score : {r2_score}
        feature_importance : {feature_importance}
        loss_function_change: {loss_function_change}
        prediction_values_change: {prediction_values_change}
        best_params: {best_params}
        '''
    notify_bot(report)

except Exception as error:
    print("An exception occurred during CatBoostClassifier main process:", error)
    notify_bot(f"An exception occurred during CatBoostClassifier main process: {error}")
