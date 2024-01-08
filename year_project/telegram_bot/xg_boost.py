import pickle

from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from year_project.telegram_bot.functions import data_dictionary
from year_project.telegram_bot.notify_bot_service import notify_bot

dataframe_train = data_dictionary(part="train").astype(float)
dataframe_test = data_dictionary(part="test").astype(float)

y_train = dataframe_train['target']

dataframe_train.drop('target', axis=1, inplace=True)

X_train = dataframe_train

y_test = dataframe_test['target']

dataframe_test.drop('target', axis=1, inplace=True)

X_test = dataframe_test

model = XGBClassifier(base_score=0.5, booster='gbtree', num_class=23,
                      colsample_bylevel=1,
                      colsample_bynode=1, colsample_bytree=1,
                      gamma=0,
                      max_depth=5,
                      max_delta_step=0, min_child_weight=1, missing=1,
                      n_jobs=1, nthread=None, seed=42,
                      objective='multi:softmax', random_state=0, reg_alpha=0,
                      reg_lambda=1, silent=None,
                      subsample=1, verbosity=1)

param_grid = {
    'learning_rate': [0.1],
    'n_estimators': [600]
}

grid_search = GridSearchCV(estimator=model, scoring='r2', param_grid=param_grid, cv=1)

grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_

with open('xgb_classifier.pickle', 'wb') as f:
    pickle.dump(model, f)

pickled_model = pickle.load(open('xgb_classifier.pickle', 'rb'))

predicted_probabilities = pickled_model.predict_proba(X_test)

predicted = pickled_model.predict(X_test)

roc_auc = roc_auc_score(y_test, predicted_probabilities, multi_class='ovr')

r2_score = r2_score(y_test, predicted)

accuracy = accuracy_score(y_test, predicted)

print("Accuracy cat_boost:", accuracy)
print(f"xg XGBClassifier roc_auc : {roc_auc}")

report = f'''
    XGBClassifier report:
    roc_auc_score : {roc_auc}
    Accuracy : {accuracy}
    r2_score : {r2_score}
    best_params : {grid_search.best_params_}
    best_score : {grid_search.best_score_}
    '''
notify_bot(report)
