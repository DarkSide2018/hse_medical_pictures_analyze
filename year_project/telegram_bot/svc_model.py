import pickle

from sklearn import svm
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV

from year_project.telegram_bot.functions import data_dictionary
from year_project.telegram_bot.notify_bot_service import notify_bot

notify_bot("SVC model started")

dataframe_train = data_dictionary(part="train")
dataframe_test = data_dictionary(part="test")

print(f"dataframe_train : {dataframe_train}")
print(f"dataframe_test: {dataframe_test}")

y_train = dataframe_train['target']

dataframe_train.drop('target', axis=1, inplace=True)

X_train = dataframe_train

y_test = dataframe_test['target']

dataframe_test.drop('target', axis=1, inplace=True)

X_test = dataframe_test

clf = svm.SVC(kernel='rbf', probability=True)

print(f"nulls in dataframe : {X_train.isnull().sum()}")

params = {
    'C': [1]
}

gs_svm_svc = GridSearchCV(clf, params, cv=1, scoring='r2', verbose=2)

notify_bot(f"svm.SVC started grid search")

gs_svm_svc.fit(X_train, y_train)

gs_svm_svc_best_estimator = gs_svm_svc.best_estimator_

with open('svm_svc_best_estimator.pickle', 'wb') as f:
    pickle.dump(gs_svm_svc_best_estimator, f)

pickled_model = pickle.load(open('svm_svc_best_estimator.pickle', 'rb'))

pred_cb = pickled_model.predict_proba(X_test)
predicted = pickled_model.predict(X_test)

r2_score = r2_score(y_test, predicted)

accuracy = accuracy_score(y_test, predicted)

roc_auc = roc_auc_score(y_test, pred_cb, multi_class='ovr')

report = f'''
      SVM machine report:
      roc_auc_score : {roc_auc}
      Accuracy : {accuracy}
      r2_score : {r2_score}
      best_params : {gs_svm_svc.best_params_}
      best_score : {gs_svm_svc.best_score_}
      '''
notify_bot(report)
