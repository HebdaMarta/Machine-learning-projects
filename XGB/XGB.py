import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import sklearn.model_selection as msel
from matplotlib.pylab import rcParams

data = pd.read_csv("heart.csv")
data.info()

data = data.replace('?', np.NaN)
print(data)

columns_number = data.isna().sum(axis=0)
print(columns_number)

data = data.drop(['chol', 'slope', 'ca', 'thal'], axis = 1)
data

data = data.fillna(method = 'backfill')

#  Sprawdzenie
columns_number = data.isna().sum()
print(columns_number)

print(data.columns)

data = data.rename(columns={'num       ': 'num'})
data = data.astype(str).astype(float)

data.info()

fig = plt.figure(figsize = (15,20))
ax = fig.gca()
data.hist(ax = ax)
plt.show()


y = data['num']
x = data.drop(['num'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

clf1 = xgb.XGBClassifier(objective="binary:logistic", max_depth=3,
                        learning_rate=0.01, n_estimators=25, use_label_encoder=False)
clf2 = xgb.XGBClassifier(objective="binary:logistic", max_depth=3,
                        learning_rate=0.03, n_estimators=25, use_label_encoder=False)
clf3 = xgb.XGBClassifier(objective="binary:logistic", max_depth=3,
                        learning_rate=0.05, n_estimators=25, use_label_encoder=False)
clf4 = xgb.XGBClassifier(objective="binary:logistic", max_depth=3,
                         learning_rate=0.1, n_estimators=25, use_label_encoder=False)
clf5 = xgb.XGBClassifier(objective="binary:logistic", max_depth=3,
                         learning_rate=0.3, n_estimators=25, use_label_encoder=False)
clf6 = xgb.XGBClassifier(objective="binary:logistic", max_depth=3,
                         learning_rate=0.5, n_estimators=25, use_label_encoder=False)
clf7 = xgb.XGBClassifier(objective="binary:logistic", max_depth=3,
                         learning_rate=0.8, n_estimators=25, use_label_encoder=False)

models = {
    0.1: clf1,
    0.2: clf2,
    0.3: clf3,
    0.4: clf4,
    0.5: clf5,
    0.6: clf6,
    0.7: clf7
}

scores = {}

for lr, clf in models.items():
    score_array = msel.cross_val_score(clf, x_train, y_train, cv=4, scoring ='accuracy')
    print(lr, score_array.mean(), "+-", score_array.std())
    scores[lr] = score_array.mean()


scores_lr, scores_values = zip(*scores.items())
max_lr = scores_lr[scores_values.index(max(scores_values))]
max_values = max(scores_values)
print("Maximum score lr", max_lr, ':', max_values)

lr_array = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.8]
plt.plot(lr_array, scores_values)
plt.xlabel("Learning rate")
plt.ylabel("Score")
plt.show()

data_dm = xgb.DMatrix(data=x, label=y)
params = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "booster": "gbtree",
    "learning_rate": 0.5
}
cv_results = xgb.cv(dtrain=data_dm, params=params, num_boost_round=250,
                        metrics=["auc", "logloss"], as_pandas=True, seed=123)
print(cv_results)

plt.figure(figsize=(15, 7))
plt.title("Błąd na danych treningowych i testowych")
plt.plot(cv_results["train-logloss-mean"], color="b")
plt.plot(cv_results["test-logloss-mean"], color="r")
plt.xlabel("Ilość klasyfikatorów")
plt.ylabel("Wartość funkcji błędu logloss")
plt.show()
# The lower the value the better

plt.figure(figsize=(15, 7))
plt.title("Błąd na danych treningowych i testowych")
plt.plot(cv_results["train-auc-mean"], color="b")
plt.plot(cv_results["test-auc-mean"], color="r")
plt.xlabel("Ilość klasyfikatorów")
plt.ylabel("Wartość funkcji błędu auc")
plt.show()
# The higher the value the better


xgb_model = xgb.XGBClassifier(objective="binary:logistic", learning_rate = 0.5, n_estimators=7, max_depth=3, use_label_encoder=False)
xgb_model.fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

#  Let's also check the average of 10 iterations for roc_auc
model_auc = []

for iteration in range(10):
    print(iteration)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = xgb.XGBClassifier(objective="binary:logistic", learning_rate = 0.5, n_estimators=7, max_depth=3, use_label_encoder=False)
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)
    model_auc.append(roc_auc_score(y_test, predict))

print("Średni wynik modelu: ", np.mean(model_auc))

# Feature importance graph

rcParams['figure.figsize'] = 10, 8

xgb.plot_importance(xgb_model)

plt.show()
