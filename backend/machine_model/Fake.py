import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pickle
import pathlib
pth = str(pathlib.Path(__file__).parent.resolve())+"\\"


def getValues(data, columns):
    data = data[columns]
    return data


def getFloatValues(data):

    for i in range(len(data["url"])):
        if type(data["url"][i]) == str:
            data["url"][i] = 1

        if type(data["time_zone"][i]) == str:
            data["time_zone"][i] = 1

    data = data.values
    data = data.astype(np.float64)
    nanValues = np.isnan(data)
    data[nanValues] = 0
    return data


def normalize(data):

    for i in range(len(data)):
        denom = max(data[i])
        if (denom == 0):
            denom = 1
        data[i] = data[i]/denom

    return data


# ############ Read data ##############
# data = dict()
# data["fake"] = pd.read_csv("DATASET/fusers.csv")
# data["legit"] = pd.read_csv("DATASET/users.csv")

# legitLen = len(data["legit"])
# fakeLen = len(data["fake"])
# ############ Getting relevant features ##############
# columns = ["statuses_count", "followers_count", "friends_count",
#            "favourites_count", "listed_count", "url", "time_zone"]

# data["fake"] = getValues(data["fake"], columns)
# data["legit"] = getValues(data["legit"], columns)

# ############ Checking url and timezone and editing their values and removing null values ##############

# data["fake"] = getFloatValues(data["fake"])
# data["legit"] = getFloatValues(data["legit"])


# ############ Merge Fake and Legit #################

# data = np.concatenate((data["legit"], data["fake"]))

# ############ Normalize the values #################

# data = normalize(data)

# ############ Creating X and Y trains ##############
# X = data
# Y = np.zeros(legitLen + fakeLen)
# Y[0:legitLen] = 0
# Y[legitLen:legitLen+fakeLen] = 1

# ############ Spliting data into training and testing trains #############

# X_train, X_test, y_train, y_test = train_test_split(
#     X, Y, test_size=0.24, random_state=42)


# ############NaiveBayes###########################
# model1 = MultinomialNB()
# model1.fit(X_train, y_train)
# predictions_model1 = model1.predict(X_test)
# accuracy = accuracy_score(y_test, predictions_model1)*100
# precision = precision_score(y_test, predictions_model1, average='binary')
# recall = recall_score(y_test, predictions_model1, average='binary')
# F1 = 2 * (precision * recall) / (precision + recall)

# ############LinearSVC###############################
# model2 = LinearSVC()
# model2.fit(X_train, y_train)
# predictions_model2 = model2.predict(X_test)
# accuracy = accuracy_score(y_test, predictions_model2)*100
# precision = precision_score(y_test, predictions_model2, average='binary')
# recall = recall_score(y_test, predictions_model2, average='binary')
# F1 = 2 * (precision * recall) / (precision + recall)


# ############KNNClassifier####################################
# model3 = KNeighborsClassifier(n_neighbors=3)
# model3.fit(X_train, y_train)
# predictions_model3 = model3.predict(X_test)
# accuracy = accuracy_score(y_test, predictions_model3)*100
# precision = precision_score(y_test, predictions_model3, average='binary')
# recall = recall_score(y_test, predictions_model3, average='binary')
# F1 = 2 * (precision * recall) / (precision + recall)


# ################LogisticRegression#################################
# model4 = LogisticRegression(random_state=0)
# model4.fit(X_train, y_train)
# preds4 = model4.predict(X_test)
# accuracy = accuracy_score(y_test, preds4)*100
# precision = precision_score(y_test, preds4, average='binary')
# recall = recall_score(y_test, preds4, average='binary')
# F1 = 2 * (precision * recall) / (precision + recall)


# ################# XGBOOST ###############################
# xgb_cl = xgb.XGBClassifier()
# # Fit
# xgb_cl.fit(X_train, y_train)
# # Predict
# preds5 = xgb_cl.predict(X_test)
# # Score
# acc = accuracy_score(y_test, preds5)
# precision = precision_score(y_test, preds5, average='binary')
# recall = recall_score(y_test, preds5, average='binary')
# F1 = 2 * (precision * recall) / (precision + recall)


def detectAccount(userData):
    filename = pth+'finalized_model_Fakeaccounts.sav'
    if type(userData[0][5]) == str:
        userData[0][5] = 1

    if type(userData[0][6]) == str:
        userData[0][6] = 1

    userData = pd.DataFrame(userData)
    # userData = getFloatValues(userData)

    userData = userData.values
    userData = userData.astype(np.float64)
    nanValues = np.isnan(userData[0])
    userData[0][nanValues] = 0
    # bound = max(userData[0])
    # if bound == 0:
    #     bound = 1
    # userData[0] = userData[0]/bound
    userData = normalize(userData)
    loaded_model = pickle.load(open(filename, 'rb'))

    preds = loaded_model.predict(userData)

    pred = str(preds[0])

    return pred


# userData = [[112, 113, 0, 40, 0, None, None]]
# print(detectAccount(userData))
