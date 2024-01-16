import imp
import re
import joblib
import pickle
import pandas as pd
import numpy as np
from pandas import DataFrame
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
# from videoToText import path
import pathlib
pth = str(pathlib.Path(__file__).parent.resolve())+"\\"

##############################################################################
#                               READING DATASET FUNCTION


def Reading_dataset(filename):
    data = pd.read_csv(filename, sep=",", error_bad_lines=False)
    data.columns = ["text", "Hatespeech"]
    return data

############################################################
#                               PREPROCESSING


def normalize(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("_", " ", text)
    text = re.sub("#", " ", text)
    text = re.sub("@USER", " ", text)
#     text = re.sub("ؤ", "و", text)
    #text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    noise = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(noise, '', text)
    return(text)


def stopwordremoval(text):
    stop = stopwords.words("arabic")
    needed_words = []
    words = word_tokenize(text)
    for w in words:
        if len(w) >= 2 and w not in stop:
            needed_words.append(w)
    filterd_sent = " ".join(needed_words)
    return filterd_sent


def removenonarabic(text):
    # n=re.sub('([@A-Za-z0-9_]+)|[^\w\s]|#|http\S+', '', text)
    n = re.sub(r'[^،-٩]+', ' ', text)
    n = re.sub('\W+', ' ', n)
    n = re.sub('_', '', n)
    n = ''.join([i for i in n if not i.isdigit()])
    return n


def stemming(text):
    st = ISRIStemmer()
    stemmed_words = []
    words = word_tokenize(text)
    for w in words:
        stemmed_words.append(st.stem(w))
    stemmed_sent = " ".join(stemmed_words)
    return stemmed_sent


def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)


def lemmatization(txt):
    lemmatizer = nltk.WordNetLemmatizer()
    """
    Lemmatize using WordNet's morphy function.
    Returns the input word unchanged if it cannot be found in WordNet.
    :param txt: string : arabic text
    :return: lemmas : array : array contains a Lemma for each word in the text.
    """
    words = word_tokenize(txt)
    lemmas = str([lemmatizer.lemmatize(w) for w in words])
    return lemmas
##################################################################################


def Prepare_datasets(data):
    sentences = []
    for index, r in data.iterrows():
        text = normalize(r['text'])
        text = stopwordremoval(text)
        text = removenonarabic(text)
        text = remove_repeating_char(text)
#         text=lemmatization(text)
        text = stemming(text)
        sentences.append([text, r['Hatespeech']])
    df_sentence = DataFrame(sentences, columns=["text", "Hatespeech"])
    return df_sentence
##############################################################################


def Prepare_testing_sentence(data):
    # sentences=[]
    # for index,r in data.iterrows():
    text = normalize(data)
    text = stopwordremoval(text)
    text = removenonarabic(text)
    text = remove_repeating_char(text)
#         text=lemmatization(text)
    text = stemming(text)
    # sentences.append(text)
    # df_sentence=DataFrame(sentences, columns=["text", "Hatespeech"])
    return text
#########################################################################################################################################
#                               FEATURES EXTRACTION METHODS

###################################################### TFIDF ###################################################################


def TFIDF_Train(data, ngrams=1):
    tfidfconverter = TfidfVectorizer(ngram_range=(1, ngrams))
    train_data_tfidf = tfidfconverter.fit_transform(data).toarray()
    filename = 'tfidf_model.pkl'
    joblib.dump(tfidfconverter, filename)
    return train_data_tfidf


def TFIDF_Test(data):
    loaded_model = joblib.load(open('tfidf_model.pkl', 'rb'))
    test_data_tfidf = loaded_model.transform(data).toarray()
    return test_data_tfidf
###################################################### BoW #####################################################################


def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_tokenize(sentence)
        words.extend(w)
        words = sorted(list(set(words)))
    return words


def BOW(data, words):
    bag_vectors = []
    for sentence in data:
        word_s = word_tokenize(sentence)
        bag_vector = numpy.zeros(len(words))
        for w in word_s:
            for i, word in enumerate(words):
                if word == w:
                    bag_vector[i] += 1
        bag_vectors.append(bag_vector)
    return bag_vectors

###################################################### CHI-square ##################################################################


def chi_squared(txt):
    TWC = {}  # Two-way contingency dictionary
    feature = {}  # returned feature
    for index, row in txt.iterrows():
        words = set(word_tokenize(row['text']))
        c = row['Hatespeech']
        for w in words:
            if (w, c) not in TWC:
                TWC[(w, c)] = 1
            else:
                TWC[(w, c)] += 1
    for word in words:
        temp = [(item[1], TWC[item]) for item in TWC.keys() if item[0] == word]
        N = len(data)  # number of tweets
        # count of tweets contains this word
        count_word = sum(n for _, n in temp)
        count_not_word = N-count_word
        for cat in set(data.Hatespeech):
            #             temp=[(item[1],TWC[item]) for item in TWC.keys() if item[0]==word[0]]
            # get counts
            #             count_cat=cat.value_counts()           ########### count of tweets in this category
            details = data.apply(lambda x: True
                                 if x['Hatespeech'] == cat else False, axis=1)

            # Count number of True in the series
            count_cat = len(details[details == True].index)
            count_not_cat = N-count_cat

            count_word_cat = 0
            for item in temp:
                if item[0] == cat:
                    count_word_cat = item[1]
            count_word_not_cat = count_word-count_word_cat
            count_not_word_cat = count_cat-count_word_cat
            count_not_word_not_cat = N-(count_cat+count_word-count_word_cat)

            # get terms to be added

            TTterm = pow((count_word_cat-(count_word*count_cat/N)),
                         2)/(count_word*count_cat/N)
            TFterm = pow((count_word_not_cat-(count_word*count_not_cat/N)),
                         2)/(count_word*count_not_cat/N)
            FTterm = pow((count_not_word_cat-(count_not_word *
                                              count_cat/N)), 2)/(count_not_word*count_cat/N)
            FFterm = pow((count_not_word_not_cat-(count_not_word *
                                                  count_not_cat/N)), 2)/(count_not_word*count_not_cat/N)

            # sum of terms
            feature[(word, cat)] = TTterm+TFterm+FTterm+FFterm
####################################################### BoW #####################################################################


def BOW_Train(data):
    vectorizer = CountVectorizer()
    vectorizer.fit(data)
    bow_data = vectorizer.transform(data)
    filename = 'bow_model.pkl'
    joblib.dump(vectorizer, filename)
    bow_data_dataframe = pd.DataFrame(
        bow_data.toarray(), columns=vectorizer.get_feature_names())
    return bow_data_dataframe


def BOW_Test(data):
    loaded_model = joblib.load(open('bow_model.pkl', 'rb'))
    bow_data = loaded_model.transform(data)
    bow_data_dataframe = pd.DataFrame(
        bow_data.toarray(), columns=loaded_model.get_feature_names())
    return bow_data_dataframe


######################################################################################################
#           READING TRAINING DATASET
# Traindataset = Reading_dataset("Traindataset.csv")

#               Test data
#Testdata= Reading_dataset("Testdataset.csv")
######################################################################################################
#       APPLYING PREPROCESSING
# Traindata_Preprocessed = Prepare_datasets(Traindataset)


#######################################################################################################
#                       Solving Unbalanced data problem

def Smote_TFIDFTrain(data, ngrams=1):

    df_temp = data.copy(deep=True)

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, ngrams))
    tfidf_vectorizer.fit(df_temp['text'])
    filename = 'tfidf_model_smote.pkl'
    joblib.dump(tfidf_vectorizer, filename)
    list_corpus = df_temp["text"].tolist()
    list_labels = df_temp["Hatespeech"].tolist()

    X = tfidf_vectorizer.transform(list_corpus)

    return X, list_labels


def Smote_TFIDFTest(data):
    print("333333")
    data = [data]
    loaded_model = joblib.load(open(pth+'tfidf_model_smote.pkl', 'rb'))

    print("44444")
    test_data_tfidf = loaded_model.transform(data)
    print("555555555")
    return test_data_tfidf


def smote_oversampling(x, y):
    smote = SMOTE(k_neighbors=3)
    print("11111")
    X, y = smote.fit_resample(x, y)
    print("222222")
    # counter = Counter(y)
    # print(sum(counter.values()))
    # for k,v in counter.items():
    #     per = v / len(y) * 100
    #     print('Class=%s, n=%d (%.3f%%)' % (k, v, per))
    # pyplot.bar(counter.keys(), counter.values())
    # pyplot.show()
    return X, y


######################################################################################################
# X_TFIDF, Y = Smote_TFIDFTrain(Traindata_Preprocessed)
# X_train, Y_train = smote_oversampling(X_TFIDF, Y)


######################################################################################################
#                       Classify using TFIDF feature after balancing data

def random_search_RF(x, y, k):
    # Function to tune Random Forest hyper parameters using Randomized Grid Search
    # x: train data
    # y: labels
    # k: cross-validation folds

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]

    # Number of features to consider at every split
    max_features = ['log2', 'sqrt']

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)

    criterion = ["gini", "entropy", "log_loss"]

    class_weight = ['balanced']

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'class_weight': class_weight,
                   'criterion': criterion,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 5 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=100, cv=k, verbose=2, random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(x, y)

    return rf_random.best_params_


def grid_search_RF(x, y, k):
    # Function to tune Random Forest hyper parameters using Grid Search based on the previous random search
    # x: train data
    # y: labels
    # k: cross-validation folds

    # Create the parameter grid based on the results of random search
    param_grid = {
        'bootstrap': [False],
        'max_depth': [80, 90, 100, 110, 120],
        'max_features': ['log2'],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [1, 2, 4],
        'n_estimators': [600, 800, 1000],
    }
    # Create a base model
    rf = RandomForestClassifier()

    # Instantiate the grid search model
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=k, n_jobs=-1, verbose=2)
    grid_search.fit(x, y)

    return grid_search.best_params_
# 3


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('object')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


# X_train=np.asarray(data)
# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_train, Y_train, test_size=0.20, random_state=0)


################################################################################

# rf_clf = RandomForestClassifier(n_estimators=800, min_samples_split=2, min_samples_leaf=1,
#                                 bootstrap=False, max_features='log2', max_depth=None, criterion='entropy')
# rf_clf.fit(X_train, Y_train)
# save the model to disk


def testing(sentence):

    filename = pth+'finalized_model.sav'
    Testdata_Preprocessed = Prepare_testing_sentence(sentence)
    print("Preprocessed Text: ", Testdata_Preprocessed)
    X_test = Smote_TFIDFTest(Testdata_Preprocessed)
    print(X_test)
    loaded_model = pickle.load(open(filename, 'rb'))
    rf_prediction = loaded_model.predict(X_test)
    print("RESULT", rf_prediction)
    return rf_prediction[0]


# testtt = testing(
#     "#انا_مش_حتجوز_علشان الرجاله نكديين وخاينين وكدابيين <LF>ونحط تحت كدابيين دي 100 خط🖐😡")

# print("Result", testtt)

# Y_test=Testdata_Preprocessed['Hatespeech']
# pickle.dump(rf_clf, open('finalized_model.sav', 'wb'))

# load the model from disk
# rf_prediction = rf_clf.predict(X_test)
# rf_clf.score(X_test, Y_test)
# print("accuracy", accuracy_score(Y_test, rf_prediction))
# print("f1_score", f1_score(Y_test, rf_prediction, average='weighted'))
# print("recall", recall_score(Y_test, rf_prediction, average='weighted'))
# print("precision", precision_score(Y_test, rf_prediction, average='weighted'))
# print('\n clasification report:\n', classification_report(Y_test, rf_prediction))
# print(' confussion matrix:\n', confusion_matrix(Y_test, rf_prediction))

##############################################################################################

# svm_clf = SVC(C=1024, gamma=0.125, kernel='rbf')
# svm_clf.fit(X_train,Y_train)
# svm_prediction=svm_clf.predict(X_test)
# svm_clf.score(X_test,Y_test)
# print("accuracy",accuracy_score(Y_test, svm_prediction))
# print("f1_score",f1_score(Y_test, svm_prediction,average='weighted'))
# print("recall",recall_score(Y_test, svm_prediction,average='weighted'))
# print("precision",precision_score(Y_test, svm_prediction,average='weighted'))
# print ('\n clasification report:\n', classification_report(Y_test, svm_prediction))
# print (' confussion matrix:\n',confusion_matrix(Y_test, svm_prediction))

# 3


# from sklearn.model_selection import StratifiedKFold
# def cross_validation(model, k):

#   # Function to cross validate using Stratified K-folds
#   # model: the regressor to train
#   # k: number of folds

#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
#     r2_avg = []
#     rmsle_avg = []
#     for train_index, test_index in skf.split(X_train,Y_train):

#         x_train_fold, x_test_fold = X_train[train_index], X_train[test_index]
#         y_train_fold, y_test_fold = np.array(Y_train)[train_index], np.array(Y_train)[test_index]
#         model.fit(x_train_fold, np.ravel(y_train_fold))
#         y_pred = model.predict(x_test_fold)
#         model.score(x_test_fold,y_test_fold)
#         print("accuracy",accuracy_score(y_test_fold, y_pred))
#         print("f1_score",f1_score(y_test_fold, y_pred,average='weighted'))
#   #     r2 = metrics.r2_score(y_test_fold,y_pred)*100
#   #     r2_avg.append(r2)
#   #     rmsle = np.sqrt(metrics.mean_squared_log_error(y_test_fold, y_pred))
#   #     rmsle_avg.append(rmsle)
#   #     print('R2 Score: ', r2, '% ', 'RMSLE: ', rmsle)
#   # r2_avg = np.asarray(r2_avg)
#   # rmsle_avg = np.asarray(rmsle_avg)

#   # print("Finished Validation for " , k, " Folds..")
#   # print("Average R2 Score: ", np.sum(r2_avg)/len(r2_avg))
#   # print("Average RMSLE: ", np.sum(rmsle_avg)/len(rmsle_avg))

# cross_validation(rf_clf,5)
