import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold

from utils import prepare_data_from_file, plot_results


# Exercise 1
def exercise_1():
    file = './data/phoneme.csv'
    X, y, _ = prepare_data_from_file(file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# Exercise 2
def exercise_2():
    file = './data/balance.csv'
    X, y, _ = prepare_data_from_file(file)

    kf = KFold(n_splits=10, shuffle=True)
    classifier = KNeighborsClassifier(n_neighbors=5)

    score_array = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        score_array.append(accuracy)

    avg_score = np.mean(score_array, axis=0)
    std_score = np.std(score_array, axis=0)

    print(avg_score)
    print(std_score)


# Exercise 3
def exercise_3():
    folder = './data/'
    ext = '.csv'
    file_names = ['balance', 'phoneme', 'sonar']

    classifiers = [
        (KNeighborsClassifier(n_neighbors=5), 'k-NN'),
        (SVC(kernel="linear", C=0.025), 'SVC'),
        (DecisionTreeClassifier(max_depth=5), 'Decision Tree'),
    ]

    data = {}
    for fn in file_names:
        X, y, _ = prepare_data_from_file(folder+fn+ext)
        data[fn] = {}

        for clf, clf_name in classifiers:
            kf = KFold(n_splits=10, shuffle=True)

            learning_times = []
            prediction_times = []
            score_array = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                start = time.time()
                clf.fit(X_train, y_train)
                end = time.time()
                learning_time = end-start
                learning_times.append(learning_time)

                start = time.time()
                y_pred = clf.predict(X_test)
                end = time.time()
                prediction_time = end-start
                prediction_times.append(prediction_time)

                accuracy = accuracy_score(y_test, y_pred)
                score_array.append(accuracy)

            avg_learning_time = np.mean(learning_times)
            avg_prediction_time = np.mean(prediction_times)
            avg_score = np.mean(score_array)

            data[fn][clf_name] = [avg_learning_time, avg_prediction_time, avg_score]

    for fn, clfs in data.items():
        classifier_names = [*clfs.keys()]
        values = [*clfs.values()]

        learning_times = list(map(lambda x: x[0], values))
        prediction_times = list(map(lambda x: x[1], values))
        scores = list(map(lambda x: x[2], values))

        plot_results(learning_times, scores, classifier_names, fn)


exercise_3()
# print(classification_report(y_test, y_pred))
