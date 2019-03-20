from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import numpy as np
from math import log
import os


def load_data(path):
    text, label = [], []
    fake = open(path + '/clean_fake.txt', 'r')
    real = open(path + '/clean_real.txt', 'r')

    for line in fake.read().split('\n')[:-1]:
        text.append(line)
        label.append(0)                         # 0 represents fake
    for line in real.read().split('\n')[:-1]:
        text.append(line)                       # 1 represents fake
        label.append(1)

    global text_copy
    text_copy = text.copy()
    global label_copy
    label_copy = label.copy()

    train_size = int(0.7 * len(text))
    val_test_size = int((len(text) - train_size) / 2)

    # Vectorizer the input data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text).toarray()
    name = vectorizer.get_feature_names()

    # shuffle the whole data set simountinously
    s = np.arange(X.shape[0])
    np.random.seed(0)
    np.random.shuffle(s)
    X = X[s]
    y = np.array(label)
    y = y[s]

    X_train, X_validation, X_test = X[:train_size], X[train_size:train_size + val_test_size], X[train_size + val_test_size:]
    y_train, y_validation, y_test = y[:train_size], y[train_size:train_size + val_test_size], y[train_size + val_test_size:]

    return X_train, y_train, X_validation, y_validation, X_test, y_test, name


def select_model(X_train, y_train, X_validation, y_validation, max_depth, split_criteria):
    t = DecisionTreeClassifier(random_state=0, max_depth=max_depth, criterion=split_criteria)
    t.fit(X_train, y_train)
    y_pred = t.predict(X_validation)
    diff = y_validation - y_pred
    accuracy = 1 - np.sum(np.abs(diff)) / len(y_validation)

    return accuracy, t


def compute_information_gain(keyword):
    include, not_include = [], []
    fake_include, real_include, fake_not_include, real_not_include = 0, 0, 0, 0
    for index in range(len(text_copy)):
        if keyword in text_copy[index].split():
            include.append((text_copy[index], label_copy[index]))
        else:
            not_include.append((text_copy[index], label_copy[index]))
    for data in include:
        if data[1] == 0:
            fake_include += 1
        else:
            real_include += 1

    for data1 in not_include:
        if data1[1] == 0:
            fake_not_include += 1
        else:
            real_not_include += 1

    p_fake = (fake_not_include + fake_include) / len(text_copy)
    p_real = (real_not_include + real_include) / len(text_copy)
    p_fake_in = fake_include / len(text_copy)
    p_fake_not = fake_not_include / len(text_copy)
    p_real_in = real_include / len(text_copy)
    p_real_not = real_not_include / len(text_copy)

    p_fakegin = fake_include / len(include)
    p_fakegnot = fake_not_include / len(not_include)
    p_realgin = real_include / len(include)
    p_realgnot = real_not_include / len(not_include)

    HY = - p_fake * log(p_fake, 2) - p_real * log(p_real, 2)
    HYgX = - p_fake_in * log(p_fakegin, 2) - p_fake_not * log(p_fakegnot, 2) - p_real_in * log(p_realgin, 2) - p_real_not * log(p_realgnot, 2)

    return HY - HYgX


if __name__ == '__main__':
    news = load_data(os.path.dirname(os.path.realpath(__file__)))

    best = 0
    tree = None
    for i in range(5):
        model1 = select_model(news[0], news[1], news[2], news[3], i + 1, 'gini')
        print("Model with Gini coefficient and max depth " + str(i + 1) + ": " + str( model1[0]))
        model2 = select_model(news[0], news[1], news[2], news[3], i + 1, 'entropy')
        print("Model with information gain and max depth " + str(i + 1) + ": " + str(model1[0]))
        if model1[0] >= best:
            best = model1[0]
            tree = model1[1]
        if model2[0] >= best:
            best = model1[0]
            tree = model2[1]

    print("Best model with Accuracy: " + str(best))

    # visulize the tree
    export_graphviz(tree, out_file='tree.dot',
                    rounded=True, proportion=False,
                    precision=2, filled=True, feature_names=news[6], max_depth=2)
    print('')
    print('-----------------------------------------------------------')
    print('')
    print('Information gain for the is: ' + str(compute_information_gain('the')))
    print('Information gain for trumps is: ' + str(compute_information_gain('trumps')))
    print('Information gain for donald is: ' + str(compute_information_gain('donald')))
    print('Information gain for hillary is: ' + str(compute_information_gain('hillary')))
    print('Information gain for de is: ' + str(compute_information_gain('de')))
    print('Information gain for market is: ' + str(compute_information_gain('market')))
