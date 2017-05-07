import time
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

data = datasets.load_breast_cancer()
#data = datasets.load_iris()
#data = datasets.load_digits()

classifiers = []

dt = tree.DecisionTreeClassifier()
classifiers.append([dt, "tree"])

svm = LinearSVC(random_state=1)
classifiers.append([svm, "svm"])

ovo = OneVsOneClassifier(LinearSVC(random_state=1))
classifiers.append([ovo, "one-vs-one svm"])

ovr = OneVsRestClassifier(LinearSVC(random_state=1))
classifiers.append([ovr, "one-vs-rest svm"])

for classifier, label in classifiers:
    start = time.time()
    scores = cross_val_score(classifier, data.data, data.target, cv=10)
    stop = time.time()
    print("%20s accuracy: %0.2f (+/- %0.2f), time:%.4f" % (label, scores.mean(), scores.std() * 2, stop - start))


