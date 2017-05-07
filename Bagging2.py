import time
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_mldata

data = datasets.load_breast_cancer()
#data = datasets.load_iris()
#data = datasets.load_digits()
#data = fetch_mldata('datasets-UCI credit-g')
#data = fetch_mldata('MNIST original')

classifiers = []

dt = tree.DecisionTreeClassifier()
classifiers.append([dt, "tree"])

bagged_dt = BaggingClassifier(tree.DecisionTreeClassifier())
classifiers.append([bagged_dt, "bagged tree"])

neighbors = 10;
knn = KNeighborsClassifier(n_neighbors=neighbors)
classifiers.append([knn, "knn"])

bagged_knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=neighbors))
classifiers.append([bagged_knn, "bagged knn"])

mlp = MLPClassifier(solver="lbfgs")
classifiers.append([mlp, "neural network"])

bagged_mlp = BaggingClassifier(MLPClassifier(solver="lbfgs"), n_jobs=2)
classifiers.append([bagged_mlp, "bagged neural network"])

svm = LinearSVC()
classifiers.append([svm, "linear svm"])

bagged_svm = BaggingClassifier(LinearSVC())
classifiers.append([bagged_svm, "bagged linear model"])


for classifier, label in classifiers:
    start = time.time()
    scores = cross_val_score(classifier, data.data, data.target, cv=10)
    stop = time.time()
    print("%20s accuracy: %0.2f (+/- %0.2f), time:%.4f" % (label, scores.mean(), scores.std() * 2, stop - start))


