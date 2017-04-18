"""
===================================================
Faces recognition example using other ensemble methods and pca
pca is used to extract features 

Adapted from 
http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html#sphx-glr-auto-examples-applications-face-recognition-py
and
http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-multiclass-py

Author : zhangxiaode   1601214529

Date: 20170417

Email: zhangxiaode@pku.edu.cn
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/


"""
from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.externals.six.moves import zip
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from itertools import product
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
print("-------------------------")
print("Size of training set: ", X_train.shape[0])
print("Size of testing set: ", X_test.shape[0])

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 100

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


################################################################################
randomforest_clf = RandomForestClassifier(n_estimators=80,max_depth = 10)
"""
class sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
"""
decisiontree_clf = DecisionTreeClassifier(max_depth=10, min_samples_split=2,random_state=0)
"""
class sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False)
"""
extratree_clf = ExtraTreesClassifier(n_estimators=20, max_depth=None,min_samples_split=2, random_state=0)
"""
class sklearn.ensemble.ExtraTreesClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
"""
bagging_clf = BaggingClassifier(KNeighborsClassifier(),n_estimators=20,max_samples=0.5, max_features=0.5)
"""
class sklearn.ensemble.BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)
"""

clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
voting_clf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[1,1,5])

clf1 = clf1.fit(X,y)
clf2 = clf2.fit(X,y)
clf3 = clf3.fit(X,y)
voting_clf = voting_clf.fit(X_train_pca, y_train)

boosting_clf = GradientBoostingClassifier(n_estimators=350, learning_rate=0.8,max_depth=3, random_state=0)
"""
class sklearn.ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
"""
randomforest_clf=randomforest_clf.fit(X_train_pca, y_train)
decisiontree_clf=decisiontree_clf.fit(X_train_pca, y_train)
extratree_clf = extratree_clf.fit(X_train_pca, y_train)
bagging_clf = bagging_clf.fit(X_train_pca, y_train)
boosting_clf =boosting_clf .fit(X_train_pca, y_train)
#################################################################################

y_randomforest_pred = randomforest_clf.predict(X_test_pca)
print("""--------The results of RandomForestClassifier----------------""")
print(classification_report(y_test, y_randomforest_pred, target_names=target_names))
print(confusion_matrix(y_test, y_randomforest_pred, labels=range(n_classes)))

y_decisiontree_pred = decisiontree_clf.predict(X_test_pca)
print("""--------The results of DecisionTreeClassifier----------------""")
print(classification_report(y_test, y_decisiontree_pred, target_names=target_names))
print(confusion_matrix(y_test, y_decisiontree_pred, labels=range(n_classes)))

y_extratree_pred = extratree_clf.predict(X_test_pca)
print("""--------The results of ExtraTreesClassifier----------------""")
print(classification_report(y_test, y_extratree_pred, target_names=target_names))
print(confusion_matrix(y_test, y_extratree_pred, labels=range(n_classes)))

y_bagging_pred = bagging_clf.predict(X_test_pca)
print("""--------The results of BaggingClassifier----------------""")
print(classification_report(y_test, y_bagging_pred, target_names=target_names))
print(confusion_matrix(y_test, y_bagging_pred, labels=range(n_classes)))

y_voting_pred = voting_clf.predict(X_test_pca)
print("""--------The results of VotingClassifier----------------""")
print(classification_report(y_test, y_voting_pred, target_names=target_names))
print(confusion_matrix(y_test, y_voting_pred, labels=range(n_classes)))

y_boosting_pred = boosting_clf.predict(X_test_pca)
print("""--------The results of GradientBoostingClassifier----------------""")
print(classification_report(y_test, y_boosting_pred, target_names=target_names))
print(confusion_matrix(y_test, y_boosting_pred, labels=range(n_classes)))

