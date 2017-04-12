from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_validation import cross_val_score
import numpy as np

from .preprocess import CoVectorizer


# All classifiers we are going to use
all_clfs = {
    'rbf_svc': GridSearchCV(SVC(kernel='rbf'),
                            {'C': np.logspace(-3.5, 3.5, 12),
                             'gamma': np.logspace(-3, 3, 12)}),
    'logit': GridSearchCV(LogisticRegression(penalty='l2', solver='liblinear'),
                          {'C': np.logspace(-3, 4, 20)}),
    'multi_nb': MultinomialNB(alpha=.01)
}


def train_clfs(x, y):
    """
    Train all the classifiers from all_clfs

    :param x: x data
    :param y: y data (categories)
    :return: trained classifiers
    """
    answer = dict()
    for clf_name, clf in all_clfs.items():
        if isinstance(clf, GridSearchCV):
            answer[clf_name] = clf.fit(x, y).best_estimator_
        else:
            answer[clf_name] = clf.fit(x, y)

    return answer


def train_mbclf_from_texts(texts, categories, surface_forms, w=30):
    co_vec = CoVectorizer(w=w, proximity_func=lambda x: (w - abs(x) + 1) / w)
    x = co_vec.fit_transform(texts, categories, surface_forms)
    normalizer = Normalizer()
    normed_x = normalizer.fit_transform(x)

    mb_clf = MultinomialNB(alpha=.01).fit(normed_x, categories)
    return co_vec, normalizer, mb_clf


def train_logit_from_texts(texts, categories, surface_forms, w=30):
    co_vec = CoVectorizer(w=w, proximity_func=lambda x: (w - abs(x) + 1) / w)
    x = co_vec.fit_transform(texts, categories, surface_forms)
    normalizer = Normalizer()
    normed_x = normalizer.fit_transform(x)

    logit_clf = GridSearchCV(
        LogisticRegression(penalty='l2', solver='liblinear'),
        {'C': np.logspace(-3, 4, 10)}
    )
    logit_clf.fit(normed_x, categories)
    return co_vec, normalizer, logit_clf.best_estimator_


def get_unknown_prediction_func(points_2d, method_name='x0'):
    """
    Prepare a predictor for an unknown class.

    :return: function: x -> [True, False]    - if x belongs to unknown class (True = yes, x is in unknown)
    """
    if method_name == 'hyp':
        measures = np.asarray(sorted([abs(x[0]*x[1]) for x in points_2d])).reshape((len(points_2d), 1))
    elif method_name == 'x0':
        measures = np.asarray(sorted([x[0] for x in points_2d])).reshape((len(points_2d), 1))
    elif method_name == 'circle':
        measures = np.asarray(sorted([(x[0]**2 + x[1]**2)**.5
                                      for x in points_2d])).reshape((len(points_2d), 1))
    else:
        raise ValueError('method name should be in [hyp, x0, circle], but it is {}'.format(method_name))

    typ_size = max(5, round(.1*len(measures)))
    nn = NearestNeighbors(n_neighbors=typ_size).fit(measures)
    dists, neighbors = nn.kneighbors(measures[:typ_size])
    proximity_size = round(typ_size/3)

    for i in range(proximity_size):
        if i not in neighbors[i+1][:proximity_size]:
            # outliers = list(range(i+1))
            first_cluster = list(range(i+1, typ_size+1))
            break
    else:
        # outliers = []
        first_cluster = list(range(typ_size+1))
    decision_border = (measures[first_cluster[0]] -
                       np.std(measures[:proximity_size+1]))
    if decision_border < 0:
        decision_border = 0

    if method_name == 'hyp':
        return lambda point: int(point[0]*point[1] > decision_border)
    elif method_name == 'x0':
        return lambda point: int(point[0] > decision_border)
    elif method_name == 'circle':
        return lambda point: int((point[0]**2 + point[1]**2)**.5 >
                                 decision_border and
                                 point[0] > 0)

def classify(texts, classes, new_text, surface_forms, w):
    """
    Make two predictions: if a new_text is from a known category and from which one.


    :param texts: list of texts
    :param classes: categories
    :param new_text: new text
    :param new_form: regular expressions representing the sought entity in new_text
    """
    co_vec = CoVectorizer(w=w, proximity_func=lambda x: (w - abs(x) + 1) / w)
    x = co_vec.fit_transform(texts, classes, surface_forms)
    normalizer = Normalizer()
    normed_x = normalizer.fit_transform(x)
    clfs = train_clfs(normed_x, classes)

    new_x = co_vec.transform([new_text])
    normed_new_x = normalizer.transform(new_x)
    prediction = dict()
    for clf_name in clfs:
        prediction[clf_name] = clfs[clf_name].predict(normed_new_x)[0]

    feature_names = np.asarray(co_vec.get_feature_names())
    fired_features = list(feature_names[new_x.nonzero()[1]])

    return prediction, fired_features