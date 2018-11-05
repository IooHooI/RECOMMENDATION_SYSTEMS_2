from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import multiprocessing


class XGBoostBasedRecommender(BaseEstimator, ClassifierMixin):
    def __init__(self, val_size=0.3):
        self.val_size = val_size

        self.model = XGBClassifier(n_jobs=multiprocessing.cpu_count())

    def fit(self, X, y=None):
        self.model.fit(X=X, y=y)

        return self

    def predict_proba(self, X, y=None):
        return self.model.predict_proba(X)[:, 1]

    def fit_predict_proba(self, X, y=None):
        self.fit(X, y)

        return self.predict_proba(X)

    def score(self, X, y=None, **kwargs):
        y_pred = self.predict_proba(X, y)

        return roc_auc_score(y, y_pred)
