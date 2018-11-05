from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from tqdm import tqdm
import multiprocessing
import pandas as pd


class EnsembleBasedRecommender(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimators, ens_train_size=0.4):
        self.ens_train_size = ens_train_size

        self.base_estimators = base_estimators

        self.ens_estimator = LGBMClassifier(n_jobs=multiprocessing.cpu_count())

    def fit(self, X, y=None):
        X_base, X_ens, y_base, y_ens = train_test_split(
            X,
            y,
            test_size=self.ens_train_size,
            random_state=42,
            stratify=y
        )

        ens_train_data = {}

        for i, base_estimator in enumerate(tqdm(self.base_estimators)):
            base_estimator.fit(X=X_base, y=y_base)

            ens_train_data['est_{}'.format(i)] = base_estimator.predict_proba(X_ens)

        self.ens_estimator.fit(X=pd.DataFrame(ens_train_data).values, y=y_ens)

        return self

    def predict_proba(self, X, y=None):
        ens_train_data = {}

        for i, base_estimator in enumerate(tqdm(self.base_estimators)):
            ens_train_data['est_{}'.format(i)] = base_estimator.predict_proba(X)

        return self.ens_estimator.predict_proba(pd.DataFrame(ens_train_data).values)[:, 1]

    def fit_predict_proba(self, X, y=None):
        self.fit(X, y)

        return self.predict_proba(X)

    def score(self, X, y=None, **kwargs):
        y_pred = self.predict_proba(X, y)

        return roc_auc_score(y, y_pred)
