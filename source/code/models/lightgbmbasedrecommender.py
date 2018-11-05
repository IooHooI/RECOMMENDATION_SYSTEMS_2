from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import multiprocessing


class LightGBMBasedRecommender(BaseEstimator, ClassifierMixin):
    def __init__(self, val_size=0.3):
        self.val_size = val_size
        self.model = LGBMClassifier(n_jobs=multiprocessing.cpu_count())

    def fit(self, X, y=None):
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.val_size,
            random_state=42,
            stratify=y
        )

        self.model.fit(
            X=X_train,
            y=y_train,
            eval_metric='auc',
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=5,
            verbose=True
        )

        return self

    def predict(self, X, y=None):
        return self.model.predict_proba(X)[:, 1]

    def fit_predict(self, X, y=None):
        self.fit(X, y)

        return self.predict(X)

    def score(self, X, y=None, **kwargs):
        y_pred = self.predict(X, y)

        return roc_auc_score(y, y_pred)
