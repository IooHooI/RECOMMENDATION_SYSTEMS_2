from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from tffm import TFFMClassifier
import tensorflow as tf


class FactorizationMachineBasedRecommender(BaseEstimator, ClassifierMixin):
    def __init__(self, show_progress=False):
        self.show_progress = show_progress
        self.model = TFFMClassifier(
            order=6,
            rank=10,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
            n_epochs=100,
            batch_size=-1,
            init_std=0.001,
            input_type='dense'
        )

    def fit(self, X, y=None):
        self.model.fit(X, y, show_progress=self.show_progress)

        return self

    def predict_proba(self, X, y=None):
        return self.model.predict_proba(X)[:, 1]

    def fit_predict_proba(self, X, y=None):
        self.fit(X, y)

        return self.predict_proba(X)

    def score(self, X, y=None, **kwargs):
        y_pred = self.predict_proba(X, y)

        return roc_auc_score(y, y_pred)
