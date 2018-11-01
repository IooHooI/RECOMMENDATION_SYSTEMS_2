from sklearn.base import BaseEstimator, ClassifierMixin
from tffm import TFFMClassifier
import tensorflow as tf


class FactorizationMachineBasedRecommender(BaseEstimator, ClassifierMixin):
    def __init__(self):
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
        self.model.fit(X, y, show_progress=True)
        pass

    def predict(self, X, y=None):

        pass

    def score(self, X, y=None, **kwargs):

        pass
