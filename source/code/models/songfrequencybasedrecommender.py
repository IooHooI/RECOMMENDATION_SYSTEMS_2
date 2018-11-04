from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
import pandas as pd


class SongFrequencyBasedRecommender(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.unique_songs_count = None
        self.songs_popularity = None

    def fit(self, X, y=None, **kwargs):
        self.unique_songs_count = y.sum()

        self.songs_popularity = pd.concat(
            [
                X,
                y.rename('prob')
            ],
            axis=1
        ).groupby('song_id')['prob'].apply(lambda x: sum(x) / self.unique_songs_count).to_dict()

        return self

    def predict(self, X, y=None, **kwargs):
        y_pred = [self.songs_popularity.get(song_id, 0.5) for song_id in X.song_id]

        return y_pred

    def score(self, X, y=None, **kwargs):
        y_pred = self.predict(X, y)

        return roc_auc_score(y, y_pred)
