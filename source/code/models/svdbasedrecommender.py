from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix
from tqdm.autonotebook import tqdm
import numpy as np


class SVDBasedRecommender(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.users_representations = None
        self.song_representations = None
        self.unique_users = None
        self.unique_songs = None
        self.user2index = None
        self.song2index = None
        self.index2user = None
        self.index2song = None

    def fit(self, X, y=None, **kwargs):

        self.unique_users = list(set(X.msno))

        self.unique_songs = list(set(X.song_id))

        self.user2index = dict(zip(self.unique_users, range(len(self.unique_users))))

        self.song2index = dict(zip(self.unique_songs, range(len(self.unique_songs))))

        self.index2user = dict(zip(range(len(self.unique_users)), self.unique_users))

        self.index2song = dict(zip(range(len(self.unique_songs)), self.unique_songs))

        user_indexes = X.msno.map(self.user2index).values

        song_indexes = X.song_id.map(self.song2index).values

        flags = y.values

        sparse_matrix = csr_matrix(
            arg1=(flags, (user_indexes, song_indexes)),
            shape=(len(self.unique_users), len(self.unique_songs))
        )

        svd = TruncatedSVD(n_components=10, n_iter=20, random_state=42)

        svd.fit(sparse_matrix)

        self.users_representations = svd.transform(sparse_matrix)

        self.song_representations = svd.components_

        return self

    def predict(self, X, y=None, **kwargs):
        y_pred = []

        for user, song in tqdm(zip(X.msno.values, X.song_id.values)):
            if user in self.user2index and song in self.song2index:
                y_pred.append(
                    np.dot(
                        self.users_representations[self.user2index[user], :],
                        self.song_representations[:, self.song2index[song]]
                    )
                )
            else:
                y_pred.append(0.5)

        y_pred_std = (y_pred - min(y_pred)) / (max(y_pred) - min(y_pred))

        y_pred_std = y_pred_std * (1 - 0) + 0

        return y_pred_std

    def score(self, X, y=None, **kwargs):
        y_pred = self.predict(X, y)

        return roc_auc_score(y, y_pred)
