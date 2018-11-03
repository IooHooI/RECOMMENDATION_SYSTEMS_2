import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MetaFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, user_meta=None, item_meta=None):
        self.user_meta = user_meta
        self.item_meta = item_meta
        self.X_with_meta = None

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        self.X_with_meta = X.copy()
        self.X_with_meta = pd.merge(self.X_with_meta, self.user_meta, on='msno', how='left')
        self.X_with_meta = pd.merge(self.X_with_meta, self.item_meta, on='song_id', how='left')
        self.X_with_meta[
            'days_registered'
        ] = self.X_with_meta.expiration_date - self.X_with_meta.registration_init_time
        self.X_with_meta['days_registered'] = self.X_with_meta.days_registered.apply(lambda x: x.days)
        return self.X_with_meta
