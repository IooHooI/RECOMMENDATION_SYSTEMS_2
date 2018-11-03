import os
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from tffm import TFFMClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from source.code.transformers.metafeaturesextractor import MetaFeaturesExtractor

data_directory = '../../../data/datasets/'

train = pd.read_csv(os.path.join(data_directory, 'train.csv'), engine='python')

songs = pd.read_csv(os.path.join(data_directory, 'songs.csv'))

members = pd.read_csv(os.path.join(data_directory, 'members.csv'))

members.registration_init_time = pd.to_datetime(members.registration_init_time, format='%Y%m%d')

members.expiration_date = pd.to_datetime(members.expiration_date, format='%Y%m%d')

X, y = train[train.columns[:-1]], train[train.columns[-1]]


class TestPipeline(unittest.TestCase):

    def test_case_1(self):
        categorical_features = [
            'source_system_tab',
            'source_screen_name',
            'city',
            'gender'
        ]

        categorical_features_lang = [
            'language'
        ]

        numerical_features = [
            'bd',
            'song_length',
            'days_registered'
        ]

        num_features_pipeline = Pipeline([
            ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('discretize', KBinsDiscretizer(n_bins=4, encode='onehot-dense'))
        ])

        cat_features_pipeline = Pipeline([
            ('impute', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        cat_features_pipeline_lang = Pipeline([
            ('impute', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', num_features_pipeline, numerical_features),
            ('cat', cat_features_pipeline, categorical_features),
            ('cat_lang', cat_features_pipeline_lang, categorical_features_lang)
        ])

        unified_pipeline = Pipeline(steps=[
            ('add_meta_info', MetaFeaturesExtractor(user_meta=members, item_meta=songs)),
            ('preprocessing', preprocessor)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=42, stratify=y)

        X_train = unified_pipeline.fit_transform(X_train, y_train)

        self.assertTrue(len(X_train) > 0)

        model = TFFMClassifier(
            order=6,
            rank=10,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
            n_epochs=100,
            batch_size=-1,
            init_std=0.001,
            input_type='dense'
        )

        model.fit(X_train, y_train.values, show_progress=True)
