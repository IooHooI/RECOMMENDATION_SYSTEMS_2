import os
import unittest
import numpy as np
import pandas as pd
import source.code.models.factorizationmachinebasedrecommender as fmbr

data_directory = '../../data/datasets/'

train = pd.read_csv(os.path.join(data_directory, 'train.csv'), engine='python')

songs = pd.read_csv(os.path.join(data_directory, 'songs.csv'))

songs.lyricist.fillna('unknown', inplace=True)

songs.composer.fillna('unknown', inplace=True)

songs.genre_ids.fillna('unknown', inplace=True)

songs.language.fillna(-1, inplace=True)

songs.language = songs.language.astype(np.int64)

members = pd.read_csv(os.path.join(data_directory, 'members.csv'))

members.fillna('unknown', inplace=True)

members.registration_init_time = pd.to_datetime(members.registration_init_time, format='%Y%m%d')

members.expiration_date = pd.to_datetime(members.expiration_date, format='%Y%m%d')


class TestPipeline(unittest.TestCase):

    def test_case_1(self):

        recommender = fmbr.FactorizationMachineBasedRecommender()
