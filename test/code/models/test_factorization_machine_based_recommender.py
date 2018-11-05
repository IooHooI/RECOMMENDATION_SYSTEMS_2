import os
import unittest
import numpy as np
import pandas as pd
import source.code.models.factorizationmachinebasedrecommender as fmbr

data_directory = '../../data/datasets/'

train = pd.read_csv(os.path.join(data_directory, 'train.csv'), engine='python')

songs = pd.read_csv(os.path.join(data_directory, 'songs.csv'))

members = pd.read_csv(os.path.join(data_directory, 'members.csv'))


class TestPipeline(unittest.TestCase):

    def test_case_1(self):
        recommender = fmbr.FactorizationMachineBasedRecommender()