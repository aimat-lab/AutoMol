from abc import ABC


from .datasets import *


import numpy
from rdkit import Chem


class FeatureGenerator:

    def __init__(self, data_set:Dataset):
        self.data_set = data_set
        self.__generated_features = {}

    def get_feature(self, feature_name):
        1


class Fingerprint:

    @staticmethod
    def requires():
        return ['smiles']

    @staticmethod
    def produce(data):
        return Chem.RDKFingerprint(Chem.MolFromSmiles(data['smiles']))
