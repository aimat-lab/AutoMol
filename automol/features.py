from automol.datasets import Dataset
import numpy
import pandas as pd
import inspect
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors


class Rdkit:

    @staticmethod
    def produce(df: pd.DataFrame,
                to_exclude: list = None,
                sanitize: bool = True,
                axis: int = 0) -> pd.DataFrame:
        to_exclude = to_exclude or []
        to_exclude.append('setupAUTOCorrDescriptors')
        calc_props = {k: v for k, v in inspect.getmembers(Descriptors, inspect.isfunction)
                      if not k.startswith('_') and k not in to_exclude}

        df = pd.DataFrame(
            [[
                [
                    v(Chem.MolFromSmiles(smi)) for k, v in calc_props.items()
                ]
                for smi in df['smiles']
            ]],
            columns='rdkit')

        if sanitize:
            df.dropna(axis=axis, how='any', inplace=True)

        return df


class FeatureGenerator:
    __featureList = {
        'fingerprint': {
            'iam': {'vector'},
            'requirements': ['smiles'],
            'transform':
                lambda df: pd.DataFrame([[Chem.RDKFingerprint(Chem.MolFromSmiles(smi)) for smi in df['smiles']]],
                                        columns=['fingerprint'])
        },
        'rdkit': {
            'iam': {'vector'},
            'requirements': ['smiles'],
            'transform': lambda df: Rdkit.produce(df)
        },
    }

    def __init__(self, data_set: Dataset):
        self.data_set = data_set
        # feature_name : sub data_frame of features associated with feature_name
        self.__generated_features = {}

    def get_feature(self, feature_name):
        if feature_name not in FeatureGenerator.__featureList:
            raise Exception('unknown feature %s' % feature_name)

        if any(req not in self.data_set for req in FeatureGenerator.__featureList[feature_name]['requirements']):
            raise Exception('requirements for feature %s not satisfied by data set' % feature_name)

        if feature_name not in self.__generated_features:
            self.__generated_features[feature_name] = numpy.array(
                FeatureGenerator.__featureList[feature_name]['transform'](self.data_set.data)
            )

        return self.__generated_features[feature_name]

    def get_acceptable_features(self, acceptable_types):
        """
            returns list of feature_names that can be offered and are acceptable
        :rtype: list[str]
        """
        return [k for k, v in FeatureGenerator.__featureList if acceptable_types & v['iam']]


"""
class Fingerprint:

    @staticmethod
    def requires():
        return ['smiles']

    @staticmethod
    def produce(data):
        return Chem.RDKFingerprint(Chem.MolFromSmiles(data['smiles']))
"""
