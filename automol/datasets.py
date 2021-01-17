import glob
from abc import ABC
from automol.features import FeatureGenerator

import os
import pandas
from rdkit import Chem


class Dataset(ABC):

    def __init__(self, data):
        self.data = data
        if self.data.empty:
            raise Exception("dataset empty")
        # cached feature generator
        self.__featureGenerator = None

    def feature_generator(self):
        if self.__featureGenerator is None:
            self.__featureGenerator = FeatureGenerator(self)
        return self.__featureGenerator

    def get_feature(self, feature_name):
        """
        wrapper on getting feature generator
        :param feature_name:
        :return:
        """
        return self.feature_generator().get_feature(feature_name)

    def split(self, split_right):
        return Dataset(self.data[0: split_right]), Dataset(self.data[split_right:])

    @classmethod
    def from_spec(cls, spec):
        class_name = spec['dataset_class']
        class_ = globals().get(class_name)
        if class_ is None or not issubclass(type(class_), type) or not issubclass(class_, cls):
            raise Exception("%s is not a valid class name" % class_name)

        data = {}

        if not os.path.isdir(spec['dataset_location']):
            raise Exception("path %s doesn't exist" % spec['dataset_location'])

        amount = spec.get('amount', -1)
        for fn in glob.iglob(spec['dataset_location'] + '/*'):
            if amount == 0:
                break
            with open(fn) as f:
                text = f.read()
                try:
                    data = {**data, **{k: data[k] + [v] if k in data else [v] for k, v in class_.parse(text).items()}}
                    amount -= 1
                except Exception as e:
                    with open("erroneous.txt", "w") as out:
                        out.write(text)
                        out.write(str(e))
                    raise

        data = pandas.DataFrame(data, columns=data.keys())

        return class_(data)

    @classmethod
    def parse(cls, text) -> dict:
        pass

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, key):
        return self.data[key]


class QM9(Dataset):

    @classmethod
    def parse(cls, text):
        r = {}
        lines = str.splitlines(text)
        r['atom_count'] = int(lines[0])
        line2 = lines[1].strip().split()
        r['tag'] = line2.pop(0)
        r['index'], r['A'], r['B'], r['C'], r['mu'], r['alpha'], r['homo'], r['lumo'], r['gap'], r['r2'], r['zpve'], r[
            'U0'], r['U'], r['H'], r['G'], r['Cv'], = [float(a) for a in line2]
        r['smiles'] = Chem.MolToSmiles(Chem.MolFromSmiles(lines[3 + r['atom_count']].split('\t')[0], sanitize=True),
                                       isomericSmiles=False, canonical=True)
        return r
