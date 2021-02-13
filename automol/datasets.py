import glob
from abc import ABC
from automol.features.features import Features

import os
import pandas
from rdkit import Chem

import pysftp
import paramiko


class Dataset(ABC):

    def __init__(self, data):
        self.data = data
        if self.data.empty:
            raise Exception("dataset empty")
        # cached feature generator
        self.__features = None

    def features(self):
        if self.__features is None:
            self.__features = Features(self.data)
        return self.__features

    def get_feature(self, feature_name):
        """
        wrapper on getting feature from generator
        :param feature_name:
        :return:
        """
        return self.features().get_feature(feature_name)

    def get_acceptable_features(self, acceptable_types):
        return self.features().get_acceptable_feature_gens(acceptable_types)

    @classmethod
    def from_spec(cls, spec):
        class_name = spec['dataset_class']
        class_ = globals().get(class_name)
        if class_ is None or not issubclass(type(class_), type) or not issubclass(class_, cls):
            raise Exception("%s is not a valid class name" % class_name)

        data = {}
        amount = spec.get('amount', -1)

        def parse_and_catch(text):
            nonlocal amount
            try:
                data.update((k, data[k] + [v] if k in data else [v]) for k, v in class_.parse(text).items())
                amount -= 1
            except Exception as e:
                with open("erroneous.txt", "w") as out:
                    out.write(text)
                    out.write('\n' + str(e))
                raise

        # lsdf dataset
        if spec['dataset_location'].startswith('lsdf://'):
            import io
            config = paramiko.config.SSHConfig()
            config.parse(open(os.path.expanduser('~/.ssh/config')))
            conf = config.lookup('lsdf')
            lsdf_dataset_path = spec['dataset_location'][len('lsdf://'):]
            with pysftp.Connection(host=conf['hostname'],
                                   username=conf['user'], private_key=conf['identityfile'][0]) as sftp:
                with sftp.cd(lsdf_dataset_path):
                    for file_attr in sftp.sftp_client.listdir_iter():
                        if amount == 0:
                            break
                        file = file_attr.filename
                        f = io.BytesIO()
                        sftp.getfo(file, f)
                        f.seek(0)
                        text = io.TextIOWrapper(f, encoding='utf-8').read()
                        parse_and_catch(text)
        # local dataset
        else:
            if not os.path.isdir(spec['dataset_location']):
                raise Exception("path %s doesn't exist" % spec['dataset_location'])

            for fn in glob.iglob(spec['dataset_location'] + '/*'):
                if amount == 0:
                    break
                with open(fn) as f:
                    text = f.read()
                parse_and_catch(text)

        data = pandas.DataFrame(data, columns=data.keys())

        return class_(data)

    @classmethod
    def parse(cls, text) -> dict:
        pass

    @classmethod
    def get_indices(cls) -> list:
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
        r['index'] = int(r['index'])
        return r

    @classmethod
    def get_indices(cls) -> list:
        return ["{:06d}".format(index) for index in cls.data['index']]


class DataSplit:

    @staticmethod
    def invoke(data: pandas.DataFrame, method: str, param):
        if hasattr(DataSplit, method):
            return getattr(DataSplit, method)(data, param)
        raise TypeError('method %s is illegal' % method)

    @staticmethod
    def k_fold(data, k):
        """
        k fold split iterator that doesn't copy data
        doesn't support mutability of datasets
        :param data
        :param k: k-fold
        :return: generator for tuples (valid 1/k, train (k-1)/k)
        """
        data = data.sample(frac=1)
        inc = len(data.index) / k
        return ((next_valid, data.drop(next_valid.index)) for next_valid in
                (data.index[round(i * inc): round((i+1) * inc)] for i in range(k)))

    @staticmethod
    def split(data, split_sep):
        return (data[0: split_sep], data[split_sep:]),
