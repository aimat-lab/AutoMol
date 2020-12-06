import glob
from abc import ABC

import pandas
import yaml
from rdkit import Chem


def add(r, k, v):
    if k not in r:
        r[k] = []
    r[k].append(v)


def merge(a, b):
    for k in b:
        add(a, k, b[k])


class Dataset(ABC):

    def __init__(self, spec):

        data = {}
        self.models = spec['models']
        self.labels = [label.lower() for label in spec['labels']]

        self.index = 0

        amount = spec.get('amount', -1)
        for fn in glob.iglob(spec['dataset_location'] + '/*'):
            if amount == 0:
                break
            with open(fn) as f:
                text = f.read()
                try:
                    merge(data, self.parse(text))
                    amount -= 1
                except Exception as e:
                    with open("erroneous.txt", "w") as out:
                        out.write(text)
                        out.write(str(e))
                    raise

        self.data = pandas.DataFrame(data, columns=data.keys())

    @classmethod
    def from_input(cls, input_file):
        with open(input_file, 'r') as file:
            try:
                spec = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise e

        class_name = spec['dataset_class']
        class_ = globals().get(class_name)
        if class_ is None or not issubclass(type(class_), type) or not issubclass(class_, cls):
            raise Exception("%s is not a valid class name" % class_name)

        return class_(spec)

    def parse(self, text) -> dict:
        pass

    def get_batch(self, amount):
        self.index += amount
        return self.data[self.index - amount: self.index]

    def __iter__(self):
        return self.data.__iter__()


class QM9(Dataset):

    def parse(self, text):
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
