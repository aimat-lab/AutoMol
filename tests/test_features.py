from unittest import TestCase
import pandas as pd
from automol.features import RdkitDescriptors

records = [["C",
            0,
            0],
           ["CSC1=C2C(C=C(C=CC=C3)C3=C2)=C(SC)C4=CC5=CC=CC=C5C=C41",
            -5.0700745250712,
            -3.2600479263974],
           ["CCC1=CC=CC(CC)=C1C2=C3C(C=C(C=CC=C4)C4=C3)=C(C5=C(CC)C=CC=C5CC)C6=CC7=CC=CC=C7C=C62",
            -5.0300739303396,
            -2.9900439527762],
           ["C[Si](C)(C)C#CC1=C(C=C(C=CC=C2)C2=C3)C3=C(C#C[Si](C)(C)C)C4=CC5=CC=CC=C5C=C41",
            -5.1100751198028,
            -3.4200502781124],
           ["CC1=CC=CC(C)=C1C2=C3C(C=C(C=CC=C4)C4=C3)=C(C5=C(C)C=CC=C5C)C6=CC7=CC=CC=C7C=C62",
            -5.0100736601852,
            -3.0000440878534]]
columns = ['smiles', 'HOMO', 'LUMO']
data = pd.DataFrame.from_records(data=records, columns=columns)


class TestRdkitDescriptors(TestCase):
    def test_produce(self):
        f = RdkitDescriptors()
        features = f.produce(data)
        features = f.produce(data, to_calculate=["MolLogP", "qed"])
        print(features)
