# # Install PyTDC via pip if needed: pip install PyTDC

# from tdc.single_pred import ADME

# data = ADME(name = 'Lipophilicity_AstraZeneca')
# split = data.get_split(method = 'scaffold')
# train = split['train']
# valid = split['valid']
# test = split['test']

# If not on linux:

import pandas as pd

train = pd.read_csv('data/lipophilicity_train.csv')
valid = pd.read_csv('data/lipophilicity_val.csv')
test = pd.read_csv('data/lipophilicity_test.csv')

