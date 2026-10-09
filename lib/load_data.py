# LOAD THE TRAIN AND TEST SPLIT FROM REMOTE AND STORE THEM

from datasets import load_dataset
datacard_name = "LingoIITGN/COMI-LINGUA"
datacard_configs = ['LID', 'MLI', 'POS', 'NER', 'TN', 'MT']
load_dataset(datacard_name, datacard_configs[0], split='train').to_pandas().to_csv('comi_lingua_dataset/train.csv', index=False)
load_dataset(datacard_name, datacard_configs[0], split='test').to_pandas().to_csv('comi_lingua_dataset/test.csv', index=False)