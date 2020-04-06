from Project.modules.dataset_builder import build_dataset_raw
import pandas as pd

"""
Build raw datasets from csv data files
"""

CHURNED_START_DATE = '2019-09-01' 
CHURNED_END_DATE = '2019-10-01'

INTER_1 = (1,7)
INTER_2 = (8,14)
INTER_3 = (15,21)
INTER_4 = (22,28)
INTER_LIST = [INTER_1, INTER_2, INTER_3, INTER_4]

def build_dataset():
    build_dataset_raw(churned_start_date=CHURNED_START_DATE,
                    churned_end_date=CHURNED_END_DATE,
                    inter_list=INTER_LIST,
                    raw_data_path='./Project/data/train/',
                    dataset_path='./Project/data/dataset/', 
                    mode='train')

    build_dataset_raw(churned_start_date=CHURNED_START_DATE,
                    churned_end_date=CHURNED_END_DATE,
                    inter_list=INTER_LIST,
                    raw_data_path='./Project/data/test/',
                    dataset_path='./Project/data/dataset/', 
                    mode='test')

def get_builded_dataset():
    train = pd.read_csv('./Project/data/dataset/dataset_raw_train.csv', sep=';')
    test  = pd.read_csv('./Project/data/dataset/dataset_raw_test.csv',  sep=';')
    return train, test
