import pandas as pd

def load(path="dataset/dataset_train.csv", sep=";"):
    return pd.read_csv(path, sep=sep)