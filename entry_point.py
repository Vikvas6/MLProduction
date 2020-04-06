from Project.modules.train import fit_predict
from Project.modules.preprocessing import prepare_dataset_balanced, prepare_dataset
from Project.modules.run_build import get_builded_dataset, INTER_LIST
from Project.utils import evaluation
import pandas as pd

def process_model():
    train, test  = get_builded_dataset()
    columns = train.drop(['user_id', 'is_churned'], axis=1).columns

    X_train, y_train = prepare_dataset_balanced(train, inter_list=INTER_LIST)
    X_test = prepare_dataset(test, dataset_type='test', inter_list=INTER_LIST)
    #clf, prediction, prediction_proba = fit_predict(X_train, y_train, X_test, columns, sel_type="FI", count=30)
    clf, prediction, prediction_proba = fit_predict(X_train, y_train, X_test, columns, sel_type="None")

if __name__ == "__main__":
    process_model()