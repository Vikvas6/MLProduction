from sklearn.model_selection import train_test_split
from Project.utils import xgb_fit_predict, features_selection
import pandas as pd

def fit_predict(X, y, X_test, columns, sel_type="FI", count=30):
    """
    Fit model on param:X and param:y with printing validation information and save prediction of param:X_test to the csv file.
    """
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42,)
    X_train, y_train, X_valid, y_valid, selected_features = features_selection(X_train, y_train, X_valid, y_valid, columns, sel_type, count)
    fitted_clf, predict_test, predict_proba_test = xgb_fit_predict(X_train, y_train, X_valid, y_valid)

    prediction_data = fitted_clf.predict(X_test[selected_features])
    prediction_data_proba = fitted_clf.predict_proba(X_test[selected_features])
    prediction = pd.DataFrame({'user_id': X_test['user_id'], 'is_churned': prediction_data})
    prediction.to_csv('./Project/data/prediction.csv', index=False)

    print('Model was fitted and prediction was saved to data/prediction.csv file.')
    return fitted_clf, prediction_data, prediction_data_proba