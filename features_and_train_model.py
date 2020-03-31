import xgboost as xgb
from sklearn.feature_selection import chi2, mutual_info_classif, RFECV
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from ELI5 import PermutationImportance


def xgb_fit(X_train, y_train):
    clf = xgb.XGBClassifier(max_depth=3,
                            n_estimators=100,
                            learning_rate=0.1,
                            nthread=5,
                            subsample=1.,
                            colsample_bytree=0.5,
                            min_child_weight = 3,
                            reg_alpha=0.,
                            reg_lambda=0.,
                            seed=42,
                            missing=1e10)

    clf.fit(X_train, y_train, eval_metric='aucpr', verbose=10)
    return clf


def prep_importance(importance, features, name):
    fi = pd.DataFrame(list(zip(features, importance))).sort_values(by=1, ascending=False)
    return fi


def features_selection(X_train, y_train, X_test, y_test, sel_type="FI", count=30):    
    if sel_type == "FI":
        fitted_clf = xgb_fit(X_train, y_train, X_test, y_test)
        feature_importance = prep_importance(fitted_clf.feature_importances_, X_train.columns, 'Features Importance')
        feats = feature_importance[0][:count]
    elif sel_type == "HI":
        chi2_test = chi2(X_train, y_train)
        feature_importance = prep_importance(chi2_test[0], X_train.columns, 'Chi2')
        feats = feature_importance[0][:count]
    elif sel_type == "MI":
        mi = mutual_info_classif(X_train, y_train)
        feature_importance = prep_importance(mi, X_train.columns, 'Mutual_Info')
        feats = feature_importance[0][:count]
    elif sel_type == "RFE":
        # By optimal, not by count
        logit = LogisticRegression(random_state=42)
        selector = RFECV(estimator=logit, step=5, cv=StratifiedKFold(2), scoring='f1')
        selector.fit(X_train, y_train)
        feats = X_train.columns[selector.support_]
    elif sel_type == "PI":
        perm = PermutationImportance(fitted_clf, random_state=42).fit(X_train, y_train)
        res = pd.DataFrame(X_train.columns, columns=['feature'])
        res['score'] = perm.feature_importances_
        res['std'] = perm.feature_importances_std_
        feature_importance = res.sort_values(by='score', ascending=False).reset_index(drop=True)
        feats = feature_importance["feature"][0:count]
    else:
        print(f"Feature selection type {sel_type} not implemented.")
        return

    X_train = pd.DataFrame(X_train, columns=X_train.columns)[feats]
    X_test = pd.DataFrame(X_test, columns=X_test.columns)[feats]

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    fitted_clf = xgb_fit(X_train, y_train)
    return fitted_clf