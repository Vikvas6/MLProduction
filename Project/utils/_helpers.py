from datetime import timedelta
import time
from sklearn.metrics import auc, average_precision_score, confusion_matrix, f1_score, log_loss, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
import xgboost as xgb
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFECV, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from ELI5 import PermutationImportance
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interp

def time_format(sec):
    """
    Return time in seconds as user-friendly time format
    """
    return str(timedelta(seconds=sec))

def etl_loop(func):
    """
    Decorator for etl operations repeats operation 100 times in case of errors
    """
    def _func(*args, **kwargs):
        _max_iter_cnt = 100
        for i in range(_max_iter_cnt):
            try:
                start_t = time.time()
                res = func(*args, **kwargs)
                run_time = time_format(time.time() - start_t)
                print('Run time "{}": {}'.format(func.__name__, run_time))
                return res
            except Exception as er:
                run_time = time_format(time.time() - start_t)
                print('Run time "{}": {}'.format(func.__name__, run_time))
                print('-'*50)
                print(er, '''Try № {}'''.format(i + 1))
                print('-'*50)
        raise Exception('Max error limit exceeded: {}'.format(_max_iter_cnt))
    return _func

def evaluation(y_true, y_pred, y_prob):
    """
    Evaluate model and print precision, recall, f1-score, log loss and ROC AUC metrics.
    """
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    ll = log_loss(y_true=y_true, y_pred=y_prob)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_prob)
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1: {}'.format(f1))
    print('Log Loss: {}'.format(ll)) 
    print('ROC AUC: {}'.format(roc_auc)) 
    return precision, recall, f1, ll, roc_auc

def xgb_fit_predict(X_train, y_train, X_test, y_test):
    """
    Fit xgboost classifier and validate it on test data
    """
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
    predict_proba_test = clf.predict_proba(X_test)
    predict_test = clf.predict(X_test)
    precision_test, recall_test, f1_test, log_loss_test, roc_auc_test = \
        evaluation(y_test, predict_test, predict_proba_test[:, 1])
    return clf, predict_test, predict_proba_test

def plot_importance(importance, features, name):
    """
    Plot graph of feature importance
    """
    fi = pd.DataFrame(list(zip(features, importance))).sort_values(by=1, ascending=False)
    plt.figure(figsize=(16,6))
    plt.bar(range(fi.shape[0]), fi[1], align='center')
    plt.xticks(range(fi.shape[0]), fi[0], rotation=90)
    plt.title(name)
    plt.show()
    return fi


def features_selection(X_train, y_train, X_test, y_test, columns, sel_type="FI", count=30):
    """
    Select features. Following types (param:sel_type) are available:
      - FI - feature importance for choosen classifier
      - HI - feature selection based of Chi-square test
      - MI - feature selection based of mutual info value
      - RFE - performing Recursive Feature Elimination
      - PI - calculating Permutation Importance
      - None - don't perform feature selection
    """
    if sel_type == "FI":
        fitted_clf, _, _ = xgb_fit_predict(X_train, y_train, X_test, y_test)
        feature_importance = plot_importance(fitted_clf.feature_importances_, columns, 'Features Importance')
        feats = feature_importance[0][:count]
    elif sel_type == "HI":
        chi2_test = chi2(X_train, y_train)
        feature_importance = plot_importance(chi2_test[0], columns, 'Chi2')
        feats = feature_importance[0][:count]
    elif sel_type == "MI":
        mi = mutual_info_classif(X_train, y_train)
        feature_importance = plot_importance(mi, columns, 'Mutual_Info')
        feats = feature_importance[0][:count]
    elif sel_type == "RFE":
        # By optimal, not by count
        logit = LogisticRegression(random_state=42)
        selector = RFECV(estimator=logit, step=5, cv=StratifiedKFold(2), scoring='f1')
        selector.fit(X_train, y_train)
        feats = columns[selector.support_]
    elif sel_type == "PI":
        perm = PermutationImportance(fitted_clf, random_state=42).fit(X_train, y_train)
        res = pd.DataFrame(columns, columns=['feature'])
        res['score'] = perm.feature_importances_
        res['std'] = perm.feature_importances_std_
        feature_importance = res.sort_values(by='score', ascending=False).reset_index(drop=True)
        feats = feature_importance["feature"][0:count]
    elif sel_type == "None":
        feats = columns
    else:
        print(f"Feature selection type {sel_type} not implemented.")
        return

    X_train = pd.DataFrame(X_train, columns=columns)[feats]
    X_test = pd.DataFrame(X_test, columns=columns)[feats]

    return X_train, y_train, X_test, y_test, feats

def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues):
    """
    Plot confusion matrix
    """
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    cm = np.array([[TP, FP],
                   [FN, TN]])
    cm_normalized = cm.astype('float') / cm.sum(axis=0)
    # Plot both matrixes - basic and normalized
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    for ax, normalize, data, title in zip(ax,
                                          [False, True], 
                                          [cm, cm_normalized], 
                                          ['Confusion matrix (without normalization)', 
                                           'Сonfusion matrix (normalized)']):
        im = ax.imshow(data, interpolation='nearest', cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(im, cax=cax)
        ax.set(xticks=np.arange(data.shape[1]),
               yticks=np.arange(data.shape[0]),
               xticklabels=classes, 
               yticklabels=classes,
               title=title,
               ylabel='Predicted label',
               xlabel='True label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')        
        # Loop over data dimensions and create text annotations
        fmt = '.2f' if normalize else 'd'
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", 
                        color="white" if data[i, j] > data.max() / 2. else "black")                
    fig.tight_layout()
    return fig 

def plot_PR_curve(y_true, y_pred, y_prob):
    """
    Plot precision curve
    """
    AP = average_precision_score(y_true=y_true, y_score=y_prob)
    precisions, recalls, thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_prob)
    
    plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
    plt.fill_between(recalls, precisions, alpha=0.2, color='b', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('PR curve: AP={0:0.2f}'.format(AP))
    
def plot_ROC_curve(classifier, X, y, n_folds):
    """
    Plot Receiver Operating Characteristic (ROC) curve
    """
    cv = StratifiedKFold(n_splits=n_folds)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % \
             (mean_auc, std_auc), lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
