from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss

def evaluation(y_true, y_pred, y_prob):
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

def validate(clf, X_valid, y_valid):
    predict_proba_test = clf.predict_proba(X_valid)
    predict_test = clf.predict(X_valid)
    precision_test, recall_test, f1_test, log_loss_test, roc_auc_test = \
        evaluation(y_valid, predict_test, predict_proba_test[:, 1])