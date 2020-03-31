from prepare_dataset import prepare_dataset
from features_and_train_model import features_selection, train_model
from validate_model import validate

def process_model():
    X_train, y_train, X_test, y_test = prepare_dataset()
    X_train, y_train, X_test, y_test = features_selection(X_train, y_train, X_test, y_test, "HI", 25)
    clf = train_model(X_train, y_train)
    validate(clf, X_test, y_test)

if __name__ == "__main__":
    process_model()