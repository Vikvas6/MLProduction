import time
from Project.utils import time_format
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def prepare_dataset(dataset, 
                    dataset_type='train',
                    dataset_path='./Project/data/dataset/',
                    inter_list=[(1,7),(8,14)]):
    """
    Prepare databaset - check values
    """
    print(dataset_type)
    start_t = time.time()
    print('Dealing with missing values, outliers, categorical features...')
    
    # Профили
    dataset['age'] = dataset['age'].fillna(dataset['age'].median())
    dataset['gender'] = dataset['gender'].fillna(dataset['gender'].mode()[0])
    dataset.loc[~dataset['gender'].isin(['M', 'F']), 'gender'] = dataset['gender'].mode()[0]
    dataset['gender'] = dataset['gender'].map({'M': 1., 'F':0.})
    dataset.loc[(dataset['age'] > 80) | (dataset['age'] < 7), 'age'] = round(dataset['age'].median())
    dataset.loc[dataset['days_between_fl_df'] < -1, 'days_between_fl_df'] = -1
    # Пинги
    for period in range(1,len(inter_list)+1):
        col = 'avg_min_ping_{}'.format(period)
        dataset.loc[(dataset[col] < 0) | 
                    (dataset[col].isnull()), col] = dataset.loc[dataset[col] >= 0][col].median()
    # Сессии и прочее
    dataset.fillna(0, inplace=True)
    dataset.to_csv('{}dataset_{}.csv'.format(dataset_path, dataset_type), sep=';', index=False)
         
    print('Dataset is successfully prepared and saved to {}, run time (dealing with bad values): {}'.\
          format(dataset_path, time_format(time.time()-start_t)))
    return dataset

def prepare_dataset_balanced(dataset, 
                    dataset_type='train',
                    dataset_path='./Project/data/dataset/',
                    inter_list=[(1,7),(8,14)]):
    """
    Prepare dataset (see f. prepare_dataset()) and balance it
    """
    dataset = prepare_dataset(dataset, dataset_type, dataset_path)
    X_train = dataset.drop(['user_id', 'is_churned'], axis=1)
    y_train = dataset['is_churned']

    X_train_mm = MinMaxScaler().fit_transform(X_train)
    X_train_balanced, y_train_balanced = SMOTE(random_state=42, ratio=0.3).fit_sample(X_train_mm, y_train.values)
    return X_train_balanced, y_train_balanced