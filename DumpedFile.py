import pandas as pd
import numpy as np
import pickle
dataset = pd.read_csv('cleaned_data.csv')
features = dataset.columns.tolist()
items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university']
features = [items for items in features if items not in items_to_remove]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset[features[:-1]].values,dataset['default payment next month'].values,
                                                    test_size=0.2, stratify=dataset['default payment next month'].values, random_state=24)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, max_depth = 8, criterion = 'gini')
classifier.fit(X_train, y_train)
pickle.dump(classifier,open('model.pkl','wb'))