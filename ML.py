from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



train_data=pd.read_csv('new_gait_dataset/only_sample_gait_dataset.csv')
test_data=pd.read_csv('new_gait_dataset/original_test_gait_dataset.csv')


x_train=train_data.iloc[:,:17]
y_train=train_data.iloc[:,17]

x_test=test_data.iloc[:,:17]
y_test=test_data.iloc[:,17]

#

Stand_X = StandardScaler() 
x_train = Stand_X.fit_transform(x_train)
x_test= Stand_X.fit_transform(x_test)





def knn():
    knn = KNeighborsClassifier(n_neighbors=5,weights='distance', metric='manhattan')

    knn.fit(x_train, y_train)  

    y_pred = knn.predict(x_test)

    print(knn.predict(x_test))  
    accuracy = accuracy_score(y_test,y_pred)
    print(accuracy)
    joblib.dump(knn, 'model/only_knn.pkl')

def svm():

    clf = SVC( C=15, kernel='rbf', decision_function_shape='ovr',probability=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(accuracy)
    joblib.dump(clf, 'model/only_svm.pkl')

def rf():
    forest = RandomForestClassifier(n_estimators=200, random_state=42)
    forest.fit(x_train, y_train)
    y_pred = forest.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    joblib.dump(forest, 'model/only_rf.pkl')




if __name__ == '__main__':
    svm()
    knn()
    rf()


