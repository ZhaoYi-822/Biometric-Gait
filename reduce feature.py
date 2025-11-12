import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, RFECV
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def avova(data):
    X=data.iloc[:, :17]
    y=data.iloc[:, 17]
    F_values, p_values = f_classif(X, y)
    indices = np.argsort(F_values)[::-1]
    sorted_F_values = F_values[indices]
    sorted_features = indices

    # indices = np.argsort(p_values)[::-1]
    # sorted_p_values = p_values[indices]
    # sorted_features = indices

    for i in range(len(sorted_features)):
        print('P_values:', p_values)



def heatmap(data):

    X=data.iloc[:, :17]
    corr_matrix = np.corrcoef(X, rowvar=False)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', xticklabels=True, yticklabels=True)
    plt.title("Correlation Heatmap")
    plt.show()


def  feature_imp(data):

    X = data.iloc[:, :17]
    y = data.iloc[:, 17]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_

  
    indices = np.argsort(importances)[::-1] 
    sorted_importances = importances[indices]  
    sorted_features = indices 
    for i in range(len(sorted_features)):
        print(f"Feature {sorted_features[i]}: {sorted_importances[i]}")










def RFE(data):
    X = data.iloc[:, :17]
    y = data.iloc[:, 17]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=500)
    selector = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring='accuracy')
    selector.fit(X_scaled , y)

    print(f"最优特征数量: {selector.n_features_}")
    print(f"选择的特征: {selector.support_}")
    plt.figure(figsize=(10, 6))
    plt.title("RFECV - Optimal Number of Features")
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Cross Validation Score")
    plt.plot(range(1, X.shape[1] + 1), selector.grid_scores_, marker='o')
    plt.show()




if __name__=='__main__':
    data = pd.read_csv('new_gait_dataset/original_gait_dataset.csv')
    # avova(data)
    # heatmap(data)
    feature_imp(data)

    # RFE(data)
