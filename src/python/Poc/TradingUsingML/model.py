# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:32:23 2018

@author: Aditya
"""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt



df_tcs = pd.read_excel('data/TCS.NS.xlsx', index_col = 0 )

print (df_tcs.head())

df_tcs_after_cleanup = df_tcs.dropna()


sns.heatmap (df_tcs_after_cleanup.corr() > 0.6)

plt.show()



df_tcs_after_removing_columns = df_tcs_after_cleanup.drop (['Open','High','Low', 'Close', 'Adj Close', 'NIFTY',
                                                   'returns',
                                                   '10DMA', '20DMA','21DMA', 'Avg_gain', 'SMBL' , 'SMOL'  
                                                   ,  'Avg_loss'  ,'gain','loss'  , 'UL','LL',   'STD', 'GL', 'RS'], axis = 1)



print (df_tcs_after_removing_columns.head())

print (df_tcs_after_removing_columns.info())

print (df_tcs_after_removing_columns.describe())




sns.heatmap (df_tcs_after_removing_columns.corr() > 0.6)

df_tcs_after_removing_columns.hist(figsize=(10,8))


plt.show()




boxplot = sns.boxplot (data = df_tcs_after_removing_columns)


loc, labels = plt.xticks()
boxplot.set_xticklabels(labels, rotation=45)

plt.show ()



print (df_tcs_after_removing_columns.columns)

y = df_tcs_after_removing_columns ['BuySell'].values
                
X = df_tcs_after_removing_columns.drop('BuySell' , axis =1)

columns = X.columns

X = X.values

print (X)






from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() 
X = sc_X.fit_transform(X)


from sklearn.decomposition import PCA 

pca_model = PCA()

pca_model.fit (X)

transformed = pca_model.transform (X)

features = range (pca_model.n_components_)



plt.bar(features, pca_model.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)

plt.show()

import numpy as np

a = np.array(pca_model.explained_variance_ratio_ )

print (a)







from sklearn.model_selection import train_test_split

X_train, X_test, y_train , y_test = train_test_split (X, y , test_size = 0.3 , random_state = 42)




# Import suite of algorithms.
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
#from sklearn.neural_network import MLPClassifier


# Create objects of required models.
models = []
models.append(("LR",LogisticRegression()))
models.append(("GNB",GaussianNB()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("DecisionTree",DecisionTreeClassifier()))
models.append(("LDA",  LinearDiscriminantAnalysis()))
models.append(("QDA",  QuadraticDiscriminantAnalysis()))
models.append(("AdaBoost", AdaBoostClassifier()))
models.append(("SVM Linear",SVC(kernel="linear")))
models.append(("SVM RBF",SVC(kernel="rbf")))
models.append(("Random Forest",  RandomForestClassifier()))
models.append(("Bagging",BaggingClassifier()))
models.append(("Calibrated classifier",CalibratedClassifierCV()))
models.append(("GradientBoosting",GradientBoostingClassifier()))
models.append(("LinearSVC",LinearSVC()))
models.append(("Ridge",RidgeClassifier()))
#models.append(("MLP",MLPClassifier()))




# Find accuracy of models.
results = []
for name,model in models:
    kfold = KFold(n_splits=10, random_state=0)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    results.append(tuple([name,cv_result.mean(), cv_result.std()]))
  
results.sort(key=lambda x: x[1], reverse = True)    
for i in range(len(results)):
    print('{:20s} {:2.2f} (+/-) {:2.2f} '.format(results[i][0] , results[i][1] * 100, results[i][2] * 100))
    


# Import xgboost
import xgboost as xgb

# Create arrays for the features and the target: X, y
#X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the training and test sets
#


#X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=.20, random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier()

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
y_pred = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(y_pred==y_test))/y_test.shape[0]

print("accuracy: %f" % (accuracy))


from sklearn.metrics import confusion_matrix, accuracy_score
cf = confusion_matrix(y_test, y_pred)

print(cf)
print(accuracy_score(y_test, y_pred) * 100) 



feat_imp = pd.Series(xg_cl.feature_importances_, columns).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()