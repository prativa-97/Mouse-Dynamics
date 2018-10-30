import numpy as np
import pandas as pd
import extractor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", DeprecationWarning)

# data extraction
X_nik = extractor.features(13,"data/nikhil/")
X_nik= pd.DataFrame(X_nik)
X_nik['Class']=0
print('\nDataset 1 extracted')

X_ark = extractor.features(13,"data/arka/")
X_ark= pd.DataFrame(X_ark)
X_ark['Class']=1
print('\nDataset 2 extracted')

X_prv = extractor.features(13,"data/prativa/")
X_prv= pd.DataFrame(X_prv)
X_prv['Class']=2
print('\nDataset 3 extracted')

X_deb = extractor.features(13,"data/debalina/")
X_deb= pd.DataFrame(X_deb)
X_deb['Class']=3
print('\nDataset 4 extracted')

X_arn = extractor.features(13,"data/aryendra/")
X_arn= pd.DataFrame(X_arn)
X_arn['Class']=4
print('\nDataset 5 extracted')

X_avd = extractor.features(13,"data/avadh/")
X_avd= pd.DataFrame(X_avd)
X_avd['Class']=5
print('\nDataset 6 extracted')

X_ksn = extractor.features(13,"data/kasana/")
X_ksn= pd.DataFrame(X_ksn)
X_ksn['Class']=6
print('\nDataset 7 extracted')

X_hrs = extractor.features(13,"data/harsh/")
X_hrs= pd.DataFrame(X_hrs)
X_hrs['Class']=7
print('\nDataset 8 extracted')

X_dhn = extractor.features(13,"data/dhananjay/")
X_dhn= pd.DataFrame(X_dhn)
X_dhn['Class']=8
print('\nDataset 9 extracted')

print('\nData Extraction Complete.')

#defining training and testing data
X_nik_train = X_nik[:int(X_nik.shape[0]*0.8)]
X_nik_test = X_nik[int(X_nik.shape[0]*0.8):]

X_ark_train = X_ark[:int(X_ark.shape[0]*0.8)]
X_ark_test = X_ark[int(X_ark.shape[0]*0.8):]

X_prv_train = X_prv[:int(X_prv.shape[0]*0.8)]
X_prv_test = X_prv[int(X_prv.shape[0]*0.8):]

X_deb_train = X_deb[:int(X_deb.shape[0]*0.8)]
X_deb_test = X_deb[int(X_deb.shape[0]*0.8):]

X_arn_train = X_arn[:int(X_arn.shape[0]*0.8)]
X_arn_test = X_arn[int(X_arn.shape[0]*0.8):]

X_avd_train = X_avd[:int(X_avd.shape[0]*0.8)]
X_avd_test = X_avd[int(X_avd.shape[0]*0.8):]

X_ksn_train = X_ksn[:int(X_ksn.shape[0]*0.8)]
X_ksn_test = X_ksn[int(X_ksn.shape[0]*0.8):]

X_hrs_train = X_hrs[:int(X_hrs.shape[0]*0.8)]
X_hrs_test = X_hrs[int(X_hrs.shape[0]*0.8):]

X_dhn_train = X_hrs[:int(X_dhn.shape[0]*0.8)]
X_dhn_test = X_hrs[int(X_dhn.shape[0]*0.8):]

X_train=X_nik_train.append([X_ark_train,X_prv_train,X_deb_train,X_arn_train,X_avd_train,X_ksn_train,X_hrs_train, X_dhn_train])
X_test=X_nik_test.append([X_ark_test,X_prv_test,X_deb_test,X_arn_test,X_avd_test,X_ksn_test,X_hrs_test, X_dhn_test])

y_train=X_train[['Class']]
y_test=X_test[['Class']]
X_data=X_train.append(X_test)
X_data = X_data.reset_index(drop=True)
y_data=y_train.append(y_test)
y_data = y_data.reset_index(drop=True)
print('\nPre-processing Done.')

print('\nCount of different classes in Train set:')
print(X_train['Class'].value_counts())

print('\nCount of different classes in Test set:')
print(X_test['Class'].value_counts())

feats=[c for c in X_train.columns if c!='Class']

# Train classifier
print('\nImplementing Gaussian Naive Bayes Model.')
gnb = GaussianNB()
gnb.fit(
    X_train[feats].values,
    y_train['Class']
)
y_pred = gnb.predict(X_test[feats].values)

print("\nNumber of mislabeled points out of a total {} points : {}, Accuracy: {:05.5f}%"
      .format(
          X_test.shape[0],
          (X_test["Class"] != y_pred).sum(),
          100*(1-(X_test["Class"] != y_pred).sum()/X_test.shape[0])
))

#five fold cross validation to increase the accuracy
cv = KFold(n_splits=5)
clf = GaussianNB()
X_data=X_data.values
y_data=y_data.values
accuracy=0
for traincv, testcv in cv.split(X_data):
        clf.fit(X_data[traincv], y_data[traincv])
        train_predictions = clf.predict(X_data[testcv])
        acc = accuracy_score(y_data[testcv], train_predictions)
        accuracy+= acc
       
accuracy = 20*accuracy
print('\n5 Fold Cross Validation Accuracy on Training Set: '+str(accuracy))
