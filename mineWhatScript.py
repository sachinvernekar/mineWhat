import numpy as np
import pandas as pd
import cPickle as pickle
import scipy.sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from scipy.sparse import hstack
import xgboost as xgb

train = pd.read_csv("productsOrig.csv")
header = train.columns




######get one grams##############


train = np.array(train)


uncatInd = np.where(train[:,2] == '"uncategorised"')[0]
cat = train[:,-5]
uniqueCat = np.unique(train[:,-5]) #####141 categories


nameVec = CountVectorizer(min_df=1)
descVec = CountVectorizer(min_df=1)

name = train[:,-7]
desc = train[:,-6]


nameX = nameVec.fit_transform(name)
descX = descVec.fit_transform(desc)

X = hstack((nameX,descX))
#X = np.concatenate((nameX,descX),axis=1)

X = X.toarray()
X_test = X[uncatInd,:]
X_train = np.delete(X,uncatInd,axis=0)
#cat = np.array(cat)
cat = np.delete(cat,uncatInd)

lbl = preprocessing.LabelEncoder()
lbl.fit(list(cat))
labels = lbl.transform(cat)

tr = pd.DataFrame(X_train)
te = pd.DataFrame(X_test)

tr.to_csv("trainMat.cv",index=None,header=None)
te.to_csv("testMat.csv",index=None,header=None)

l = pd.DataFrame(labels)
l.to_csv("labels.csv",index=None,header=None)

X_train = scipy.sparse.csr_matrix(X_train)
X_test = scipy.sparse.csr_matrix(X_test)

with open('train_sparse_mat.dat', 'wb') as outfile:
	pickle.dump(X_train, outfile, pickle.HIGHEST_PROTOCOL)


with open('test_sparse_mat.dat', 'wb') as outfile:
	pickle.dump(X_test, outfile, pickle.HIGHEST_PROTOCOL)



params = {"objective": "multi:softprob",
          "eta": 0.01,# used to be 0.2 or 0.1
          "max_depth": 15, # used to be 5 or 6
          "min_child_weight": 1,
		  "max_delta_step": 6,
          "silent": 1,
          "colsample_bytree": 0.7,
		  "subsample": 0.8,
		  "eval_metric" : "merror",
          "seed": 1,
		  "num_class": 141}

num_rounds = 1000

xgtrain = xgb.DMatrix(X_train,labels)
xgb.cv(params, xgtrain, num_rounds, nfold=4)
num_rounds = 2000
watchlist = [(xgtrain, 'train')]
gbm2 = xgb.train(params, xgtrain, num_rounds, watchlist)

xgtest = xgb.DMatrix(X_test)
preds = gbm2.predict(xgtest)

test_labels = np.zeros(X_test.shape[0])
for i in xrange(0,36):
	test_labels[i] = np.argmax(preds[i,:])


test_labels = test_labels.astype('int')
k = lbl.inverse_transform(test_labels)



train[uncatInd,-5] = k

for i in xrange(0,36):
	k[i] = "["+k[i]+"]"

train_out = train;
g=k


preds = pd.DataFrame(train_out)
preds.columns = header
preds.to_csv('solutionCat.csv',index=None)




