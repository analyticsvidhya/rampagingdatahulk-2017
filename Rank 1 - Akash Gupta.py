import pandas as pd
import numpy as np
import datetime
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import operator
from matplotlib import pylab as plt

#Read train test files
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

memid=test['ID']
#separate and drop outcome variables from train
outcome=train['Outcome']
# outcome=np.log(outcome)
drop_fea=['ID','timestamp','Twenty_Day_Moving_Average','Ten_Day_Moving_Average','Five_Day_Moving_Average',
'Three_Day_Moving_Average','Volume']
train_drop_fea=['Outcome']
train.drop(train_drop_fea,axis=1,inplace=True)



#combine train test
ntrain=train.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True)

train_test.fillna(-1,inplace=True)
train_test['minus']=train_test.Three_Day_Moving_Average - train_test.Five_Day_Moving_Average
train_test['minus2']=train_test.Ten_Day_Moving_Average - train_test.Five_Day_Moving_Average
train_test['dirminus']=train_test.Positive_Directional_Movement- train_test.Negative_Directional_Movement
# df=train_test

# train_test.sort(['Stock_ID', 'timestamp'], inplace=True)
# train_test['diffs'] = train_test.Three_Day_Moving_Average.diff()
# mask = train_test.Stock_ID != train_test.Stock_ID.shift(1)
# train_test['diffs'][mask] = np.nan
# train_test.sort(['timestamp','Stock_ID'], inplace=True)
# train_test.fillna(0,inplace=True)

#drop redundant features
train_test.drop(drop_fea,axis=1,inplace=True)
# print train_test.dtypes
#make final train and test files
train=train_test.iloc[:ntrain,:]
test=train_test.iloc[ntrain:,:]

#train test split
x_train,x_val,y_train,y_val=train_test_split(train,outcome,test_size=0.1,random_state=7)
dtrain = xgb.DMatrix(np.array(x_train), label=y_train)
dval=xgb.DMatrix(np.array(x_val),label=y_val)
dtest = xgb.DMatrix(np.array(test))

xgb_params = {
    'seed': 7,
    # 'colsample_bytree': 0.7,
    'silent': 0,
    # 'subsample': 0.7,
    'learning_rate': 0.2,
    'objective': 'binary:logistic',
    'max_depth': 3,
    # 'num_parallel_tree': 1,
    'min_child_weight': 500,
    'eval_metric': 'logloss',
    
}
num_round=800
watchlist=[(dtrain,'train'),(dval,'eval')]

gbdt = xgb.train(xgb_params, dtrain,num_round,watchlist,early_stopping_rounds=50)

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()
ceate_feature_map(list(train.columns))
print train.columns
xgb.plot_importance(gbdt)
# gbdt=gbdt
importance = gbdt.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
print importance
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
# print df
plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False)
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')


#output file
submission=pd.DataFrame({'ID':memid,'Outcome':gbdt.predict(dtest)})
# submission=submission[['member_id','loan_status']]
submission.to_csv('1.csv',index=False)

