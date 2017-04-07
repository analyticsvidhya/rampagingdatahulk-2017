

import graphlab
import numpy as np

## Load the train and test data
train = graphlab.SFrame('train.csv')
test = graphlab.SFrame('test.csv')


# In[ ]:

### This is just for clarity purpose .. Combining all the relevant steps together from the final solution and discarding the 
## code that is not required



## Feature engineering 
def std_compute(row):
    return np.std([np.int(row['Three_Day_Moving_Average']),np.int(row['Five_Day_Moving_Average']),np.int(row['Ten_Day_Moving_Average'])])
    
def create_feature(df):
    df['Three_Day_Moving_Average_na'] = df['Three_Day_Moving_Average'].apply(lambda x:1 if x == None else 0)
    df['Five_Day_Moving_Average_na'] = df['Five_Day_Moving_Average'].apply(lambda x:1 if x == None else 0)
    df['Ten_Day_Moving_Average_na'] = df['Ten_Day_Moving_Average'].apply(lambda x:1 if x == None else 0)
    df['Twenty_Day_Moving_Average_na'] = df['Twenty_Day_Moving_Average'].apply(lambda x:1 if x == None else 0)
    df = df.fillna('Three_Day_Moving_Average',99999)
    df = df.fillna('Five_Day_Moving_Average',99999)
    df = df.fillna('Ten_Day_Moving_Average',99999)
    df = df.fillna('Twenty_Day_Moving_Average',99999)
    df['MA_3_5']  = df['Three_Day_Moving_Average'] - df['Five_Day_Moving_Average']
    df['MA_3_10']  = df['Three_Day_Moving_Average'] - df['Ten_Day_Moving_Average']
    df['MA_5_10']  = df['Five_Day_Moving_Average'] - df['Ten_Day_Moving_Average']
    df['MA_20_10']  = df['Twenty_Day_Moving_Average'] - df['Ten_Day_Moving_Average']
    df['MA_20_5']  = df['Twenty_Day_Moving_Average'] - df['Five_Day_Moving_Average']
    df['MA_20_3']  = df['Twenty_Day_Moving_Average'] - df['Three_Day_Moving_Average']
    ## releasing the vol features
    df['vol_3']    = df['Three_Day_Moving_Average'] - df['Volume']
    df['vol_5']    = df['Five_Day_Moving_Average'] - df['Volume']
    df['vol_10']    = df['Ten_Day_Moving_Average'] - df['Volume']
    df['vol_20']    = df['Twenty_Day_Moving_Average'] - df['Volume']
    df['Direction_add'] = df['Positive_Directional_Movement'] + df['Negative_Directional_Movement']
    df['Direction_sub'] = df['Positive_Directional_Movement'] - df['Negative_Directional_Movement']
    df['t_a']  = df['True_Range'] - df['Average_True_Range']
    # last hour feature'
    df['t_a_p']  = df['True_Range'] + df['Average_True_Range']
    df['MA_last_10_3'] = (df['Ten_Day_Moving_Average']*10 - df['Three_Day_Moving_Average']*3)/7
    df['MA_last_10_5'] = (df['Ten_Day_Moving_Average']*10 - df['Five_Day_Moving_Average']*5)/5
    df['MA_last_5_3'] = (df['Five_Day_Moving_Average']*5 - df['Three_Day_Moving_Average']*3)/2
    # last hour feature'
    #df['MA_std'] = df.apply(std_compute)
    return df
train = create_feature(train)
test = create_feature(test)


## exclude the unwanted features
features = test.column_names()
features_exclude = ['ID','timestamp','Stock_ID']
features1 = []

for f in features :
    if f not in features_exclude:
        features1.append(f)
train_1,val_1 = train.random_split(0.8,seed=0)        
x_train = train_1[features1].to_numpy()
y_train = train_1['Outcome'].to_numpy()
x_test = val_1[features1].to_numpy()
y_test = val_1['Outcome'].to_numpy()

import xgboost as xgb

dTrain = xgb.DMatrix(x_train, label=y_train)
dVal   = xgb.DMatrix(x_test, label=y_test)

xgb_params = {
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'eval_metric': 'logloss',
    'eta': 0.02, 
    'max_depth': 6,
    'lambda': 2,
    'alpha': 0.02,
    'subsample': 0.8,
    #'colsample_bytree': 1 / F_train.shape[1]**0.5,
    'colsample_bytree': 0.8,
    'min_child_weight': 4,
    'silent': 1
}
bst = xgb.train(xgb_params, dTrain, 700,  [(dTrain,'train'),((dVal,'val'))], 
                verbose_eval=10, early_stopping_rounds=20)


## predictions on the final model... 
F_test = test[features1].to_numpy() 
dTest = xgb.DMatrix(F_test.astype('float32'))
pred = bst.predict(dTest, ntree_limit=bst.best_ntree_limit)
results2 = graphlab.SFrame()
results2['ID'] = test['ID']
results2['Outcome'] = pred
results2.save('submission_xgb5.csv')






