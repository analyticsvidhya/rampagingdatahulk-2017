mport numpy as np
import pandas as pd
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)

pd.options.display.mpl_style = 'default'


# In[2]:

train = pd.read_csv("train.csv")


# In[3]:

test = pd.read_csv("test.csv")


# In[4]:

#feature based on compariosn of third moving average and fifth moving average
def f35(row):
    if row['Three_Day_Moving_Average'] == row['Five_Day_Moving_Average']:
        val = 0
    elif row['Three_Day_Moving_Average'] > row['Five_Day_Moving_Average']:
        val = 1
    else:
        val = -1
    return val


# In[5]:

#feature based on compariosn of third moving average and tenth moving average
def f310(row):
    if row['Three_Day_Moving_Average'] == row['Ten_Day_Moving_Average']:
        val = 0
    elif row['Three_Day_Moving_Average'] > row['Ten_Day_Moving_Average']:
        val = 1
    else:
        val = -1
    return val


# In[6]:

#feature based on compariosn of third moving average and twenty day moving average
def f320(row):
    if row['Three_Day_Moving_Average'] == row['Twenty_Day_Moving_Average']:
        val = 0
    elif row['Three_Day_Moving_Average'] > row['Twenty_Day_Moving_Average']:
        val = 1
    else:
        val = -1
    return val


# In[7]:

#feature based on compariosn of Five_Day_Moving_Average and tenth moving average
def f510(row):
    if row['Five_Day_Moving_Average'] == row['Ten_Day_Moving_Average']:
        val = 0
    elif row['Five_Day_Moving_Average'] > row['Ten_Day_Moving_Average']:
        val = 1
    else:
        val = -1
    return val


# In[8]:

#feature based on compariosn of Five_Day_Moving_Average and twenty moving average
def f520(row):
    if row['Five_Day_Moving_Average'] == row['Twenty_Day_Moving_Average']:
        val = 0
    elif row['Five_Day_Moving_Average'] > row['Twenty_Day_Moving_Average']:
        val = 1
    else:
        val = -1
    return val


# In[9]:

#feature based on compariosn of Ten_Day_Moving_Average and twenty moving average
def f1020(row):
    if row['Ten_Day_Moving_Average'] == row['Twenty_Day_Moving_Average']:
        val = 0
    elif row['Ten_Day_Moving_Average'] > row['Twenty_Day_Moving_Average']:
        val = 1
    else:
        val = -1
    return val


# In[10]:

train['Compare_3_5'] = train.apply(f35, axis=1)
test['Compare_3_5'] = test.apply(f35, axis=1)


# In[11]:

train['Compare_3_10'] = train.apply(f310, axis=1)
test['Compare_3_10'] = test.apply(f310, axis=1)


# In[12]:

train['Compare_3_20'] = train.apply(f320, axis=1)
test['Compare_3_20'] = test.apply(f320, axis=1)


# In[13]:

train['Compare_5_10'] = train.apply(f510, axis=1)
test['Compare_5_10'] = test.apply(f510, axis=1)


# In[14]:

train['Compare_5_20'] = train.apply(f520, axis=1)
test['Compare_5_20'] = test.apply(f520, axis=1)


# In[15]:

train['SUM_3'] = train['Compare_3_5']+train['Compare_3_10']+train['Compare_3_20']


# In[16]:

test['SUM_3'] = test['Compare_3_5']+test['Compare_3_10']+test['Compare_3_20']


# In[17]:

#train['Compare_10_20'] = train.apply(f1020, axis=1)
#test['Compare_10_20'] = test.apply(f1020, axis=1)


# In[18]:

#dropping unnecessary columns
train.drop(train.columns[[ 0,1, 2,4,5,6,7]], axis=1, inplace= True)


# In[19]:

train.head()


# In[20]:

#train.info()


# As we can clearly see there are null values, let us find out how many values are null

# In[21]:

#train.isnull().sum()


# In[22]:

#test.info()


# In[23]:

#test.isnull().sum()


# Filling Null values in ATR with TR values

# In[24]:

#train['Average_TR']= train['Average_True_Range']
#test['Average_TR']= test['Average_True_Range']


# In[25]:

#train.Average_TR.fillna(train.True_Range, inplace=True)
#test.Average_TR.fillna(test.True_Range, inplace=True)


# In[26]:

#train['PlusADX']= (train['Positive_Directional_Movement']/train['Average_TR'])
#train['MinusADX']= (train['Negative_Directional_Movement']/train['Average_TR'])
#test['PlusADX']= (test['Positive_Directional_Movement']/test['Average_TR'])
#test['MinusADX']= (test['Negative_Directional_Movement']/test['Average_TR'])


# In[27]:

#dropping fieds which were used to calculate plus and minus ADX
#train.drop('Positive_Directional_Movement', axis=1, inplace = True)
#train.drop('Average_TR', axis=1, inplace = True)
#train.drop('Negative_Directional_Movement', axis=1, inplace = True)


# In[28]:

#test.drop('Positive_Directional_Movement', axis=1, inplace = True)
#test.drop('Average_TR', axis=1, inplace = True)
#test.drop('Negative_Directional_Movement', axis=1, inplace = True)


# In[29]:

x_train = train
y_train = train['Outcome'].values


# In[30]:

x_train.drop('Outcome', axis=1, inplace = True)


# In[31]:

x_train.head()


# In[32]:

#checking correlation plot

#corr=train.corr()
#plt.figure(figsize=(10, 10))

#sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='cubehelix')
#plt.title('Correlation between features')


# In[33]:

# Finally, we split some of the data off for validation

from sklearn.cross_validation import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)


# In[34]:

import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.11 #0.02
params['max_depth'] = 5
#params['subsample']=0.7
#params['min_child_weight']=1
#params['colsample_bytree']=0.7
params['silent']=1
#params['seed']=1632

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, verbose_eval=10)


# In[35]:

test.head()
df_test = pd.DataFrame()
df_test['ID'] =  test['ID']


# In[36]:

test.drop(test.columns[[  0,1, 2,4,5,6,7]], axis=1, inplace= True)


# In[37]:

test.head()


# In[38]:

d_test = xgb.DMatrix(test)
p_test = bst.predict(d_test)


# In[39]:

sub = pd.DataFrame()
sub['ID'] = df_test['ID']
sub['Outcome'] = p_test
sub.to_csv('2.csv', index=False)
