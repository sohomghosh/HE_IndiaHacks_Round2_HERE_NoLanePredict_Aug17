import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

train=pd.read_csv("/home/sohom/Desktop/HE_IndiaHacks_ML_Round2/train.csv")
test=pd.read_csv("/home/sohom/Desktop/HE_IndiaHacks_ML_Round2/test.csv")
labels=pd.read_csv("/home/sohom/Desktop/HE_IndiaHacks_ML_Round2/labels.csv")

#removing spaves from column names
train.columns=[str(i).strip() for i in list(train.columns)]
test.columns=[str(i).strip() for i in list(test.columns)]
labels.columns=[str(i).strip() for i in list(labels.columns)]

##labels['noOfLanes'].value_counts()
'''
3    1744
2    1477
4     774
1     656
5     166
6       8
'''

train_set=pd.merge(train,labels,on='roadId',how='inner')

##train.shape
#(28914, 14)

##test.shape
#(2493, 14)

##train_set.shape
#(28914, 16)

##labels.shape
#(4825, 3)

##train_set.columns
#Index(['roadId', 'totalLaneLines', 'laneLineId', 'laneLineCoordinates','totalLinesOnLeft', 'totalLaneLinesOnRight','distFromLaneLineOnLeft','distFromLaneLineOnRight', 'laneLineLength', 'roadLength',   'noOfIntersectingLaneLinesLeft', 'noOfIntersectingLaneLinesRight','isIntersectingWithRoadGeometry', 'roadCategory', 'roadCoordinates','noOfLanes'], dtype='object')

#noOfLanes: TARGET VARIABLE

##set(train_set.columns)-set(test.columns)
#{'roadCoordinates', 'noOfLanes'}

##test['roadCoordinates']='NA'
##test['noOfLanes']='NA'

##train_test=train_set.append(test)

litr_rd_cat=[]
lifl_rd_cat=[]

for bl in train_set['isIntersectingWithRoadGeometry'].values:
	if bl==' false':
		litr_rd_cat.append(0)
		lifl_rd_cat.append(1)
	if bl==' true':
		litr_rd_cat.append(1)
		lifl_rd_cat.append(0)


train_set['tr_rd_cat']=litr_rd_cat
train_set['fl_rd_cat']=lifl_rd_cat

litr_rd_cat=[]
lifl_rd_cat=[]

for bl in test['isIntersectingWithRoadGeometry'].values:
	if bl==' false':
		litr_rd_cat.append(0)
		lifl_rd_cat.append(1)
	if bl==' true':
		litr_rd_cat.append(1)
		lifl_rd_cat.append(0)


test['tr_rd_cat']=litr_rd_cat
test['fl_rd_cat']=lifl_rd_cat


# lbl = LabelEncoder()
# lbl.fit(list(train_test['isIntersectingWithRoadGeometry'].values))
# train_test['num_isIntersectingWithRoadGeometry'] = lbl.transform(list(train_test['isIntersectingWithRoadGeometry'].values))


roadCat_index = {1:5, 2:4, 3:3, 4:2, 5:1} #Reserving the weightage as (1 being the broadest, for e.g. highways)
train_set['num_roadCategory'] = train_set['roadCategory'].replace(to_replace = roadCat_index)

test['num_roadCategory'] = test['roadCategory'].replace(to_replace = roadCat_index)



features=['totalLaneLines','totalLinesOnLeft','totalLaneLinesOnRight','distFromLaneLineOnLeft','distFromLaneLineOnRight','laneLineLength','roadLength','noOfIntersectingLaneLinesLeft','noOfIntersectingLaneLinesRight','num_roadCategory','tr_rd_cat','fl_rd_cat']

train_set['noOfLanes_encoded']=train_set['noOfLanes']-1

train_set['distFromLaneLineOnLeft']=pd.to_numeric(pd.Series(train_set['distFromLaneLineOnLeft']),errors='coerce')
train_set['distFromLaneLineOnRight']=pd.to_numeric(pd.Series(train_set['distFromLaneLineOnRight']),errors='coerce')
test['distFromLaneLineOnLeft']=pd.to_numeric(pd.Series(test['distFromLaneLineOnLeft']),errors='coerce')
test['distFromLaneLineOnRight']=pd.to_numeric(pd.Series(test['distFromLaneLineOnRight']),errors='coerce')


X_train=train_set.sample(frac=0.2, replace=False)
X_valid=pd.concat([train_set, X_train]).drop_duplicates(keep=False)
X_test=test

dtrain = xgb.DMatrix(X_train[features], X_train['noOfLanes_encoded'], missing=np.nan)
dvalid = xgb.DMatrix(X_valid[features],missing=np.nan)
dtest = xgb.DMatrix(X_test[features],missing=np.nan)

params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,"eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,"min_child_weight": 1, "num_class": 6,"seed": 2016, "tree_method": "exact"}


nrounds = 260
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)

valid_preds= bst.predict(dvalid)
test_preds=bst.predict(dtest)
valid_preds_decoded=[int(i+1) for i in valid_preds]
test_preds_decoded=[int(i+1) for i in test_preds]

###print(f1_score(list(X_valid['noOfLanes']),valid_preds_decoded, average='weighted'))

predicted_valid_data=pd.DataFrame({'roadId':X_valid['roadId'].values,'noOfLanes_predicted':valid_preds_decoded})
ans_predicted_valid1=predicted_valid_data.groupby('roadId',as_index=False)['noOfLanes_predicted'].agg(lambda x: x.value_counts().index[0])
ans_true_pred_valid1=pd.merge(labels,ans_predicted_valid1,on='roadId',how='inner')

#noOfLanes  noOfLanes_predicted
print(f1_score(list(ans_true_pred_valid1['noOfLanes']),list(ans_true_pred_valid1['noOfLanes_predicted']), average='weighted'))



###group by: mean take round off
#ans=predicted_test_data.groupby('roadId').mean()
#sub=[int(round(i,0)) for i in list(ans['noOfLanes'])]
#submit = pd.DataFrame({'roadId': list(ans.index), 'noOfLanes': sub})
#submit=submit[['roadId','noOfLanes']]
#submit.to_csv("sub1.csv", index=False)

###group by: majority taking
predicted_test_data=pd.DataFrame({'roadId':test['roadId'].values,'noOfLanes':test_preds_decoded})
ans1=predicted_test_data.groupby('roadId',as_index=False)['noOfLanes'].agg(lambda x: x.value_counts().index[0])
ans1.to_csv("sub2.csv", index=False)

#df.groupby('tag')['category'].agg(lambda x: x.value_counts().index[0])
#f1_score(y_true, y_pred, average='micro') num_roadCategory
#f1_score(labels['noOfLanes'], labels['noOfLanes'], average='weighted')

#h20
#print(gbm_best.cross_validation_metrics_summary())
