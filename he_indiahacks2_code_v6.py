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

################ ADDING NEW FEATURES OVER HERE ##################
train_set['num_laneLineCoordinates']=[len(i.split('|')) for i in train_set['laneLineCoordinates'].values]
test['num_laneLineCoordinates']=[len(i.split('|')) for i in test['laneLineCoordinates'].values]

##### Largest difference logitudes of laneLineCoordinates #####
largest_lat_diff=[]
largest_long_diff=[]
for i in train_set['laneLineCoordinates'].values:
	li_lat=[]
	li_long=[]
	for j in i.strip().split('|'):
		li_lat.append(float(j.split(' ')[0]))
		li_long.append(float(j.split(' ')[1]))
	largest_lat_diff.append(max(li_lat)-min(li_lat))
	largest_long_diff.append(max(li_long)-min(li_long))

train_set['max_lat_diff']=largest_lat_diff
train_set['max_long_diff']=largest_long_diff


largest_lat_diff=[]
largest_long_diff=[]
for i in test['laneLineCoordinates'].values:
	li_lat=[]
	li_long=[]
	for j in i.strip().split('|'):
		li_lat.append(float(j.split(' ')[0]))
		li_long.append(float(j.split(' ')[1]))
	largest_lat_diff.append(max(li_lat)-min(li_lat))
	largest_long_diff.append(max(li_long)-min(li_long))

test['max_lat_diff']=largest_lat_diff
test['max_long_diff']=largest_long_diff

#[float(j.split(' ')[1]) for i in train_set['laneLineCoordinates'].values for j in i.strip().split('|')]


########### Finding minimum difference between lattitudes & longitudes ####################
smallest_lat_diff=[]
smallest_long_diff=[]
for i in train_set['laneLineCoordinates'].values:
	li_lat=[]
	li_long=[]
	for j in i.strip().split('|'):
		li_lat.append(float(j.split(' ')[0]))
		li_long.append(float(j.split(' ')[1]))
	a1, size1 = sorted(li_lat), len(li_lat)
	res1 = [a1[ii + 1] - a1[ii] for ii in range(size1) if ii+1 < size1]
	smallest_lat_diff.append(min(res1))
	a2, size2 = sorted(li_long), len(li_long)
	res2 = [a2[iii + 1] - a2[iii] for iii in range(size2) if iii+1 < size2]
	smallest_long_diff.append(min(res2))


train_set['min_lat_diff']=smallest_lat_diff
train_set['min_long_diff']=smallest_long_diff


smallest_lat_diff=[]
smallest_long_diff=[]
for i in test['laneLineCoordinates'].values:
	li_lat=[]
	li_long=[]
	for j in i.strip().split('|'):
		li_lat.append(float(j.split(' ')[0]))
		li_long.append(float(j.split(' ')[1]))
	a1, size1 = sorted(li_lat), len(li_lat)
	res1 = [a1[ii + 1] - a1[ii] for ii in range(size1) if ii+1 < size1]
	smallest_lat_diff.append(min(res1))
	a2, size2 = sorted(li_long), len(li_long)
	res2 = [a2[iii + 1] - a2[iii] for iii in range(size2) if iii+1 < size2]
	smallest_long_diff.append(min(res2))


test['min_lat_diff']=smallest_lat_diff
test['min_long_diff']=smallest_long_diff


#a, size = sorted(a), len(a)
#res = [a[i + 1] - a[i] for i in range(size) if i+1 < size]
#min(res)



train_set['sum_intersecting_lane_line_left_right']=train_set['noOfIntersectingLaneLinesRight']+train_set['noOfIntersectingLaneLinesLeft']
test['sum_intersecting_lane_line_left_right']=test['noOfIntersectingLaneLinesRight']+test['noOfIntersectingLaneLinesLeft']


#################################################################

features=['totalLaneLines','totalLinesOnLeft','totalLaneLinesOnRight','distFromLaneLineOnLeft','distFromLaneLineOnRight','laneLineLength','roadLength','noOfIntersectingLaneLinesLeft','noOfIntersectingLaneLinesRight','num_roadCategory','tr_rd_cat','fl_rd_cat','num_laneLineCoordinates','max_lat_diff','max_long_diff','min_lat_diff','min_long_diff','sum_intersecting_lane_line_left_right']

train_set['noOfLanes_encoded']=train_set['noOfLanes']-1

train_set['distFromLaneLineOnLeft']=pd.to_numeric(pd.Series(train_set['distFromLaneLineOnLeft']),errors='coerce')
train_set['distFromLaneLineOnRight']=pd.to_numeric(pd.Series(train_set['distFromLaneLineOnRight']),errors='coerce')
test['distFromLaneLineOnLeft']=pd.to_numeric(pd.Series(test['distFromLaneLineOnLeft']),errors='coerce')
test['distFromLaneLineOnRight']=pd.to_numeric(pd.Series(test['distFromLaneLineOnRight']),errors='coerce')


X_train=train_set.sample(frac=0.2, replace=False,random_state=44)
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

###group by: majority taking
predicted_test_data=pd.DataFrame({'roadId':test['roadId'].values,'noOfLanes':test_preds_decoded})
ans1=predicted_test_data.groupby('roadId',as_index=False)['noOfLanes'].agg(lambda x: x.value_counts().index[0])
ans1.to_csv("sub7.csv", index=False)


###group by: mean take round off
#ans=predicted_test_data.groupby('roadId').mean()
#sub=[int(round(i,0)) for i in list(ans['noOfLanes'])]
#submit = pd.DataFrame({'roadId': list(ans.index), 'noOfLanes': sub})
#submit=submit[['roadId','noOfLanes']]
#submit.to_csv("sub1.csv", index=False)


#df.groupby('tag')['category'].agg(lambda x: x.value_counts().index[0])
#f1_score(y_true, y_pred, average='micro') num_roadCategory
#f1_score(labels['noOfLanes'], labels['noOfLanes'], average='weighted')

#h20
#print(gbm_best.cross_validation_metrics_summary())
