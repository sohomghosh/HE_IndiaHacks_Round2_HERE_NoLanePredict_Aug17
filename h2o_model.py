#Line 1 to 191: As it is from he_indiahacks2_code.py
import h2o

h2o.init()
#h2o.init(nthreads = -1,max_mem_size = "6G") 
h2o.connect()


train_h2o=h2o.H2OFrame(X_train.values.tolist())
train_h2o.set_names(list(X_train.columns))

valid_h2o=h2o.H2OFrame(X_valid.values.tolist())
valid_h2o.set_names(list(X_valid.columns))

test_h2o=h2o.H2OFrame(X_test.values.tolist())
test_h2o.set_names(list(X_test.columns))

#train["Var1"]=train["Var1"].asfactor()

train_h2o["noOfLanes"]=train_h2o["noOfLanes"].asfactor()

features_h2o = features
features_h2
### GRID SEARCH ###o = list(np.setdiff1d(features, ['noOfIntersectingLaneLinesLeft','noOfIntersectingLaneLinesRight','fl_rd_cat']))
##may remove these features
# noOfIntersectingLaneLinesLeft   99.0253                0.0567472            0.0103716
# noOfIntersectingLaneLinesRight  91.3454                0.0523462            0.00956726
# fl_rd_cat                       0                      0                    0


############################################## GBM ###################################################################################
from h2o.estimators.gbm import H2OGradientBoostingEstimator
model_gbm = H2OGradientBoostingEstimator(distribution='multinomial', ntrees=100, max_depth=6, min_rows=1,learn_rate=0.09,sample_rate=.8,seed=44)
#Other parameters Reference: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm.html
model_gbm.train(x=features_h2o, y="noOfLanes", training_frame=train_h2o, validation_frame=valid_h2o)
print(model_gbm)
pred = model_gbm.predict(test_h2o)
pred.head()
submit_pred= pred[:,0]
submit_pred.head()
test_h2o[:,'roadId']=test_h2o[:,'roadId'].asfactor()
submission_dataframe =(test_h2o[:,'roadId']).cbind(submit_pred)
submission_dataframe.set_name(1,"noOfLanes")
submission_dataframe_pd=pd.DataFrame(h2o.as_list(submission_dataframe))
#h2o.h2o.export_file(submission_dataframe, path ="submission_gbm_1.csv")

ans1=submission_dataframe_pd.groupby("roadId",as_index=False)['noOfLanes'].agg(lambda x: x.value_counts().index[0])
ans1.to_csv("sub"+str(i)+".csv",index=False)
i=i+1

#ans1.to_csv("sub33.csv",index=False)

### GRID SEARCH ###
from h2o.grid.grid_search import H2OGridSearch
ntrees_opt = [5,50,100]
max_depth_opt = [2,3,5]
learn_rate_opt = [0.1,0.2]
hyper_params = {'ntrees': ntrees_opt, 'max_depth': max_depth_opt,'learn_rate': learn_rate_opt}
#Other parameters Reference:  http://docs.h2o.ai/h2o/latest-stable/h2o-docs/grid-search.html#grid-search-in-python
gs = H2OGridSearch(H2OGradientBoostingEstimator, hyper_params = hyper_params)
gs.train(x=features_h2o, y="noOfLanes", training_frame=train_h2o, validation_frame=valid_h2o)
print(gs)

for g in gs:
    print(g)

for g in gs:
    print(g.auc)




############################################## Random Forest ###########################################################################

from h2o.estimators.random_forest import H2ORandomForestEstimator
model_rf = H2ORandomForestEstimator(ntrees=250, max_depth=30)
#Other parameters Reference: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/drf.html?highlight=random%20forest
#https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/product/upgrade/H2ODevPortingRScripts.md#DRF
model_rf.train(x=features_h2o,y="noOfLanes",training_frame  =train_h2o,validation_frame=valid_h2o)
print(model_rf)
pred = model_gbm.predict(test_h2o)
pred.head()
submit_pred= pred[:,0]
submit_pred.head()
test_h2o[:,'roadId']=test_h2o[:,'roadId'].asfactor()
submission_dataframe =(test_h2o[:,'roadId']).cbind(submit_pred)
submission_dataframe.set_name(1,"noOfLanes")
submission_dataframe_pd=pd.DataFrame(h2o.as_list(submission_dataframe))
#h2o.h2o.export_file(submission_dataframe, path ="submission_gbm_1.csv")

ans1=submission_dataframe_pd.groupby("roadId",as_index=False)['noOfLanes'].agg(lambda x: x.value_counts().index[0])
ans1.to_csv("sub"+str(i)+".csv",index=False)
i=i+1

### GRID SEARCH ###
from h2o.grid.grid_search import H2OGridSearch
ntrees_opt = [5,50,100]
max_depth_opt = [2,3,5]
hyper_params = {'ntrees': ntrees_opt, 'max_depth': max_depth_opt}
#Other parameters Reference:  http://docs.h2o.ai/h2o/latest-stable/h2o-docs/grid-search.html#grid-search-in-python
gs = H2OGridSearch(H2ORandomForestEstimator, hyper_params = hyper_params)
gs.train(x=features_h2o, y="noOfLanes", training_frame=train_h2o, validation_frame=valid_h2o)
print(gs)

for g in gs:
    print(g)

for g in gs:
    print(g.auc)



############################################## Deep Learning ###########################################################################
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
model_dl = H2ODeepLearningEstimator(distribution="multinomial",activation="RectifierWithDropout", hidden=[100,200,100],input_dropout_ratio=0.2, sparse=True, l1=1e-5, epochs=100)
#Other parameters Reference: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html

model_dl.train(x= features_h2o,y="noOfLanes",training_frame=train_h2o,validation_frame=valid_h2o)
model_dl.params
print(model_dl)

pred = model_dl.predict(test_h2o)
pred.head()
submit_pred= pred[:,0]
submit_pred.head()
test_h2o[:,'roadId']=test_h2o[:,'roadId'].asfactor()
submission_dataframe =(test_h2o[:,'roadId']).cbind(submit_pred)
submission_dataframe.set_name(1,"noOfLanes")
submission_dataframe_pd=pd.DataFrame(h2o.as_list(submission_dataframe))
#h2o.h2o.export_file(submission_dataframe, path ="submission_gbm_1.csv")

ans1=submission_dataframe_pd.groupby("roadId",as_index=False)['noOfLanes'].agg(lambda x: x.value_counts().index[0])
ans1.to_csv("sub"+str(i)+".csv",index=False)
i=i+1

### GRID SEARCH ###
from h2o.grid.grid_search import H2OGridSearch
input_dropout_ratio_li=[0.2,0.4,.06]
epochs_li=[100,200]
hyper_params = {'input_dropout_ratio': input_dropout_ratio_li, 'epochs': epochs_li}
#Other parameters Reference:  http://docs.h2o.ai/h2o/latest-stable/h2o-docs/grid-search.html#grid-search-in-python
gs = H2OGridSearch(H2ODeepLearningEstimator, hyper_params = hyper_params)
gs.train(x=features_h2o, y="noOfLanes", training_frame=train_h2o, validation_frame=valid_h2o)
print(gs)

for g in gs:
    print(g)

for g in gs:
    print(g.auc)



############################################## Auto ML ###########################################################################
from h2o.automl import H2OAutoML
#Other parameters Reference: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
#1) max_runtime_secs: This argument controls how long the AutoML run will execute. This defaults to 3600 seconds (1 hour).
#2) max_models: Specify the maximum number of models to build in an AutoML run. (Does not include the Stacked Ensemble model.)
#3) stopping_metric: Specifies the metric to use for early stopping. Defaults to "AUTO"
#4) stopping_tolerance: This option specifies the relative tolerance for the metric-based stopping to stop the AutoML run if the improvement is less than this value. This value defaults to 0.001 if the dataset is at least 1 million rows; otherwise it defaults to a bigger value determined by the size of the dataset and the non-NA-rate. In that case, the value is computed as 1/sqrt(nrows * non-NA-rate).
#5) leaderboard_frame: This argument allows the user to specify a particular data frame to rank the models on the leaderboard. This frame will not be used for anything besides creating the leaderboard. If this option is not specified, then a leaderboard_frame will be created from the training_frame.
#6) fold_column: Specifies a column with cross-validation fold index assignment per observation. This is used to override the default, randomized, 5-fold cross-validation scheme for individual models in the AutoML run.
#7) weights_column: Specifies a column with observation weights. Giving some observation a weight of zero is equivalent to excluding it from the dataset; giving an observation a relative weight of 2 is equivalent to repeating that row twice. Negative weights are not allowed.
#8) seed: Integer. Set a seed for reproducibility
#9) project_name: Character string to identify an AutoML project

aml = H2OAutoML(max_runtime_secs = 30)
aml.train(x = features_h2o, y = "noOfLanes",training_frame = train_h2o,leaderboard_frame = test_h2o)#NullPointerException

# View the AutoML Leaderboard
lb = aml.leaderboard
lb

aml.leader
preds = aml.predict(test_h2o)
preds = aml.leader.predict(test_h2o)

############################################## XG Boost ###########################################################################
from h2o.estimators.xgboost import H2OXGBoostEstimator
model_xg = H2OXGBoostEstimator(ntrees=100,max_depth=6,min_rows=1)
#Other parameters Refernce: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/xgboost.html

model_xg.train(x= features_h2o,y="noOfLanes",training_frame=train_h2o,validation_frame=valid_h2o)
model_xg.params
print(model_xg)

pred = model_xg.predict(test_h2o)#Check this not working fine
pred.head()



########Save models in h2o
model_path = h2o.save_model(model=model, path="/tmp/mymodel", force=True)#model is like model_dl, model_rf
print(model_path)
#OUTPUT: /tmp/mymodel/DeepLearning_model_python_1441838096933

#########Load models in h2o
saved_model_loaded = h2o.load_model(model_path)

##########Check h2o version
h2o.cluster_info()

######### h2o close session
h2o.shutdown()
h2o.cluster().shutdown()
