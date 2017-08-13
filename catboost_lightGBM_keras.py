#Line 1 to 191: As it is from he_indiahacks2_code.py

########################## CAT-BOOST #########################
##Purpose of catboost: has support for categorical features
#Reference: https://tech.yandex.com/catboost/doc/dg/concepts/python-usages-examples-docpage

from catboost import CatBoostClassifier

#Remember cat_cols will take only ints
cat_cols =[features.index(i) for i in ['num_roadCategory','tr_rd_cat','fl_rd_cat']]

#cols_to_use refers to the columns to be used as features
cols_to_use = features

model = CatBoostClassifier(depth=10, iterations=10, learning_rate=0.1, eval_metric='AUC', random_seed=1,loss_function='MultiClass',use_best_model=True)
#eval_metric : RMSE,Logloss,MAE,CrossEntropy,Quantile,LogLinQuantile,MultiClass,MAPE,Poisson,Recall,Precision,AUC,Accuracy,R2
#See other parameters: https://tech.yandex.com/catboost/doc/dg/concepts/parameter-tuning_trees-number-docpage/
#1) od_pval: Use the Overfitting detector to stop training when the threshold is reached. For best results, it is recommended to set a value in the range 10^-10 to 10^-2; The larger the value, the earlier overfitting is detected.
#2) use_best_model: NEED TO EXPLORE MORE; keeps number of trees
#3) l2_leaf_reg: L2 regularization coefficient. Used for leaf value calculation. Any positive values are allowed.
#4) one_hot_max_size: Convert the feature to float if the number of different values that it takes exceeds the specified value. Ctrs are not calculated for such features.
#5) random_strength: Score standard deviation multiplier.
#6) bagging_temperature: Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is. Typical values are in the range [0, 1] (0 is for no bagging). The default value is 1. Possible values are in the range .
#7) ctr_description: NEED TO EXPLORE MORE; Binarization settings for categorical features. Format: 
'''
	<CTR type 1>:[<number of borders 1>:<Binarization type 1>],...,<CTR type N>:[<number of borders N>:<Binarization type N>]
	Components:
	CTR types:
	Borders
	Buckets
	MeanValue
	Counter
	 The number of borders for target binarization. Only used for regression problems. Allowed values are integers from 1 to 255 inclusively. The default value is 1.
	 The binarization type for the target. Only used for regression problems.
	Possible values:
	Median
	Uniform
	UniformAndQuantiles
	MaxSumLog
	MinEntropy
	GreedyLogSum
	By default, MinEntropy
'''
#8) rsm: Random subspace method. The percentage of features to use at each iteration of building trees. At each iteration, features are selected over again at random. The value must be in the range [0;1].
#9) verbose: Verbose output to stdout
#10) border_count: The number of splits for numerical features. Allowed values are integers from 1 to 255 inclusively.
#11) ctr_border_count: The number of splits for categorical features. Allowed values are integers from 1 to 255 inclusively.
#12) leaf_estimation_method: The method used to calculate the values in leaves. Possible values: i)Newton ii)Gradient
#13) gradient_iterations: The number of gradient steps when calculating the values in leaves.
#14) priors: NEED TO EXPLORE ; Use the specified priors during training. Format: <prior 1>:<prior 2>:...:<prior N>; For example:–2:0:0.5:10
#15) feature_priors	: NEED TO EXPLORE ; Specify individual priors for categorical features (used at the Transforming categorical features to numerical features stage). Given in the form of a comma-separated list of prior descriptions for each specified feature. The description for each feature contains a colon-separated feature index and prior values. Format:<ID of feature 1>:<prior 1.1>:<prior 1.2>:...:<prior 1.N1>,...,<ID of feature M>:<prior M.1>:<prior M.2>:...:<prior M.NM>
#16) fold_permutation_block_size: Objects in the dataset are grouped in blocks before the random permutations. This parameter defines the size of the blocks. The smaller is the value, the slower is the training. Large values may result in quality degradation.
#17) has_time: Use the order of objects in the input data (do not perform random permutations during the Transforming categorical features to numerical features and Choosing the tree structure stages).
#18) fold_len_multiplier: Coefficient for changing the length of folds. The value must be greater than 1. The best validation result is achieved with minimum values. With values close to 1 (for example,  ), each iteration takes a quadratic amount of memory and time for the number of objects in the iteration. Thus, low values are possible only when there is a small number of objects.



#For Binary Classification
#model = CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='Logloss')

model.fit(X_train[features],X_train['noOfLanes_encoded'],cat_features=cat_cols,eval_set = (X_valid[features], X_valid['noOfLanes_encoded']),use_best_model = True)
pred = model.predict(X_test[features])
pred_ans=list(pred[:,0])
#To get probability of predictions
#pred = model.predict_proba(X_test[features])[:,1]

#To get raw 
#pred = model.predict(X_test[features],prediction_type='RawFormulaVal')


####OTHER NOTE: Regression using CatBoost
#from catboost import CatBoostRegressor
#model = CatBoostRegressor(iterations=2, learning_rate=1, depth=2)
# Fit model
#model.fit(train_data, train_labels, cat_features)
# Get predictions
#preds = model.predict(test_data)

#Using user defined objective function i.e. LoglossObjective; Remember LoglossObjective is a python_class
#model = CatBoostClassifier(random_seed=0, loss_function=LoglossObjective(), eval_metric="Logloss")

#Using user defined custom loss function for overfitting detector; Remember LoglossMetric() is a python_class
#model = CatBoostClassifier(iterations=5, random_seed=0, eval_metric=LoglossMetric())


################################ Light GBM #####################
#Source: https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
##Purpose of light gbm: Light GBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithm, used for ranking, classification and many other machine learning tasks.  So when growing on the same leaf in Light GBM, the leaf-wise algorithm can reduce more loss than the level-wise algorithm and hence results in much better accuracy which can rarely be achieved by any of the existing boosting algorithms. Also, it is surprisingly very fast, hence the word ‘Light’.
#Leaf wise splits lead to increase in complexity and may lead to overfitting and it can be overcome by specifying another parameter max-depth which specifies the depth to which splitting will occur.
#Advantages: 1) Faster training speed and higher efficiency 2) Lower memory usage 3) Better accuracy than any other boosting algorithm 4) Compatibility with Large Datasets 5) Parallel learning supported.
#https://media.readthedocs.org/pdf/lightgbm/latest/lightgbm.pdf

import lightgbm as lgb


############### LIGHT GBM TYPE-1 ############### 
model_lgb = lgb.LGBMClassifier(n_estimators=2900, max_depth=3, subsample=0.7, colsample_bytree= 0.7)
model_lgb_fit = model_lgb.fit(X_train[features], X_train["noOfLanes_encoded"])
#For probabilities
#Y_valid = model_lgb_fit.predict_proba(X_valid[features])
#Y_test = model_lgb_fit.predict_proba(X_test[features])
Y_valid = model_lgb_fit.predict(X_valid[features])
Y_test = model_lgb_fit.predict(X_test[features])


############### LIGHT GBM TYPE-2 ###############
train_data=lgb.Dataset(X_train[features],label=X_train["noOfLanes_encoded"])
#POINT TO REMEMBER: class of labels must start from zero: so line 181 of he_indiahacks2_code.py train_set['noOfLanes_encoded']=train_set['noOfLanes']-1 is important
param = {'num_leaves':150, 'objective':'multiclass','max_depth':7,'learning_rate':.05,'max_bin':200,'metric':'auc','num_class':6}
#param['metric'] = ['auc', 'binary_logloss']
num_round=50
lgbm=lgb.train(param,train_data,num_round)
lgbm.feature_importance()
lgbm.save_model('model.txt')

#num_round = 10
#lgbm.cv(param, train_data, num_round, nfold=5)

#predicting on test set
ypred2=lgbm.predict(X_test[features])
ypred2[0:5]  # showing first 5 predictions

'''
#converting probabilities into 0 or 1
for i in range(0,9769):
    if ypred2[i]>=.5:       # setting threshold to .5
       ypred2[i]=1
    else:  
       ypred2[i]=0
'''

####REMEMBER TO RECODE noOFLanes_encode back to previous form
'''
####PARAMETERS
#Reference-1: https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.md
#Reference-2: http://lightgbm.readthedocs.io/en/latest/Parameters.html
#Reference-3: https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/

1) task : default value = train ; options = train , prediction ; Specifies the task we wish to perform which is either train or prediction.
2) application: default=regression, type=enum, options= options :
	regression : perform regression task
	binary : Binary classification
	multiclass: Multiclass Classification
	lambdarank : lambdarank application
3) data: type=string; training data , LightGBM will train from this data
4) num_iterations: number of boosting iterations to be performed ; default=100; type=int
5) num_leaves : number of leaves in one tree ; default = 31 ; type =int
6) device : default= cpu ; options = gpu,cpu. Device on which we want to train our model. Choose GPU for faster training.
7) max_depth: Specify the max depth to which tree will grow. This parameter is used to deal with overfitting.
8) min_data_in_leaf: Min number of data in one leaf.
9) feature_fraction: default=1 ; specifies the fraction of features to be taken for each iteration
10) bagging_fraction: default=1 ; specifies the fraction of data to be used for each iteration and is generally used to speed up the training and avoid overfitting.
11) min_gain_to_split: default=.1 ; min gain to perform splitting
12) max_bin : max number of bins to bucket the feature values.
13) min_data_in_bin : min number of data in one bin
14) num_threads: default=OpenMP_default, type=int ;Number of threads for Light GBM.
15) label : type=string ; specify the label column
16) categorical_feature : type=string ; specify the categorical features we want to use for training our model
17) num_class: default=1 ; type=int ; used only for multi-class classification
'''

### TUNNING LIGHT GBM ARAMETERS
#Reference-1: http://lightgbm.readthedocs.io/en/latest/Parameters-tuning.html
#reference-2: https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
'''
For best fit
1) num_leaves : This parameter is used to set the number of leaves to be formed in a tree. Theoretically relation between num_leaves and max_depth is num_leaves= 2^(max_depth). However, this is not a good estimate in case of Light GBM since splitting takes place leaf wise rather than depth wise. Hence num_leaves set must be smaller than 2^(max_depth) otherwise it may lead to overfitting. Light GBM does not have a direct relation between num_leaves and max_depth and hence the two must not be linked with each other.
2) min_data_in_leaf : It is also one of the important parameters in dealing with overfitting. Setting its value smaller may cause overfitting and hence must be set accordingly. Its value should be hundreds to thousands of large datasets.
3) max_depth: It specifies the maximum depth or level up to which tree can grow.
 

#For faster speed
1) bagging_fraction : Is used to perform bagging for faster results
2) feature_fraction : Set fraction of the features to be used at each iteration
3) max_bin : Smaller value of max_bin can save much time as it buckets the feature values in discrete bins which is computationally inexpensive.
 

#For better accuracy
1) Use bigger training data
2) num_leaves : Setting it to high value produces deeper trees with increased accuracy but lead to overfitting. Hence its higher value is not preferred.
3) max_bin : Setting it to high values has similar effect as caused by increasing value of num_leaves and also slower our training procedure.
'''



################################# KERAS #############################
### TO BE UPDATED
### References: http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
### http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
### http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/


from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
#from sklearn.preprocessing import StandardScaler


#model = Sequential()

#model.add(Dense(8, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))

####Standardize data before scaling
#scaler = StandardScaler().fit(X_train[features])
#strain = scaler.transform(X_train[features])
#stest = scaler.transform(X_test[features])

#One hot encoder
Y_train = to_categorical(X_train['noOfLanes'])
Y_valid = to_categorical(X_valid['noOfLanes'])

input_dim = len(features)
classes = 7
model = Sequential()
model.add(Dense(28, activation = 'relu', input_shape = (input_dim,))) #layer 1
model.add(Dense(30, activation = 'relu')) #layer 2
model.add(Dense(classes, activation = 'sigmoid')) #output
model.compile(optimizer = 'adam', loss='binary_crossentropy',metrics = ['accuracy'])
callback = EarlyStopping(monitor='val_acc',patience=3)
model.fit(X_train[features],Y_train, 100, 50, callbacks=[callback],validation_data=(X_valid[features], Y_valid),shuffle=True)


model = Sequential()
model.add(Dense(12, input_dim=input_dim, activation='relu'))#12 neurons in first layer with number of features = input_dim
#model.add(Dense(100, activation = 'relu', input_shape = (input_dim,))) #layer 1
#model.add(Dense(30, activation = 'relu')) #layer 2
model.add(Dense(classes, activation = 'sigmoid')) #output
model.compile(optimizer = 'adam', loss='binary_crossentropy',metrics = ['accuracy'])

callback = EarlyStopping(monitor='val_acc',patience=3)

model.fit(X_train[features], X_train['noOfLanes'], epochs=150, batch_size=10)
#model.fit(X_train[features], X_train['noOfLanes'], epochs=100, batch_size=5, callbacks=[callback],validation_data=(X_valid[features], X_valid['noOfLanes']),shuffle=True)

# check validation accuracy
vpreds = model.predict_proba(X_valid[features])[:,1]
roc_auc_score(y_true = X_valid[noOfLanes], y_score=vpreds)

# predict on test data
test_preds = model.predict_proba(X_test[features])[:,1]

test_class = model.predict(X_test[features])
