import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge

import time

def load_data(trainfile, testfile):
    """
        load raw data
    """
    raw_train = pd.read_csv(trainfile, header=0, index_col=0)
    raw_test = pd.read_csv(testfile, header=0, index_col=0)
    return raw_train, raw_test

raw_train, raw_test = load_data('train.csv', 'test.csv')
full = pd.concat([raw_train,raw_test])
col = full.columns
num_list = []
for item in col:
    if raw_train[item].dtype != 'O':
        num_list.append(item)
type_list = []
for i in full.columns:
    if i not in num_list:
        type_list.append(i)

# data processing

## null value processing
missing = full.isnull().sum()
missing[missing > 0].sort_values(ascending=False)
# Number of features with missing values -->  35
'''
PoolQC          2909
MiscFeature     2814
Alley           2721
Fence           2348
SalePrice       1459
FireplaceQu     1420
LotFrontage      486
GarageYrBlt      159
GarageFinish     159
GarageQual       159
GarageCond       159
GarageType       157
BsmtCond          82
BsmtExposure      82
BsmtQual          81
BsmtFinType2      80
BsmtFinType1      79
MasVnrType        24
MasVnrArea        23
MSZoning           4
BsmtFullBath       2
BsmtHalfBath       2
Functional         2
Utilities          2
BsmtFinSF2         1
BsmtUnfSF          1
BsmtFinSF1         1
TotalBsmtSF        1
SaleType           1
KitchenQual        1
Exterior2nd        1
Exterior1st        1
GarageCars         1
GarageArea         1
Electrical         1
'''

features1 = ["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars",
             "BsmtFinSF2", "BsmtFinSF1", "GarageArea",'GarageYrBlt']
for col in features1:
    full[col].fillna(0, inplace=True)

features2 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu",
             "GarageQual", "GarageCond", "GarageFinish",
             "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual",
             "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in features2:
    full[col].fillna("None", inplace=True)

features3 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities",
             "Functional", "Electrical", "KitchenQual", "SaleType",
             "Exterior1st", "Exterior2nd"]
for col in features3:
    full[col].fillna(full[col].mode()[0], inplace=True)

# LotFrontage : Linear feet of street connected to property
corr = {}
# for item in num_list:
#     corr[item] = full['LotFrontage'].corr(full[item])
# sorted(corr.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
'''
[('LotFrontage', 1.0),
 ('1stFlrSF', 0.4571810019946551),
 ('LotArea', 0.42609501877180805),
 ('GrLivArea', 0.4027974140853199),
 ('TotalBsmtSF', 0.3920745763791724),
 ('TotRmsAbvGrd', 0.3520959476602247),
 ('SalePrice', 0.351799096570678),
 ('GarageArea', 0.3449967241064825),
 ('GarageCars', 0.28569092468510077),
 ('Fireplaces', 0.2666394825603031),
 ('BedroomAbvGr', 0.26316991588105854),
 ('OverallQual', 0.2516457754806126),
 ('BsmtFinSF1', 0.23363316701975537),
 ('PoolArea', 0.20616677527607644),
 ('FullBath', 0.19876867789737834),
 ('MasVnrArea', 0.1934580605582684),
 ('OpenPorchSF', 0.15197222768060673),
 ('BsmtUnfSF', 0.13264374162493384),
 ('YearBuilt', 0.12334946703331692),
 ('BsmtFullBath', 0.10094856694926865),
 ('YearRemodAdd', 0.08886557249206828),
 ('WoodDeckSF', 0.08852093328940812),
 ('2ndFlrSF', 0.08017727062420496),
 ('GarageYrBlt', 0.07024978191661553),
 ('3SsnPorch', 0.07002922773085395),
 ('HalfBath', 0.05353185497960598),
 ('BsmtFinSF2', 0.049899676690989596),
 ('ScreenPorch', 0.041382790675005894),
 ('LowQualFinSF', 0.038468534328959546),
 ('MoSold', 0.011199954759134234),
 ('EnclosedPorch', 0.010700336638882448),
 ('YrSold', 0.007449589209775671),
 ('MiscVal', 0.003367556596191324),
 ('KitchenAbvGr', -0.006068830161309134),
 ('BsmtHalfBath', -0.007234304524918148),
 ('OverallCond', -0.05921345000524684),
 ('MSSubClass', -0.3863468853449292)]
'''
not_null_Lot = full.loc[np.logical_not(full["LotFrontage"].isnull()), "LotArea"]
not_null_1St = full.loc[np.logical_not(full['LotFrontage'].isnull()), '1stFlrSF']
not_Lot_1St = pd.merge(not_null_Lot,not_null_1St,how='outer',on='Id')
not_null_LotF = full.loc[np.logical_not(full["LotFrontage"].isnull()), "LotFrontage"]
null_Lot = full.loc[full["LotFrontage"].isnull(), "LotArea"]
null_1St = full.loc[full['LotFrontage'].isnull(), '1stFlrSF']
Lot_1St = pd.merge(null_Lot, null_1St, how='outer',on='Id')

# 多元多项式拟合（LotArea + 1stFlrSF）
poly_reg_2 = PolynomialFeatures(degree=3)
lin_reg_2=linear_model.LinearRegression()

not_Lot_1St_ploy = poly_reg_2.fit_transform(not_Lot_1St)
lin_reg_2.fit(not_Lot_1St_ploy,not_null_LotF)

Lot_1St_ploy = poly_reg_2.fit_transform(Lot_1St)
predict_LotF = lin_reg_2.predict(Lot_1St_ploy)
predict_LotF[predict_LotF<0] = np.mean(predict_LotF)
full.loc[full['LotFrontage'].isnull(), 'LotFrontage'] = predict_LotF

##--------------------------Features Engineering-----------------------##

# Qualitative features

Sort_var_list = []
for item in type_list:
    if 'Qu' in item or 'Cond' in item or 'QC' in item:
        Sort_var_list.append(item)

Sort_var_list.remove('Condition1')
Sort_var_list.remove('Condition2')
Sort_var_list.remove('SaleCondition')

grading = set(full[Sort_var_list].values.flat)
grading1 = list(filter(lambda x:len(x)<=4, grading))
grading2 = list(filter(lambda x:len(x)>=4, grading))

grading_dict = {'None':0,'Ex':5,'Gd':4,'Fa':2,'Po':1,'TA':3}


def map_values_1():
    full['ExterQual'] = full.ExterQual.map(grading_dict)
    full['ExterCond'] = full.ExterCond.map(grading_dict)
    full['BsmtQual'] = full.BsmtQual.map(grading_dict)
    full['BsmtCond'] = full.BsmtCond.map(grading_dict)
    full['HeatingQC'] = full.HeatingQC.map(grading_dict)
    full['KitchenQual'] = full.KitchenQual.map(grading_dict)
    full['FireplaceQu'] = full.FireplaceQu.map(grading_dict)
    full['GarageQual'] = full.GarageQual.map(grading_dict)
    full['GarageCond'] = full.GarageCond.map(grading_dict)
    full['PoolQC'] = full.PoolQC.map(grading_dict)
    return "Well done"

map_values_1()

for i in Sort_var_list: type_list.remove(i)

type_sublist = ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageFinish','Fence']
for i in type_sublist: type_list.remove(i)

def map_values_2():
    full['BsmtExposure'] = full.BsmtExposure.map({'Gd':4,'Av':3,'Mn':2,'No':1,'None':0})
    full['BsmtFinType1'] = full.BsmtFinType1.map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'None':0})
    full['BsmtFinType2'] = full.BsmtFinType2.map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'None':0})
    full['Electrical'] = full.Electrical.map({'SBrkr':4,'FuseA':3,'FuseF':2,'FuseP':1,'Mix':0})
    full['GarageFinish'] = full.GarageFinish.map({'Fin':3,'RFn':2,'Unf':1,'None':0})
    full['Fence'] = full.Fence.map({'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'None':0})
    return "well done!"
map_values_2()






onehot_df = pd.get_dummies(full[type_list])
full = full.drop(type_list, axis=1)
full = pd.merge(full,onehot_df,how='outer',on='Id')


# Quantitative feature

# PCA
y = full.SalePrice[:len(raw_train)]
y_log = np.log(y)
full.drop(['SalePrice'],axis=1,inplace=True)
train = full[:len(raw_train)]
test = full[len(raw_train):]

pca = PCA(n_components=219)

X_scaled = pca.fit_transform(train)
test_X_scaled = pca.transform(test)

#------------------------Basic Modeling & Evaluation----------------------#

def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse
# models = [LinearRegression(),
#           RandomForestRegressor(),
#           GradientBoostingRegressor(),
#           Lasso(alpha=0.01,max_iter=10000),
#           Ridge(),
#           ElasticNet(alpha=0.001,max_iter=10000),
#           SVR(),
#           LinearSVR(),
#           SGDRegressor(max_iter=1000,tol=1e-3),
#           BayesianRidge()
#           ]
models = [
    GradientBoostingRegressor(),
    Lasso(alpha=0.01,max_iter=10000),
    Ridge(),
    ElasticNet(alpha=0.001,max_iter=10000),
    SVR(),
    BayesianRidge()
]

# import xgboost
#
# xg = xgboost.XGBRegressor()


names = ["GBR","Lasso","Ridge",'ElasticNet',"SVR","BayesianRidge"]

dict = {'Id':[],"GBR":[],"Lasso":[],"Ridge":[],'ElasticNet':[],"SVR":[],"BayesianRidge":[]}
# for i in range(215,225,1):
#     pca = PCA(n_components=i)
#     X_scaled = pca.fit_transform(train)
#     test_X_scaled = pca.transform(test)
#     dict['Id'].append(i)
#     for name, model in zip(names, models):
#         score = rmse_cv(model, X_scaled, y_log)
#         dict[name].append(score.mean())
#         # print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))
# df = pd.DataFrame(dict,index=dict["Id"])
# df = df.drop("Id",axis=1)
# df.plot(kind='line')
# plt.ylabel('RMSE'),plt.xlabel('n_components')
# plt.show()


# Automatic tuning hyper-parameters

class grid():
    def __init__(self, model):
        self.model = model

    def grid_get(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X, y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])
        print(grid_search.best_params_)

        # for key, values in param_grid.items():
        #     plt.subplot(121)
        #     plt.title(key)
        #     plt.plot(values,grid_search.cv_results_['mean_test_score'])
        #     plt.ylabel('RMSE')
        #     plt.subplot(122)
        #     plt.plot(values,grid_search.cv_results_['std_test_score']/
        #              grid_search.cv_results_['mean_test_score'])
        #     plt.ylabel('STD')
        #     plt.show()


## GBR
# step 1 : n_estimators         ----> 211
# param_test1 = {'n_estimators':range(205,216,1)}
# grid(GradientBoostingRegressor()).\
#     grid_get(X_scaled,y_log,param_grid=param_test1)

# step 2 :  max_depth           ----> 3
# param_test2 = {'max_depth':range(3,14,2)}
# grid(GradientBoostingRegressor(n_estimators=211)).\
#     grid_get(X_scaled,y_log,param_grid=param_test2)
#
# step 3 :  min_samples_split   ----> 15
# param_test3 = {'min_samples_split':range(10,20,1)}
# grid(GradientBoostingRegressor(n_estimators=211,max_depth=3)).\
#     grid_get(X_scaled,y_log,param_grid=param_test3)

# step 4 :  min_samples_leaf    ----> 8
# param_test4 = {'min_samples_leaf':range(1,11,1)}
# grid(GradientBoostingRegressor(
#     n_estimators=211,max_depth=3,min_samples_split=15)).\
#     grid_get(X_scaled,y_log,param_grid=param_test4)
#
# step 5 :  subsample           ----> 0.7
# param_test5 = {'subsample':np.arange(0.1,1.1,0.1)}
# grid(GradientBoostingRegressor(n_estimators=211,
#                                max_depth=3,
#                                min_samples_split=15,
#                                min_samples_leaf=8)).\
#     grid_get(X_scaled,y_log,param_grid=param_test5)


## Lasso
# step 1 :  learning rate       ----> 0.0003
#           max iteration       ----> 1000
# param_test1 = {'alpha':np.arange(0.0001,0.0011,0.0001),
#                'max_iter':range(1000,11000,1000)}
# grid(Lasso()).grid_get(X_scaled,y_log,param_grid=param_test1)

## Ridge
# step 1 :  learning rate       ----> 10.3
# param_test1 = {'alpha':np.arange(9,11,0.1)}
# grid(Ridge()).grid_get(X_scaled,y_log,param_grid=param_test1)

## BayesianRidge
# param_test1 = {'alpha_1':range(20,31,1),
#                # 'alpha_2':range(1,101,1),
#                'lambda_1':range(1,6,1)}
#                # 'lambda_2':range(1,101,1)
# time1 = time.perf_counter()
# grid(BayesianRidge()).grid_get(X_scaled,y_log,param_grid=param_test1)
# time2 = time.perf_counter()
# print(time2-time1)

## ElasticNet alpha 0.007, l1_ratio 0.01
# param_test1 = {'alpha':np.arange(0.001,0.011,0.001),
#                'l1_ratio':np.arange(1,11,1)}
# grid(ElasticNet()).grid_get(X_scaled,y_log,param_grid=param_test1)



models_set = [GradientBoostingRegressor(n_estimators=211,
                                        max_depth=3,
                                        min_samples_split=15,
                                        min_samples_leaf=8,
                                        subsample=0.7),
              Lasso(alpha=0.0003,max_iter=1000),
              Ridge(alpha=10.3),
              BayesianRidge(),
              ElasticNet(alpha=0.007,l1_ratio=0.01)]


#-------------------------------Stacking-----------------------------------#
def get_stacking(model, x_train, y_train, x_test, n_folds=5):
    x, y, z = x_train, y_train.values, x_test
    x_num, z_num = x.shape[0], z.shape[0]
    second_layer_x = np.zeros((x_num,))
    second_layer_z = np.zeros((z_num,))

    z_nfolds_sets = np.zeros((z_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i, (x_index, z_index) in enumerate(kf.split(x)):
        x_tra, y_tra = x[x_index], y[x_index]
        x_tst, y_tst = x[z_index], y[z_index]
        model.fit(x_tra, y_tra)
        second_layer_x[z_index] = model.predict(x_tst)
        z_nfolds_sets[:, i] = model.predict(z)
        second_layer_z[:] = z_nfolds_sets.mean(axis=1)
    return second_layer_x, second_layer_z

def stacking(models_set, x_train, y_train, x_test):
    combine_second_layer_x = np.zeros((x_train.shape[0], len(models_set)))
    combine_second_layer_z = np.zeros((x_test.shape[0], len(models_set)))
    for i, model in enumerate(models_set):
        second_layer_x, second_layer_z = get_stacking(model, x_train,y_train,x_test)
        combine_second_layer_x[:, i] = second_layer_x
        combine_second_layer_z[:, i] = second_layer_z
    lr = LinearRegression()
    lr.fit(combine_second_layer_x, y_train)
    y_test = lr.predict(combine_second_layer_z)
    return y_test


y_test = np.exp(stacking(models_set, X_scaled,y_log,test_X_scaled))
result=pd.DataFrame({'Id':raw_test.index, 'SalePrice':y_test})
result.to_csv("submission.csv",index=False)