import numpy as np
import pandas as pd

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import miceforest as mf


from sklearn.decomposition import PCA

from sklearn.metrics import roc_auc_score

import sklearn.svm as svm
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
import sklearn.naive_bayes as n_bayes

from skopt import BayesSearchCV
from skopt.plots import plot_objective, plot_histogram

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import StackingClassifier


train = pd.read_csv('Training_Data.csv', header=0, index_col=0)
test = pd.read_csv('Test_Data.csv', header=0, index_col=0)

# view
train.head().append(train.tail())
data_describe = train.describe()
train.info()
# view missing and outlier
train.isnull().sum()    # 'clock_speed' -> 1336;  "mobile_wt' -> 30
test.isnull().sum()   # full data

binary_cols = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
num_cols = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'mobile_wt', 'm_dep', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']

cols = binary_cols + num_cols

tmp = pd.DataFrame()
tmp['count'] = train[cols].count().values
tmp['missing_rate'] = (train.shape[0] - tmp['count']) / train.shape[0]
tmp['nunique'] = train[cols].nunique().values
tmp.index = cols

# sns.violinplot(train[num_cols])

# distribution
# sns.distplot(train.price_range)

# numerical feature analysis
tmp = pd.DataFrame(index=num_cols)
for col in num_cols:
    tmp.loc[col,'train_Skewness'] = train[col].skew()
    tmp.loc[col,'test_Skewness'] = test[col].skew()
    tmp.loc[col, 'train_Kurtosis'] = train[col].kurt()
    tmp.loc[col, 'test_Kurtosis'] = test[col].kurt()

# the correlation between features and response
# correlation = train[num_cols + ['price_range']].corr()
# correlation['price_range'].sort_values()

# Visualize the relationship between features
# sns.pairplot(train[num_cols + ['price_range']])


# Complete missing values ['clock_speed', 'mobile_wt']

# GLM method
# col_no_cs_mwt = cols
# col_no_cs_mwt.remove('clock_speed')
# col_no_cs_mwt.remove('mobile_wt')
#
# not_null_cs = train.loc[np.logical_not(train['clock_speed'].isnull()), col_no_cs_mwt]
# null_cs = train.loc[train['clock_speed'].isnull(), col_no_cs_mwt]
# not_null_cs_y = train.loc[np.logical_not(train['clock_speed'].isnull()), 'clock_speed']
#
# not_null_mwt = train.loc[np.logical_not(train['mobile_wt'].isnull()), col_no_cs_mwt]
# null_mwt = train.loc[train['mobile_wt'].isnull(), col_no_cs_mwt]
# not_null_mwt_y = train.loc[np.logical_not(train['mobile_wt'].isnull()), 'mobile_wt']
#
# poly_reg = PolynomialFeatures(degree=3)
# lin_reg = linear_model.LinearRegression()
# not_null_cs_ploy = poly_reg.fit_transform(not_null_cs)
# lin_reg.fit(not_null_cs_ploy,not_null_cs_y)
# predict_cs = lin_reg.predict(poly_reg.fit_transform(null_cs))
# train.loc[train['clock_speed'].isnull(), 'clock_speed'] = predict_cs

# mice forest method
kernel = mf.MultipleImputedKernel(data=train, save_all_iterations=True, random_state=2021)
kernel.mice(3,verbose=True)
new_train = kernel.impute_new_data(train)
new_train = new_train.complete_data(0)

# Feature Engineering
y = new_train.price_range
new_train.drop('price_range',axis=1,inplace=True)
x = new_train

# pca = PCA(n_components=10)
# X = pca.fit_transform(x)



# opt = BayesSearchCV(
#     svm.SVC(),
#     {
#      'C': (1e-6, 1e+6, 'log-uniform'),
#      'gamma': (1e-6, 1e+1, 'log-uniform'),
#      'degree': (1,8),
#      'kernel': (['linear', 'poly', 'rbf']),
#     },
#     n_iter=32,
#     random_state=0,
#     cv=5,
# )
#
# opt.fit(x, y)
#
# print("val. score: %s" % opt.best_score_)
# print("best params: %s" % str(opt.best_params_))
#
# _ = plot_objective(opt.optimizer_results_[0],
#                    dimensions=["C", "degree", 'ceof0', "gamma", "tol"],
#                    n_minimum_search=int(1e8))
# plt.show()




clf1 = svm.SVC(
        kernel='poly',
        degree=1,
        gamma=1.1414652329239456,
        C=0.0001,
    )
clf2 = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=1,
    gamma=0.1,
    learning_rate=0.0688,
    max_depth=10,
    min_child_weight=1,
    subsample=0.9,
    n_estimators=5000
    )
clf3 = KNeighborsClassifier(
    algorithm='auto',
    n_neighbors=10,
    p=1,
    weights='distance')
clf4 = n_bayes.GaussianNB(var_smoothing=2.002678394914125e-09)

models = [('SVC',clf1),('xgb',clf2),('KNC',clf3),('NBC',clf4)]

stacking = StackingClassifier(
    estimators=models,
    final_estimator=LogisticRegression()
)
stacking.fit(x,y)
result=pd.DataFrame({'id':test.index, 'price_range':stacking.predict(test)})
result.to_csv("submission_edition1.csv",index=False)
