'''
settings for Vmr
'''

import numpy as np
#np.random.seed(27)    # St seed f consistency thruot

### algos t use f RandomizedSearchCV & GridSearchCV ###
algos_t_use = [
    'LogisticRegression',

#    'KNeighborsClassifier',
#    'SVC',

#    'GaussianNB',
###    'GaussianProcessClassifier',
#    'QuadraticDiscriminantAnalysis',

#    'DecisionTreeClassifier',
#    'RandomForestClassifier',
#    'GradientBoostingClassifier',
#    'AdaBoostClassifier',

#    'XGBClassifier',
#    'MLPClassifier',
]

### n_iter f RandomizedSearchCV ###
n_iter_dict = dict(
    LogisticRegression = 10,
    KNeighborsClassifier = 2,
    SVC = 2,
    QuadraticDiscriminantAnalysis = 1,
    GaussianNB = 1,
    GaussianProcessClassifier = 1,
    DecisionTreeClassifier = 2,
    RandomForestClassifier = 2,
    GradientBoostingClassifier = 2,
    AdaBoostClassifier = 2,
    XGBClassifier = 2,
    MLPClassifier = 2,
)

ensembles_t_use = [
    'simple_average',

    'accuracy_weighted',
    'f1_score_weighted',
    'logloss_weighted',

    'softmax_accuracy_weighted',
    'softmax_f1_score_weighted',
    'softmax_logloss_weighted',
]