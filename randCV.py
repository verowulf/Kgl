from LoadData import *
from V_ML_algos import *
from V_time_precise import time_precise

from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, expon
from sklearn.gaussian_process.kernels import RBF

### Parameter distributns f RandomizedSearchCV ###
param_dist_dict = dict(
    LogisticRegression = dict(
        lr__solver = ['sag', 'newton-cg', 'lbfgs'],  # 'liblinear gd f small dtsts
        lr__C = expon(0, 1),
        #lr__n_jobs = [-1],
    ),
    KNeighborsClassifier = dict(
        knn__n_neighbors = randint(3, 15),
        knn__p = [1, 2],
    ),
    SVC = dict(
        svc__kernel = ['rbf', 'poly', 'sigmoid'],  # 'linear'
        svc__gamma = [0.03, 0.1, 0.3, 1],  # for 'rbf', 'poly', 'sigmoid'
        svc__C = randint(1, 20),
        svc__probability = [True],
        svc__max_iter = [2000],
        svc__verbose = [True],
    ),
    QuadraticDiscriminantAnalysis = dict(
        qda__priors = [[0.5, 0.5]],
    ),
    GaussianNB = dict(
        gnb__priors = [[0.5, 0.5]],
    ),
    GaussianProcessClassifier = dict(
        gp__kernel = [1.0 * RBF(1.0)],
        gp__warm_start = [True],
        gp__max_iter_predict = [2],
        #gp__n_jobs = [-1],
    ),
    DecisionTreeClassifier = dict(
        dt__criterion = ['gini', 'entropy'],  # Gini impurity, Information gain --> typically vy similar results
        dt__splitter = ['best', 'random'],
        dt__max_features = [3],
        dt__max_depth = randint(3, 15),
        dt__min_samples_split = randint(3, 11),
        dt__min_samples_leaf = randint(2, 11),
    ),
    RandomForestClassifier = dict(
        rf__n_estimators = randint(20, 300),
        rf__criterion = ['gini', 'entropy'],  # Gini impurity, Information gain --> typically vy similar results
        rf__max_features = ['sqrt'],  # 'log2', 3
        rf__max_depth = randint(3, 15),
        rf__min_samples_split = randint(3, 11),
        rf__min_samples_leaf = randint(2, 11),
        rf__bootstrap = [True, False],
        #rf__warm_start = [False],
        #rf__n_jobs = [-1],
    ),
    GradientBoostingClassifier = dict(
        gb__n_estimators = randint(20, 100),
        gb__learning_rate = expon(0, 0.2),
        gb__loss = ['deviance'],  # logistic regression for probabilistic outputs
        gb__criterion = ['friedman_mse'],  # 'mse', 'mae'],  # 'friedman_mse' generally the best
        gb__max_features = [3],
        gb__max_depth = randint(3, 5),
        gb__min_samples_split = randint(3, 11),
        gb__min_samples_leaf = randint(2, 11),
        gb__warm_start = [True, False],
    ),
    AdaBoostClassifier = dict(
        #base_estimator = ['DecisionTreeClassifier'],  # I need a better understanding for this
        adb__n_estimators = randint(20, 100),
        adb__learning_rate = expon(0, 0.2),
        adb__algorithm = ['SAMME.R', 'SAMME'],
    ),
    XGBClassifier = dict(
        xgb__objective = ['binary:logistic'],
        xgb__n_estimators = randint(100, 300),
        xgb__max_depth = randint(3, 8),
        xgb__min_child_weight = [1],
        xgb__learning_rate = [0.2, 0.3, 0.4],
        xgb__subsample = [0.6, 0.7, 0.8],
        xgb__colsample_bytree = [0.6, 0.7, 0.8],
        #xgb__nthread = [-1],
    ),
    MLPClassifier=dict(
        mlp__hidden_layer_sizes = [(21, 15, 5), (21, 10, 5), (21, 10, 3), (21, 30, 15, 5)],
        mlp__activation = ['relu', 'tanh', 'logistic', 'identity'],
        mlp__alpha = expon(0, 0.02),  # L2 regularization
        mlp__max_iter = [400],
        mlp__warm_start = [True, False],

        mlp__solver = ['adam'],
        mlp__beta_1 = [0.9, 0.99, 0.999],  # only when solver='adam'
        mlp__beta_2 = [0.9, 0.999, 0.99999],  # only when solver='adam'
        mlp__epsilon = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4],  # only when solver='adam'
        mlp__learning_rate_init = expon(0, 0.02),  # only when solver='sgd' or 'adam'
        #mlp__batch_size = randint(200, 1000),  # minibatch not used if solver='lbfgs'

        #mlp__solver = ['sgd'],
        #mlp__learning_rate = ['adaptive', 'constant', 'invscaling'],  # only when solver='sgd'
        #mlp__learning_rate_init = expon(0, 0.02),  # only when solver='sgd' or 'adam'
        ##mlp__batch_size = randint(200, 1000),  # minibatch not used if solver='lbfgs'

        #mlp__solver = ['lbfgs'],
    ),
)

ls_prbbs_train = []
ls_prbbs_validation = []
ls_prbbs_test_n_live = []

ls_accuracy_validation = []
ls_f1_score_validation = []
ls_logloss_validation = []

def runRandCV(algo):
    print('# ' + algo + ' ##############################')

    cv = 5  # stratified k fold CV

    randCV = RandomizedSearchCV(
        model_dict[algo],
        param_dist_dict[algo],
        n_iter_dict[algo],
        cv=cv,
        refit=True,
        n_jobs=-1,
        verbose=1,
        scoring='neg_log_loss',
    )

    # ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples',
    # 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error',
    # 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2',
    # 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']

    start_randCV = time_precise()
    randCV.fit(X_train, y_train)
    print('\nRandomizedSearchCV took %.3f seconds\n' % (time_precise() - start_randCV))

    reportCV(randCV.cv_results_)
    runModel(randCV)

# report best scores
def reportCV(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print('Model with rank: {0}'.format(i))
            print('Mean validation score: {0:.3f} (std: {1:.3f})'.format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print('Parameters: {0}\n'.format(results['params'][candidate]))

def runModel(model):
    preds_train = model.predict(X_train)
    preds_validation = model.predict(X_validation)

    print('accuracy of train :      %.2f %%' % (100 * accuracy_score(y_train, preds_train)))
    accuracy_validation = accuracy_score(y_validation, preds_validation)
    ls_accuracy_validation.append(accuracy_validation)
    print('accuracy of validation : %.2f %%' % (100 * accuracy_validation))

    print('f1 score of train :      %.2f %%' % (100 * f1_score(y_train, preds_train)))
    f1_score_validation = f1_score(y_validation, preds_validation)
    ls_f1_score_validation.append(f1_score_validation)
    print('f1 score of validation : %.2f %%' % (100 * f1_score_validation))

    prbbs_train = model.predict_proba(X_train)[:, 1]
    prbbs_validation = model.predict_proba(X_validation)[:, 1]
    prbbs_test_n_live = model.predict_proba(X_test_n_live)[:, 1]
    ls_prbbs_train.append(prbbs_train)
    ls_prbbs_validation.append(prbbs_validation)
    ls_prbbs_test_n_live.append(prbbs_test_n_live)

    print('log loss of train :      %f' % log_loss(y_train, prbbs_train))
    logloss_validation = log_loss(y_validation, prbbs_validation)
    ls_logloss_validation.append(logloss_validation)
    print('log loss of validation : %f\n' % logloss_validation)