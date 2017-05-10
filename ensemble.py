from randCV import *
#from gridCV import *
from V_softmax_whts import softmax_whts

### Ensemble ###

ls_ens_prbbs_tournament = []

def runEnsemble():
    na_prbbs_train = np.array(ls_prbbs_train)
    na_prbbs_validation = np.array(ls_prbbs_validation)
    na_prbbs_test_n_live = np.array(ls_prbbs_test_n_live)

    na_accuracy_validation = np.array(ls_accuracy_validation)
    na_f1_score_validation = np.array(ls_f1_score_validation)
    na_logloss_validation = np.array(ls_logloss_validation)

    if 'simple_average' in ensembles_t_use:
        print('# Ensemble simple_average ####################')

        ens_prbbs_train = np.average(na_prbbs_train, axis=0)
        ens_prbbs_validation = np.average(na_prbbs_validation, axis=0)
        ens_prbbs_test_n_live = np.average(na_prbbs_test_n_live, axis=0)

        print_logloss(ens_prbbs_train, ens_prbbs_validation)
        ens_prbbs_tournament = np.concatenate([ens_prbbs_validation, ens_prbbs_test_n_live], axis=0)
        ls_ens_prbbs_tournament.append(ens_prbbs_tournament)

    if 'accuracy_weighted' in ensembles_t_use:
        print('# Ensemble accuracy_weighted ####################')
        whts_accuracy_validation = na_accuracy_validation / na_accuracy_validation.sum()

        ens_prbbs_train = np.dot(whts_accuracy_validation, na_prbbs_train)
        ens_prbbs_validation = np.dot(whts_accuracy_validation, na_prbbs_validation)
        ens_prbbs_test_n_live = np.dot(whts_accuracy_validation, na_prbbs_test_n_live)

        print_logloss(ens_prbbs_train, ens_prbbs_validation)
        ens_prbbs_tournament = np.concatenate([ens_prbbs_validation, ens_prbbs_test_n_live], axis=0)
        ls_ens_prbbs_tournament.append(ens_prbbs_tournament)

    if 'f1_score_weighted' in ensembles_t_use:
        print('# Ensemble f1_score_weighted ####################')
        whts_f1_score_validation = na_f1_score_validation / na_f1_score_validation.sum()

        ens_prbbs_train = np.dot(whts_f1_score_validation, na_prbbs_train)
        ens_prbbs_validation = np.dot(whts_f1_score_validation, na_prbbs_validation)
        ens_prbbs_test_n_live = np.dot(whts_f1_score_validation, na_prbbs_test_n_live)

        print_logloss(ens_prbbs_train, ens_prbbs_validation)
        ens_prbbs_tournament = np.concatenate([ens_prbbs_validation, ens_prbbs_test_n_live], axis=0)
        ls_ens_prbbs_tournament.append(ens_prbbs_tournament)

    if 'logloss_weighted' in ensembles_t_use:
        print('# Ensemble logloss_weighted ####################')
        whts_logloss_validation = (1 / na_logloss_validation) / (1 / na_logloss_validation).sum()

        ens_prbbs_train = np.dot(whts_logloss_validation, na_prbbs_train)  # inverse to make sense
        ens_prbbs_validation = np.dot(whts_logloss_validation, na_prbbs_validation)
        ens_prbbs_test_n_live = np.dot(whts_logloss_validation, na_prbbs_test_n_live)

        print_logloss(ens_prbbs_train, ens_prbbs_validation)
        ens_prbbs_tournament = np.concatenate([ens_prbbs_validation, ens_prbbs_test_n_live], axis=0)
        ls_ens_prbbs_tournament.append(ens_prbbs_tournament)

    if 'softmax_accuracy_weighted' in ensembles_t_use:
        print('# Ensemble softmax_accuracy_weighted ####################')
        softmax_whts_accuracy_validation = softmax_whts(na_accuracy_validation)

        ens_prbbs_train = np.dot(softmax_whts_accuracy_validation, na_prbbs_train)
        ens_prbbs_validation = np.dot(softmax_whts_accuracy_validation, na_prbbs_validation)
        ens_prbbs_test_n_live = np.dot(softmax_whts_accuracy_validation, na_prbbs_test_n_live)

        print_logloss(ens_prbbs_train, ens_prbbs_validation)
        ens_prbbs_tournament = np.concatenate([ens_prbbs_validation, ens_prbbs_test_n_live], axis=0)
        ls_ens_prbbs_tournament.append(ens_prbbs_tournament)

    if 'softmax_f1_score_weighted' in ensembles_t_use:
        print('# Ensemble softmax_f1_score_weighted ####################')
        softmax_whts_f1_score_validation = softmax_whts(na_f1_score_validation)

        ens_prbbs_train = np.dot(softmax_whts_f1_score_validation, na_prbbs_train)
        ens_prbbs_validation = np.dot(softmax_whts_f1_score_validation, na_prbbs_validation)
        ens_prbbs_test_n_live = np.dot(softmax_whts_f1_score_validation, na_prbbs_test_n_live)

        print_logloss(ens_prbbs_train, ens_prbbs_validation)
        ens_prbbs_tournament = np.concatenate([ens_prbbs_validation, ens_prbbs_test_n_live], axis=0)
        ls_ens_prbbs_tournament.append(ens_prbbs_tournament)

    if 'softmax_logloss_weighted' in ensembles_t_use:
        print('# Ensemble softmax_logloss_weighted ####################')
        softmax_whts_logloss_validation = softmax_whts(1 / na_logloss_validation)  # inverse to make sense

        ens_prbbs_train = np.dot(softmax_whts_logloss_validation, na_prbbs_train)
        ens_prbbs_validation = np.dot(softmax_whts_logloss_validation, na_prbbs_validation)
        ens_prbbs_test_n_live = np.dot(softmax_whts_logloss_validation, na_prbbs_test_n_live)

        print_logloss(ens_prbbs_train, ens_prbbs_validation)
        ens_prbbs_tournament = np.concatenate([ens_prbbs_validation, ens_prbbs_test_n_live], axis=0)
        ls_ens_prbbs_tournament.append(ens_prbbs_tournament)

def print_logloss(ens_prbbs_train, ens_prbbs_validation):
    print('log loss of train :      %f' % log_loss(y_train, ens_prbbs_train))
    print('log loss of validation : %f' % log_loss(y_validation, ens_prbbs_validation))