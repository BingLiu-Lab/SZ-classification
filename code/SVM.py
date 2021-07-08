import numpy as np
import pandas as pd
import pickle
import sklearn.preprocessing as prep
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, f1_score, precision_score, \
    recall_score, balanced_accuracy_score
import joblib
import sys


if __name__ == '__main__':
    feature = 'gene'
    sys.stdout = open('./log/' + feature + '_svm.txt', 'w')
    # Reading the path of all the data
    test_list = []
    train_list = []
    fpr_list = []
    tpr_list = []
    with open('../data/data_for_CV/' + feature + '/test_list.txt', "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            test_list.append(line)
    with open('../data/data_for_CV/' + feature + '/train_list.txt', "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            train_list.append(line)

    for i in range(8):
        print("*********************site{}**********************".format(i+1))
        print('Preparing datasets...')
        path1 = test_list[i]
        raw_data1 = pd.read_csv(path1, header=None)
        data_test = raw_data1.values

        path2 = train_list[i]
        raw_data2 = pd.read_csv(path2, header=None)
        data_train = raw_data2.values

        print('Data normalization...')
        y_train, x_train = np.split(data_train, (1,), axis=1)
        y_test, x_test = np.split(data_test, (1,), axis=1)
        scaler = prep.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        print('Start training...')
        clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced')
        param_grid = {'C': [1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1, 10]}
        grid_search = GridSearchCV(clf, param_grid, cv=10, scoring="neg_log_loss", iid=False)
        grid_search.fit(x_train, y_train.ravel())
        model_dic = './model/svm_' + feature + '_site{}.model'.format(i + 1)
        joblib.dump(grid_search.best_estimator_, model_dic)

        print('Start predicting...')
        print("Best parameters:{}".format(grid_search.best_params_))
        print("Best score on train set:{:.4f}".format(grid_search.best_score_))
        print("Test set score:{:.4f}".format(grid_search.score(x_test, y_test.ravel())))

        # Printing evaluation index
        y_pred = grid_search.best_estimator_.predict(x_test)
        ACC = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        TPR = float(cm[0][0]) / np.sum(cm[0])
        TNR = float(cm[1][1]) / np.sum(cm[1])
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        y_pred_proba = grid_search.best_estimator_.predict_proba(x_test)
        AUC = roc_auc_score(y_test, y_pred_proba[:, 1])
        print('-------------------------------')
        print("Accuracy:{:.4f}".format(ACC))
        print("Sensitivity:{:.4f}".format(TPR))
        print("Specificity:{:.4f}".format(TNR))
        print("Balanced accuracy:{:.4f}".format(balanced_accuracy))
        print("Precision:{:.4f}".format(precision))
        print("Recall:{:.4f}".format(recall))
        print("f1 score:{:.4f}".format(f1))
        print("AUC:{:.4f}".format(AUC))
        print('-------------------------------')
        print('Confusion Matrix:\n', cm)

        # Calculating fpr and tpr
        y_score = grid_search.best_estimator_.decision_function(x_test)
        fpr, tpr, threshold = roc_curve(y_test, y_score)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    f = open('./variable/svm_' + feature + '_fpr_list.pckl', 'wb')
    pickle.dump(fpr_list, f)
    f.close()
    f = open('./variable/svm_' + feature + '_tpr_list.pckl', 'wb')
    pickle.dump(tpr_list, f)
    f.close()

