import numpy as np
import pandas as pd
import pickle
import sklearn.preprocessing as prep
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import joblib


if __name__ == '__main__':
    # Reading the path of all the data
    test_list = []
    train_list = []
    fpr_list = []
    tpr_list = []
    with open(r"..\data\smri_gene\test_list.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            test_list.append(line)
    with open(r"..\data\smri_gene\train_list.txt", "r") as f:
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
        clf = svm.SVC(kernel='linear', probability=True)
        param_grid = {'C': [1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1, 10]}
        grid_search = GridSearchCV(clf, param_grid, cv=10, scoring="neg_log_loss", iid=False)
        grid_search.fit(x_train, y_train.ravel())
        model_dic = "./model/svm_smri_gene_site{}.model".format(i + 1)
        joblib.dump(grid_search.best_estimator_, model_dic)

        print('Start predicting...')
        print("Best parameters:{}".format(grid_search.best_params_))
        print("Best score on train set:{:.4f}".format(grid_search.best_score_))
        print("Test set score:{:.4f}".format(grid_search.score(x_test, y_test.ravel())))

        # Printing evaluation index: Accuracy, Sensitivity, Specificity, AUC, Confusion Matrix
        y_pred = grid_search.best_estimator_.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        TPR = float(cm[0][0]) / np.sum(cm[0])
        TNR = float(cm[1][1]) / np.sum(cm[1])
        ACC = accuracy_score(y_test, y_pred)
        print('-------------------------------')
        print("Test set Accuracy:{:.4f}".format(ACC))
        print("Sensitivity:{:.4f}".format(TPR))
        print("Specificity:{:.4f}".format(TNR))
        y_pred_proba = grid_search.best_estimator_.predict_proba(x_test)
        AUC = roc_auc_score(y_test, y_pred_proba[:, 1])
        print("AUC:{:.4f}".format(AUC))
        print('-------------------------------')
        print('Confusion Matrix:\n', cm)

        # Calculating fpr and tpr
        y_score = grid_search.best_estimator_.decision_function(x_test)
        fpr, tpr, threshold = roc_curve(y_test, y_score)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    f = open('./variable/svm_smri_gene_fpr_list.pckl', 'wb')
    pickle.dump(fpr_list, f)
    f.close()
    f = open('./variable/svm_smri_gene_tpr_list.pckl', 'wb')
    pickle.dump(tpr_list, f)
    f.close()













