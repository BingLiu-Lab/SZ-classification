import numpy as np
import joblib
import pandas as pd
import pickle
import sklearn.preprocessing as prep
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

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
    # Generating 100 weight combinations
    list = [0]
    flag = 0
    for num in range(100):
        flag = flag + 0.01
        list.append(round(flag, 2))

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
        x_test = scaler.transform(x_test)

        # Loading the SVM and LR models
        model_dic1 = "./model/svm_smri_gene_site{}.model".format(i + 1)
        model_dic2 = "./model/lr_smri_gene_site{}.model".format(i + 1)
        clf_svm = joblib.load(model_dic1)
        clf_lr = joblib.load(model_dic2)

        print('Start predicting...')
        y_pred1 = clf_svm.predict(x_test)  # Prediction labels
        y_pred2 = clf_lr.predict(x_test)
        final_pred = y_pred1  # Initialize the prediction labels of ensemble learning
        pred_proba1 = clf_svm.predict_proba(x_test)  # Prediction probability
        pred_proba2 = clf_lr.predict_proba(x_test)

        # Ensemble learning: weighted average and find the optimal weight combination
        list1 = []
        for j in range(101):
            final_pred_pro = pred_proba1 * list[j] + pred_proba2 * list[100 - j]  # Weighted average prediction probability
            for k in range(len(final_pred_pro)):
                if final_pred_pro[k][0] > final_pred_pro[k][1]:
                    final_pred[k] = 0
                else:
                    final_pred[k] = 1
            list1.append(accuracy_score(y_test, final_pred))

        index = np.argsort(list1)  # Sort the accuracy from small to large and return the corresponding index value
        print(list1)
        print('Accuracy after sorting:', sorted(list1))  # print accuracy sorted from small to large
        print('Index:', index)
        print('Maximum accuracy:{:.4f}'.format(max(list1)))

        # Printing evaluation index: Accuracy, Sensitivity, Specificity, AUC, Confusion Matrix
        best_index = index[-1]  # Index value corresponding to maximum accuracy
        best_pred_pro = pred_proba1 * list[best_index] + pred_proba2 * list[100 - best_index]  # Best predicted probability
        for m in range(len(best_pred_pro)):
            if best_pred_pro[m][0] > best_pred_pro[m][1]:
                final_pred[m] = 0
            else:
                final_pred[m] = 1  # Best predicted label
        cm = confusion_matrix(y_test, final_pred)
        TPR = float(cm[0][0]) / np.sum(cm[0])
        TNR = float(cm[1][1]) / np.sum(cm[1])
        ACC = accuracy_score(y_test, final_pred)
        print('-------------------------------')
        print("Test set Accuracy:{:.4f}".format(ACC))
        print("Sensitivity:{:.4f}".format(TPR))
        print("Specificity:{:.4f}".format(TNR))
        AUC = roc_auc_score(y_test, best_pred_pro[:, 1])
        print("AUC:{:.4f}".format(AUC))
        print('-------------------------------')
        print('Confusion Matrix:\n', cm)

        # Calculating fpr and tpr
        y_score1 = clf_svm.decision_function(x_test)
        y_score2 = clf_lr.decision_function(x_test)
        best_y_score = y_score1 * list[best_index] + y_score2 * list[100 - best_index]
        fpr, tpr, threshold = roc_curve(y_test, best_y_score)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    f = open('./variable/en_smri_gene_fpr_list.pckl', 'wb')
    pickle.dump(fpr_list, f)
    f.close()
    f = open('./variable/en_smri_gene_tpr_list.pckl', 'wb')
    pickle.dump(tpr_list, f)
    f.close()