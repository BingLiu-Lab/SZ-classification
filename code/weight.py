# The Optimal weight combination：
#          SVM    LR
# site1    0.11   0.89
# site2    1      0
# site3    0.42   0.58
# site4    1      0
# site5    0.03   0.97
# site6    0.71   0.29
# site7    0.5    0.5
# site8    0.99   0.01

import numpy as np
import joblib

weight_list = []
for i in range(8):
    # Loading the SVM and LR models
    model_dic1 = "./model/svm_smri_gene_site{}.model".format(i + 1)
    model_dic2 = "./model/lr_smri_gene_site{}.model".format(i + 1)
    clf_svm = joblib.load(model_dic1)
    clf_lr = joblib.load(model_dic2)
    weight_list.append(list(clf_svm.coef_[0]))
    weight_list.append(list(clf_lr.coef_[0]))

# Weight the feature weights according to the optimal weight combination
a = []
for i in range(0, len(weight_list[0])):
    ave = (weight_list[0][i]*0.11 + weight_list[1][i]*0.89
            + weight_list[2][i]*1 + weight_list[3][i]*0
            + weight_list[4][i]*0.42 + weight_list[5][i]*0.58
            + weight_list[6][i]*1 + weight_list[7][i]*0
            + weight_list[8][i]*0.03 + weight_list[9][i]*0.97
            + weight_list[10][i]*0.71 + weight_list[11][i]*0.29
            + weight_list[12][i]*0.5 + weight_list[13][i]*0.5
            + weight_list[14][i]*0.99 + weight_list[15][i]*0.01)/8
    a.append(ave)

a = list(map(abs, a))
print('feature weight:\n', a)
print('feature weight sorted from large to small:\n', sorted(a, reverse=True))
print('Index:\n', np.argsort(a)[::-1])  # Sort the weights from small to large and return the corresponding index value


