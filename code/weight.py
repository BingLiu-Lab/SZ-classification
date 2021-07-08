# The Optimal weight combination：
#          SVM    LR
# site1    0.08   0.92
# site2    1      0
# site3    0.47   0.53
# site4    1      0
# site5    0.03   0.97
# site6    0.3    0.7
# site7    0.64   0.36
# site8    0.98   0.02

import numpy as np
import joblib
import sys
import pandas as pd

weight_list = []
for i in range(8):
    sys.stdout = open('./log/feature_weight_no_abs.txt', 'w')
    # Loading the SVM and LR models
    model_dic1 = "./model/svm_smri_gene_site{}.model".format(i + 1)
    model_dic2 = "./model/lr_smri_gene_site{}.model".format(i + 1)
    clf_svm = joblib.load(model_dic1)
    clf_lr = joblib.load(model_dic2)
    weight_list.append(list(clf_svm.coef_[0]))
    weight_list.append(list(clf_lr.coef_[0]))

# Weight the feature weights according to the optimal weight combination
a = []
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []
list8 = []
for i in range(0, len(weight_list[0])):
    site1 = weight_list[0][i] * 0.08 + weight_list[1][i] * 0.92
    site2 = weight_list[2][i] * 1 + weight_list[3][i] * 0
    site3 = weight_list[4][i] * 0.47 + weight_list[5][i] * 0.53
    site4 = weight_list[6][i] * 1 + weight_list[7][i] * 0
    site5 = weight_list[8][i] * 0.03 + weight_list[9][i] * 0.97
    site6 = weight_list[10][i] * 0.3 + weight_list[11][i] * 0.7
    site7 = weight_list[12][i] * 0.64 + weight_list[13][i] * 0.36
    site8 = weight_list[14][i] * 0.98 + weight_list[15][i] * 0.02
    ave = (site1 + site2 + site3 + site4 + site5 + site6 + site7 + site8)/8
    a.append(ave)
    list1.append(site1)
    list2.append(site2)
    list3.append(site3)
    list4.append(site4)
    list5.append(site5)
    list6.append(site6)
    list7.append(site7)
    list8.append(site8)

# a = list(map(abs, a))
print('feature weight:\n', a)
print('feature weight sorted from large to small:\n', sorted(a, reverse=True))
print('Index:\n', np.argsort(a)[::-1])  # Sort the weights from large to small and return the corresponding index value

dataframe = pd.DataFrame({'site1': list1, 'site2': list2, 'site3': list3, 'site4': list4, 'site5': list5,
                          'site6': list6, 'site7': list7, 'site8': list8})
dataframe.to_csv(r"./log/site_weight.csv", sep=',')
