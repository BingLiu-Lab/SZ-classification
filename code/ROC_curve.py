import numpy as np
from numpy import interp
import matplotlib.pyplot as plt
import pickle


# Macro average
def get_xy(in_x, in_y):
    n = len(in_x)
    all_fpr = np.unique(np.concatenate([in_x[i] for i in range(n)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n):
        mean_tpr += interp(all_fpr, in_x[i], in_y[i])
    mean_tpr /= n
    x = all_fpr
    y = mean_tpr
    return x, y


f = open('./variable/en_gene_fpr_list.pckl', 'rb')
x_gene = pickle.load(f)
f.close()
f = open('./variable/en_gene_tpr_list.pckl', 'rb')
y_gene = pickle.load(f)
f.close()

f = open('./variable/en_smri_fpr_list.pckl', 'rb')
x_smri = pickle.load(f)
f.close()
f = open('./variable/en_smri_tpr_list.pckl', 'rb')
y_smri = pickle.load(f)
f.close()

f = open('./variable/en_smri_gene_fpr_list.pckl', 'rb')
x_smri_gene = pickle.load(f)
f.close()
f = open('./variable/en_smri_gene_tpr_list.pckl', 'rb')
y_smri_gene = pickle.load(f)
f.close()

x_gene, y_gene = get_xy(x_gene, y_gene)
x_smri, y_smri = get_xy(x_smri, y_smri)
x_smri_gene, y_smri_gene = get_xy(x_smri_gene, y_smri_gene)

plt.plot(x_gene, y_gene, color="black", lw=1, label="PGRS(AUC=0.6190)")
plt.plot(x_smri, y_smri, color="blue", lw=1, label="sMRI(AUC=0.7687)")
plt.plot(x_smri_gene, y_smri_gene, color="red", lw=2, label="sMRI&PGRS(AUC=0.7742)")
plt.plot([0, 1], [0, 1], color='dimgray', lw=2, linestyle='--', label="Random")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title("ROC curves for ensemble learning models")
plt.savefig("./ROC_curve/ensemble_learning.jpg")
plt.show()