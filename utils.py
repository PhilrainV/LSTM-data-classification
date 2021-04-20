
import numpy as np
import pandas as pd


def read_data(input_path, debug=True):
    """
    Read nasdaq stocks data.

    Args:
        input_path (str): directory to nasdaq dataset.

    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.

    """
    df = pd.read_csv(input_path, nrows=250 if debug else None)
    X = df.loc[:, [x for x in df.columns.tolist() if x != 'answered_correctly']].values
    y = np.array(df.answered_correctly)

    return X, y

# def calcAUC_byRocArea(labels,probs):
#     ###initialize
#     P = 0
#     N = 0
#     for i in labels:
#         if (i == 1):
#             P += 1
#         else:
#             N += 1
#     print(P)
#     print(N)
#     TP = 0
#     FP = 0
#     TPR_last = 0
#     FPR_last = 0
#     AUC = 0
#     pair = zip(probs, labels)
#     pair = sorted(pair, key=lambda x:x[0], reverse=True)
#     i = 0
#     while i < len(pair):
#         if (pair[i][1] == 1):
#             TP += 1
#         else:
#             FP += 1
#         ### maybe have the same probs
#         while (i + 1 < len(pair) and pair[i][0] == pair[i+1][0]):
#             i += 1
#             if (pair[i][1] == 1):
#                 TP += 1
#             else:
#                 FP += 1
#         TPR = TP / P
#         FPR = FP / N
#         AUC += 0.5 * (TPR + TPR_last) * (FPR - FPR_last)
#         TPR_last = TPR
#         FPR_last = FPR
#         i += 1
#     return AUC
def calcAUC_byRocArea(labels,prob):
    f = list(zip(prob,labels))
    rank = [values2 for values1,values2 in sorted(f,key=lambda x:x[0])]
    rankList = [i+1 for i in range(len(rank)) if rank[i]==1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if(labels[i]==1):
            posNum+=1
        else:
            negNum+=1
    AUC = (sum(rankList)- (posNum*(posNum+1))/2)/(posNum*negNum)
    print(AUC)
    return AUC

import csv
import codecs

def data_write_csv(file_name, data):#file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name,'w+','utf-8')#追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(data)
    print("保存文件成功，处理结束")