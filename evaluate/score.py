import numpy as np
def f1(label,true_label,classifications):
    # TP:预测答案正确
    # FP:错将其他类预测为本类
    # FN:本类标签预测为其他类标

    f1score=[]
    epsilon = 1e-7
    for i in range(classifications):
        TP,FP,FN=0,0,0
        for k in range(len(label)):
            if((label[k]==i)and(true_label[k]==i)):
                TP+=1
            elif((label[k]==i)and(true_label[k]!=i)):
                FP+=1
            elif((true_label[k]==i)and(label[k]!=i)):
                FN+=1
            else:
                pass
        P=TP/(TP+FP+epsilon)
        R=TP/(TP+FN+epsilon)
        score=2*P*R/(P+R+epsilon)
        f1score.append(score)
    return np.mean(f1score)

def accuracy(label,label_true):
   num=0
   for i,j in zip(label,label_true):
       if i==j:
           num+=1
   return  num/len(label_true)


