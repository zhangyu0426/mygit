import numpy as np

def get_numbers():
    with open("/home/ff/mywork/log/prediction/prediction_epoch_25.out") as input_file:
        for line in input_file:
            linestr = line.strip()
            for number in linestr.split():
                yield float(number)
predlist=[]
pred0=[]
pred1=[]
for pernumber in get_numbers():
    predlist.append(pernumber)
print len(predlist)
for i in range(len(predlist)):
    if i%2==0:
        pred0.append(predlist[i])
    else:
        pred1.append(predlist[i])
print("the length of pred0: ",len(pred0))


pospred=[]
for j in range(len(pred0)):
    if pred0[j]>pred1[j]:
        pospred.append(0)
    else:
        pospred.append(1)
#print pospred

Y_pred = np.load('/home/ff/mywork/log/prediction/xinfei1500_299random_Y_val.npy')
y_pred = Y_pred.tolist()
print("the length of pred0: ",len(y_pred))
print y_pred

z = np.zeros((2,2),dtype=np.float)
print("before 2*2 mat: ",z)

print pospred
if len(pospred)!=len(y_pred):
    print('error')
else:
    for k in range(len(pred0)):
        if pospred[k]==0 and y_pred[k]==0:
            z[1][1] += 1
        elif pospred[k] == 0 and y_pred[k] == 1:
            z[0][1] += 1
        elif (pospred[k]==1) and (y_pred[k]==0):
            z[1][0]=z[1][0]+1
        else:
            z[0][0] += 1

print("after 2*2 mat: ",z)
p_recall =round(z[0][0]/(z[0][0]+z[0][1])*100,3)
n_recall =round(z[1][1]/(z[1][1]+z[1][0])*100,3)
print ("p_recall: ",p_recall)
print ("n_recall: ",n_recall)
    

        
        
            

