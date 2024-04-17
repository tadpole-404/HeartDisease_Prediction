import utlis_ as u
import numpy as np
from tqdm import tqdm
import json
#hyperparameters
alpha=0.000001
epoch=10000
batch=1000
loss_stored=[]
#reading data
features_preprocess=["AgeCategory","Race","GenHealth","HeartDisease",'Smoking','AlcoholDrinking','Stroke','DiffWalking','Sex','AgeCategory','Race','Diabetic','PhysicalActivity','GenHealth','Asthma','KidneyDisease','SkinCancer']
df=u.read_data('archive(4)/2020/heart_2020_cleaned.csv',features_preprocess)
features_normalise=["BMI","SleepTime","PhysicalHealth","MentalHealth","AgeCategory","Race","GenHealth"]
df,mean,std=u.normalize(df,features_normalise)
with open('mean.json', 'w') as json_file:
    json.dump(mean, json_file)
    print('saved mean!!')
with open('std.json', 'w') as json_file:
    json.dump(std, json_file)
    print('saved std')


data=[]
for key in df.keys():
    if key=='HeartDisease':
        Y=np.array(df[key])
    else:
        data.append(df[key])
    
X=np.array(data,np.float32)

Y=np.reshape(Y,(1,Y.shape[0]))
# print(Y.shape)
# print(X.shape)

#random intialisation of weights
W=np.random.randn(1,17)
b=np.random.randn(1)
# print(W.shape)
# print(b.shape)
for i in tqdm(range(epoch),unit='iteration'):
    loss=0
    for j in range(0,X.shape[1],batch):
        # print(j)
        # print(type(j))
        Y_hat=u.forward_prop(W,X[:,j:j+batch],b)
        # print(Y[:,j:j+batch].shape)
        # print(Y_hat.shape)
        # print(Y_hat.shape)
        dW,db=u.gradients(X[:,j:j+batch],Y[:,j:j+batch],Y_hat,j,batch)
        W,b=u.update(dW,db,W,b,alpha)
        # print('done')
        if i%50==0:
            loss+=u.loss(Y[:,j:j+batch],Y_hat,batch)
            # print(loss)
            # print(Y[:,j:j+batch].shape)
            # print(Y_hat[:,j:j+batch].shape)
    if i%50==0:
        print(f'loss is :{loss}')
        loss_stored.append(loss)
        # alpha=alpha/2
        #weight decay
    if i%500==0:
        print('saved weights')
        np.save(f'weights/W_weights epoch:{i} loss:{loss}.npy',W)
        np.save(f'weights/b_weights epoch:{i} loss:{loss}.npy',b)
        
np.save('loss.npy',np.array(loss_stored))# saved after every 50 epoch