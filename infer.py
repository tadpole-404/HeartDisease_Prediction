import numpy as np
import json
import pandas as pd
import utlis_ as u
#load model
W_path="weights/W_weights epoch:90000 loss:0.45712472663833426.npy"
b_path="weights/b_weights epoch:90000 loss:0.45712472663833426.npy"
W=np.load(W_path)
b=np.load(b_path)


batch=50

with open('mean.json') as file:
    mean=json.load(file)
with open('std.json') as file:
    std=json.load(file)
data_path='archive(4)/2020/heart_2020_cleaned.csv'
features_preprocess=["AgeCategory","Race","GenHealth","HeartDisease",'Smoking','AlcoholDrinking','Stroke','DiffWalking','Sex','AgeCategory','Race','Diabetic','PhysicalActivity','GenHealth','Asthma','KidneyDisease','SkinCancer']
df=u.read_data(data_path,features_preprocess)
features_normalise=["BMI","SleepTime","PhysicalHealth","MentalHealth","AgeCategory","Race","GenHealth"]
df=u.normalize_with_mean_std(df,features_normalise,mean,std)
ran=np.random.randint(0,10000,(batch))
# print(df.iloc[1])
input=np.array([df.iloc[ran[i].item()] for i in range(batch)],np.float32)
input=np.transpose(input,(1,0))
X=input[1:,:]
gt=input[0,:]
# print(X.shape)
# print(input)
Y_prediction=u.forward_prop(W,X,b)
# print(f'prediction of model is :{Y_prediction}')
Y_prediction=np.round(Y_prediction)
# print(Y_prediction.shape)
print(f'prediction of model is :{Y_prediction}')

print(f'ground truth is :{gt}')

# print(mean)
# print(std)