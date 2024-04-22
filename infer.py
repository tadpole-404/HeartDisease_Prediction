import numpy as np
import json
import pandas as pd
import utlis_ as u
def infer(W_path,b_path,batch,features_preprocess,features_normalise,data_path):
#load model
    W=np.load(W_path)
    b=np.load(b_path)
    with open('mean.json') as file:
        mean=json.load(file)
    with open('std.json') as file:
        std=json.load(file)
    
    
    df=u.read_data(data_path,features_preprocess)
    
    df=u.normalize_with_mean_std(df,features_normalise,mean,std)
    ran=np.random.randint(0,281,(batch))
    # print(df.iloc[1])
    try:
        input=np.array([df.iloc[ran[i].item()] for i in range(batch)],np.float32)
    except:
        print(batch)
    input=np.transpose(input,(1,0))
    X=input[1:,:]
    gt=input[0,:]
    # print(X.shape)
    # print(input)
    print(W.shape)
    Y_prediction,_=u.forward_prop(W,X,b)
    # print(f'prediction of model is :{Y_prediction}')
    Y_prediction=np.round(Y_prediction)
    return(Y_prediction,gt)
# print(Y_prediction.shape)

# print(mean)
# print(std)
if __name__=='__main__':
    W_path="weights/W_weights epoch:90000 loss:0.45712472663833426.npy"
    b_path="weights/b_weights epoch:90000 loss:0.45712472663833426.npy"
    features_preprocess=["AgeCategory","Race","GenHealth","HeartDisease",'Smoking','AlcoholDrinking','Stroke','DiffWalking','Sex','AgeCategory','Race','Diabetic','PhysicalActivity','GenHealth','Asthma','KidneyDisease','SkinCancer']
    features_normalise=["BMI","SleepTime","PhysicalHealth","MentalHealth","AgeCategory","Race","GenHealth"]
    data_path='HeartDisease.csv'

    
    Y_prediction,gt=infer(W_path,b_path,2515,features_preprocess,features_normalise,data_path)
    print(f'prediction of model is :{Y_prediction}')

    print(f'ground truth is :{gt}')
    