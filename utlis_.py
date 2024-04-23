import pandas as pd
import numpy as np

def read_data(path,features,flag=1):
    ''' 
    reads the csv file and returns the data as pandas datafield
    '''
    data=pd.read_csv(path)
    df=pd.DataFrame(data)

    return preprocess_input(df,features)
   
   
def preprocess_input(df,features):
    ''' 
    this function is meant to preprocess csv , converting strings to numbers 
    
    params :
    df :datafield 
    features: list of features to be preprocessed
    
    returns 
    processed datafield
    
    '''
    for feature in features:
        values=df[feature].unique()
        for i,value in enumerate(values):
            df[feature]=[i if element==value else element for element in df[feature]]
    return df

def normalize(df,features):
    ''' 
    used to normalize inputs by making mean=0 and standard deviation=0
    
    params
    df: input as pandas datafield
    features: list of keys of features to be normalized
    
    returns
    df: datafield after normalisation 
    mean: dictionary containing mean of features normalised
    std: dictionary containing standard deviation of features normalised    
    
    '''
    
    mean={feature:0. for feature in features}
    std={feature:0. for feature in features}
    for feature in features:
        mean[feature]=np.mean(df[feature])
        std[feature]=np.std(df[feature])
        df[feature]=(df[feature]-mean[feature])/std[feature]
    
    return df,mean,std

def normalize_with_mean_std(df,features,mean,std):
    '''
    this function normalises the given features to given mean and standard deviation
    
    params
    df: input as pandas datafield
    features: list of keys of features to be normalized
    mean: dictionary containing mean of features normalised
    std: dictionary containing standard deviation of features normalised    
    
    returns
    df: datafield after normalisation 
    
    '''
    for feature in features:
        mean[feature]=np.mean(df[feature])
        std[feature]=np.std(df[feature])
        df[feature]=(df[feature]-mean[feature])/std[feature]
    
    return df
    
def forward_prop(W,X,b):
    return 1/(1+np.exp(-(np.dot(W,X)+b))),np.exp(-(np.dot(W,X)+b))

# binary cross entropy loss
def loss(y,y_hat,batch):

    return np.sum(-y*np.log(y_hat)-(1-y)*np.log(1-y_hat))/batch

def gradients(X,Y,Y_hat,exp):
    dW=[]
    for i in range(X.shape[0]):
        grad=(Y_hat-Y)*X/X.shape[1]
        grad=np.sum(grad)
        dW.append(grad)
    db=(Y_hat-Y)/X.shape[1]
    db=np.sum(db)
    
    return dW,db

def update(dW,db,W,b,alpha):
    for i in range(W.shape[0]):
        W[i]=W[i]-alpha*dW[i]
    b=b-alpha*db
    return W,b
    

if __name__=='__main__':
    
    features_preprocess=["AgeCategory","Race","GenHealth","HeartDisease",'Smoking','AlcoholDrinking','Stroke','DiffWalking','Sex','AgeCategory','Race','Diabetic','PhysicalActivity','GenHealth','Asthma','KidneyDisease','SkinCancer']
    df=read_data('archive(4)/2020/heart_2020_cleaned.csv',features_preprocess)
    features=["BMI","SleepTime","PhysicalHealth","MentalHealth","AgeCategory","Race","GenHealth"]
    print(normalize(df,features))
    for col in df.columns:
        print(df[col])