import utlis_ as u
import numpy as np
from tqdm import tqdm
import json
features_preprocess=['island','species']
# features_preprocess=["AgeCategory","Race","GenHealth","HeartDisease",'Smoking','AlcoholDrinking','Stroke','DiffWalking','Sex','AgeCategory','Race','Diabetic','PhysicalActivity','GenHealth','Asthma','KidneyDisease','SkinCancer']
df=u.read_data('penguins_binary_classification.csv',features_preprocess)
# features_normalise=["BMI","SleepTime","PhysicalHealth","MentalHealth","AgeCategory","Race","GenHealth"]
features_normalise=['body_mass_g','flipper_length_mm','bill_depth_mm','bill_length_mm']
# df,mean,std=u.normalize(df,features_normalise)

# with open('mean.json', 'w') as json_file:
#     json.dump(mean, json_file)
#     print('saved mean!!')
# with open('std.json', 'w') as json_file:
#     json.dump(std, json_file)
#     print('saved std')


data=[]
for key in df.keys():
    if key=='species':
        Y=np.array(df[key])
    else:
        data.append(df[key])
    
X=np.array(data,np.float32)

Y=np.reshape(Y,(1,Y.shape[0]))
# print(Y.shape)
# print(X.shape)
print(df)
#random intialisation of weights
np.random.seed(42)
W=np.random.randn(1,5)
b=np.random.randn(1)
pred=u.forward_prop(W,X,b)
print(np.unique(pred))
