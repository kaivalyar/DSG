import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

random.seed(0)
holdpct = 0.1
holdmin = 50

data=pd.read_csv('./raw/xAPI-Edu-Data.csv')
def cleanstage(x):
    if x=='L':
        return 0
    elif x=='M':
        return 1
    else:
        return 1 # converting to binary class prediction
data.StageID=data.StageID.apply(lambda y:cleanstage(y))
data['gender']=data.gender.apply(lambda x: 0 if x=='M' else 1)
data.ParentschoolSatisfaction=data.ParentschoolSatisfaction.apply(lambda x: 1 if x=='Good' else 0)
data.StudentAbsenceDays=data.StudentAbsenceDays.apply(lambda x: 0 if x=="Under-7" else 1)
data.Semester=data.Semester.apply(lambda x: 0 if x=='F' else 1)# 0=F - 1=S
data.GradeID=data.GradeID.apply(lambda x: int(x[2:]))
data.Relation=data.Relation.apply(lambda x: 0 if x=='Father' else 1)
data.ParentAnsweringSurvey=data.ParentAnsweringSurvey.apply(lambda x: 1 if x=='Yes' else 0)
# print(np.unique(data.Class))
data.Class=data.Class.apply(lambda x:cleanstage(x))
# print(np.unique(data.Class))
le = LabelEncoder().fit(data.Topic)
data.Topic = le.transform(data.Topic)
data = data.drop(columns=['PlaceofBirth', 'SectionID', 'StageID'])

data1 = data[data['NationalITy'] == 'KW']
data1 = data1.drop(columns=['NationalITy'])
y1 = data1['Class']
X1 = data1.drop(columns=['Class'])

data2 = data[data['NationalITy'] == 'Jordan']
data2 = data2.drop(columns=['NationalITy'])
y2 = data2['Class']
X2 = data2.drop(columns=['Class'])



hold_out_size = int(max(holdpct * len(data1), holdmin))
res = train_test_split(X1, y1, test_size=hold_out_size, random_state=0)


res[0].to_csv('./cleaned/geospatial/D1_train_X.csv')
res[1].to_csv('./cleaned/geospatial/D1_test_X.csv')
res[2].to_csv('./cleaned/geospatial/D1_train_y.csv')
res[3].to_csv('./cleaned/geospatial/D1_test_y.csv')



hold_out_size = int(max(holdpct * len(data2), holdmin))
res = train_test_split(X2, y2, test_size=hold_out_size, random_state=0)



res[0].to_csv('./cleaned/geospatial/D2_train_X.csv')
res[1].to_csv('./cleaned/geospatial/D2_test_X.csv')
res[2].to_csv('./cleaned/geospatial/D2_train_y.csv')
res[3].to_csv('./cleaned/geospatial/D2_test_y.csv')


