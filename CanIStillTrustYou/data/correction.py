import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

random.seed(0)
holdpct = 0.1
holdmin = 50


data2 = pd.read_csv('./raw/SouthGermanCredit.asc', sep=' ')
data2.columns = ['status',
 'duration',
 'credit_history',
 'purpose',
 'amount',
 'savings',
 'employment_duration',
 'installment_rate',
 'personal_status_sex',
 'other_debtors',
 'present_residence',
 'property',
 'age',
 'other_installment_plans',
 'housing',
 'number_credits',
 'job',
 'people_liable',
 'telephone',
 'foreign_worker',
 'credit_risk']
data1 = pd.read_csv('./raw/german.data', sep=' ', names=data2.columns)
data1.credit_risk = data1.credit_risk % 2

# matching encoding from data1 to data2:

def transform_status(s):
    key = {
         'A14': 1, # no checking account                       
         'A11': 2, # ... < 0 DM                                
         'A12': 3, # 0<= ... < 200 DM                          
         'A13': 4 # ... >= 200 DM / salary for at least 1 year
    }
    return key[s]

def transform_credit_history(s):
    key = {
         'A33': 0, # delay in paying off in the past 
         'A34': 1, # critical account/other credits elsewhere 
         'A30': 2, #no credits taken/all credits paid back duly
         'A32': 3, #existing credits paid back duly till now 
         'A31': 4, #all credits at this bank paid back duly
    }
    return key[s]

def transform_purpose(s):
    key = {
         'A410': 0, #others 
         'A40': 1, #car (new) 
         'A41': 2, #car (used) 
         'A42': 3, #furniture/equipment
         'A43': 4, #radio/television 
         'A44': 5, #domestic appliances
         'A45': 6, #repairs 
         'A46': 7, #education 
         'A47': 8, #vacation 
         'A48': 9, #retraining 
         'A49': 10 #business
    }
    return key[s]

def transform_savings(s):
    key = {
         'A65': 1, #unknown/no savings account
         'A61': 2, #... < 100 DM 
         'A62': 3, #100 <= ... < 500 DM 
         'A63': 4, #500 <= ... < 1000 DM 
         'A64': 5 #... >= 1000 DM
    }
    return key[s]

def transform_employment_duration(s):
    key = {
         'A71': 1, #unemployed 
         'A72': 2, #< 1 yr 
         'A73': 3, #1 <= ... < 4 yrs
         'A74': 4, #4 <= ... < 7 yrs
         'A75': 5 #>= 7 yrs
    }
    return key[s]

def transform_installment_rate(s):
    return s
#     key = {
#          0 
#          'A': 1, #>= 35 
#          'A': 2, #25 <= ... < 35
#          'A': 3, #20 <= ... < 25
#          'A': 4 #< 20
#     }
#     return key[s]

def transform_personal_status_sex(s):
    key = {
         'A91': 1, #male, #divorced/separated 
         'A92': 2, #female, #non-single or male, #single
         'A94': 3, #male, #married/widowed 
         'A95': 4, #female, #single
         'A93': 5, #male, single
    }
    return key[s]

def transform_other_debtors(s):
    key = {
         'A101': 1, #none 
         'A102': 2, #co-applicant
         'A103': 3 #guarantor
    }
    return key[s]

def transform_present_residence(s):
    return s
    key = {
         'A': 1, #< 1 yr 
         'A': 2, #1 <= ... < 4 yrs
         'A': 3, #4 <= ... < 7 yrs
         'A': 4 #>= 7 yrs
    }
    return key[s]

def transform_property(s):
    key = {
         'A124': 1, #unknown / no property 
         'A123': 2, #car or other 
         'A122': 3, #building soc. savings agr./life insurance
         'A121': 4 #real estate
    }
    return key[s]

def transform_other_installment_plans(s):
    key = {
         'A141': 1, #bank 
         'A142': 2, #stores
         'A143': 3 #none
    }
    return key[s]

def transform_housing(s):
    key = {
         'A153': 1, #for free
         'A151': 2, #rent 
         'A152': 3 #own
    }
    return key[s]

def transform_number_credits(s):
    return s
#     key = {
#          0 
#          'A': 1, #1 
#          'A': 2, #2-3 
#          'A': 3, #4-5 
#          'A': 4 #>= 6
#     }
#     return key[s]

def transform_job(s):
    key = {
         'A171': 1, #unemployed/unskilled - non-resident 
         'A172': 2, #unskilled - resident 
         'A173': 3, #skilled employee/official 
         'A174': 4 #manager/self-empl./highly qualif. employee
    }
    return key[s]

def transform_people_liable(s):
    return s
#     key = {
#          0 
#          'A': 1, #3 or more
#          'A': 2 #0 to 2
#     }
#     return key[s]

def transform_telephone(s):
    key = {
         'A191': 1, #no 
         'A192': 2 #yes (under customer name)
    }
    return key[s]

def transform_foreign_worker(s):
    key = {
         'A201': 1, #yes
         'A202': 2 #no
    }
    return key[s]


data1.status = data1.status.apply(lambda x: transform_status(x))
data1.purpose = data1.purpose.apply(lambda x: transform_purpose(x))
data1.savings = data1.savings.apply(lambda x: transform_savings(x))
data1.employment_duration = data1.employment_duration.apply(lambda x: transform_employment_duration(x))
data1.installment_rate = data1.installment_rate.apply(lambda x: transform_installment_rate(x))
data1.personal_status_sex = data1.personal_status_sex.apply(lambda x: transform_personal_status_sex(x))
data1.other_debtors = data1.other_debtors.apply(lambda x: transform_other_debtors(x))
data1.present_residence = data1.present_residence.apply(lambda x: transform_present_residence(x))
data1.property = data1.property.apply(lambda x: transform_property(x))
data1.other_installment_plans = data1.other_installment_plans.apply(lambda x: transform_other_installment_plans(x))
data1.housing = data1.housing.apply(lambda x: transform_housing(x))
data1.number_credits = data1.number_credits.apply(lambda x: transform_number_credits(x))
data1.job = data1.job.apply(lambda x: transform_job(x))
data1.people_liable = data1.people_liable.apply(lambda x: transform_people_liable(x))
data1.telephone = data1.telephone.apply(lambda x: transform_telephone(x))
data1.foreign_worker = data1.foreign_worker.apply(lambda x: transform_foreign_worker(x))
data1.credit_history = data1.credit_history.apply(lambda x: transform_credit_history(x))


y1 = data1['credit_risk']
X1 = data1.drop(columns=['credit_risk'])
y2 = data2['credit_risk']
X2 = data2.drop(columns=['credit_risk'])

hold_out_size = int(max(holdpct * len(data1), holdmin))
res = train_test_split(X1, y1, test_size=hold_out_size, random_state=0)

res[0].to_csv('./cleaned/correction/D1_train_X.csv')
res[1].to_csv('./cleaned/correction/D1_test_X.csv')
res[2].to_csv('./cleaned/correction/D1_train_y.csv')
res[3].to_csv('./cleaned/correction/D1_test_y.csv')

hold_out_size = int(max(holdpct * len(data2), holdmin))
res = train_test_split(X2, y2, test_size=hold_out_size, random_state=0)

res[0].to_csv('./cleaned/correction/D2_train_X.csv')
res[1].to_csv('./cleaned/correction/D2_test_X.csv')
res[2].to_csv('./cleaned/correction/D2_train_y.csv')
res[3].to_csv('./cleaned/correction/D2_test_y.csv')

