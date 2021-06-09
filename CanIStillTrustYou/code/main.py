import argparse
import numpy as np
import pandas as pd
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm as tqdm
import random
import matplotlib.pyplot as plt
# from itertools import product
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from recourse.builder import RecourseBuilder
from recourse.builder import ActionSet
from lime.lime_tabular import LimeTabularExplainer
from scipy import optimize

class SCM:
    def __init__(self):
        pass
    
    def act(self, data, actions):
        return data.flatten() + actions.flatten()

class GermanSCM(SCM):
    def __init__(self, X):
        self.f3 = LinearRegression()
        self.f4 = LinearRegression()
        self.f3.fit(X[['personal_status_sex', 'age']], X['amount'])
        self.f4.fit(X[['amount']], X['duration'])

    def act(self, data, actions):
        data = data.flatten()
#         u1 = data['personal_status_sex']
#         u2 = data['age']
#         u3 = data['amount'] - self.f3.predict([[data['personal_status_sex'], data['age']]])[0]
#         u4 = data['duration'] - self.f4.predict([[data['amount']]])[0]
        u1 = data[0]
        u2 = data[1]
        u3 = data[2] - self.f3.predict([data[:2]])[0]
        u4 = data[3] - self.f4.predict([[data[2]]])[0]
        
        actions = actions.flatten()
#         if actions[1] < 0:
#             actions *= [0, 0, 1, 0]
#         else:
#             actions *= [0, 1, 1, 0]
        
        result = []
        result.append(u1)
        if actions[1] > 0:
            result.append(u2 + actions[1])
        else:
            result.append(u2)
        result.append(self.f3.predict([result])[0] + u3 + actions[2])
        result.append(self.f4.predict([[result[-1]]])[0] + u4)
        return np.array(result).reshape(1,-1)




def get_recourses(data, model, lime=False):
    population = data.copy()
    population['PRED'] = model.predict(population.to_numpy())
    if lime:
        explainer = LimeTabularExplainer(data.to_numpy(), feature_names=data.columns, class_names=np.unique(population['PRED']), discretize_continuous=False)
    population = population[population['PRED'] == 0]
    population = population.drop(columns=['PRED'])
    
    no_recourse = []
    recourses = pd.DataFrame(columns=population.columns)
    for index, row in tqdm(population.iterrows(), total=population.shape[0]):
        row = row.values.reshape(1,-1)
        if lime:
            clf = explainer.explain_instance(row.flatten(), predict_fn=model.predict_proba)#, labels=[0,1])
            coefficients = {feat: coeff for (feat, coeff) in clf.as_list()}
            coefficients = np.array([coefficients.get(feat, 0) for feat in population.columns])
            intercept = clf.intercept[1] # - clf_base2.intercept[1]
        else:
            coefficients = np.array(model.coef_).flatten()
            intercept = model.intercept_[0]
        rb = RecourseBuilder(
              optimizer="cplex",
              coefficients=np.round(coefficients, decimals=4),
              intercept=np.round(intercept, decimals=4),
              action_set=ActionSet(X=data),
              x=row
        )
        rb = rb.fit()
        if not rb['feasible']:
            no_recourse.append(index)
            continue
        new_row = row + rb['actions']
        if model.predict(new_row)[0] == 0:
            no_recourse.append(index)
            continue
        recourses.loc[index] = new_row.flatten()
    return recourses, no_recourse

def get_grad(x_in, model):
    x_in = x_in.flatten()
    def fn(x_in, model):
        return model.predict_proba(x_in.reshape(1,-1))[0][1]
    for eps in [1e-7, 1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5]:
        grads = optimize.approx_fprime(x_in, fn, eps, model)
        if np.mean(grads) > 0:
            #print(eps)
            return grads
    return None

def get_cf(x_in, model, step_size=0.1, max_iter=1000, threshold=0.5, num_grad=True):
    if not num_grad:
        return step_size * model.coef_
    rec = np.array(x_in).reshape(1,-1).astype(float)
    iter_count = 0
    while(model.predict_proba(rec)[0][1] < threshold):
        if iter_count > max_iter:
            return None
        grad = get_grad(rec, model)
        if grad is None:
            return None
        rec += step_size * grad.reshape(1,-1)
        iter_count += 1
    return rec

def get_counterfactuals(data, model, num_grad=True):
    population = data.copy()
    population['PRED'] = model.predict(population.to_numpy())
    population = population[population['PRED'] == 0]
    population = population.drop(columns=['PRED'])
    
    no_recourse = []
    recourses = pd.DataFrame(columns=population.columns)
    for index, row in tqdm(population.iterrows(), total=population.shape[0]):
        rec = get_cf(row, model, num_grad=num_grad)
        if rec is None:
            no_recourse.append(index)
        else:
            recourses.loc[index] = rec.flatten()
    return recourses, no_recourse

def get_ccf(x_in, model, scm, step_sizes=[10, 1e2, 1e3, 1e5, 1e7, 1e10], max_iter=1000, threshold=0.5):
    rec = np.array(x_in).reshape(1,-1).astype(float)
    iter_count = 0
    while(model.predict_proba(rec)[0][1] < threshold):
        if iter_count > max_iter:
            return None
        grad = get_grad(rec, model)
        if grad is None:
            return None
        nrec = scm.act(rec, grad)
        for step_size in step_sizes:
            if np.mean(rec - nrec) > 1:
                break
            nrec = scm.act(rec, grad*step_size)
        rec = nrec
        iter_count += 1
    return rec

def get_ccounterfactuals(data, model, scm):
    #print(scm.f3.coef_)
    population = data.copy()
    population['PRED'] = model.predict(population.to_numpy())
    population = population[population['PRED'] == 0]
    population = population.drop(columns=['PRED'])
    
    no_recourse = []
    recourses = pd.DataFrame(columns=population.columns)
    for index, row in tqdm(population.iterrows(), total=population.shape[0]):
        rec = get_ccf(row, model, scm)
        if rec is None:
            no_recourse.append(index)
        else:
            recourses.loc[index] = rec.flatten()
    return recourses, no_recourse


def run(infile, modeltype, method, comment=None, **kwargs):
    X1_train = pd.read_csv('../data/cleaned/'+infile+'/D1_train_X.csv', index_col=0)
    y1_train = pd.read_csv('../data/cleaned/'+infile+'/D1_train_y.csv', index_col=0, names=['y_true'])
    X1_test = pd.read_csv('../data/cleaned/'+infile+'/D1_test_X.csv', index_col=0)
    y1_test = pd.read_csv('../data/cleaned/'+infile+'/D1_test_y.csv', index_col=0, names=['y_true'])
    X2_train = pd.read_csv('../data/cleaned/'+infile+'/D2_train_X.csv', index_col=0)
    y2_train = pd.read_csv('../data/cleaned/'+infile+'/D2_train_y.csv', index_col=0, names=['y_true'])
    X2_test = pd.read_csv('../data/cleaned/'+infile+'/D2_test_X.csv', index_col=0)
    y2_test = pd.read_csv('../data/cleaned/'+infile+'/D2_test_y.csv', index_col=0, names=['y_true'])
    
    if method == 'Causal12' or method == 'Causal11' or method == 'Causal22':
            if infile == 'correction':
                X1_train = X1_train[['personal_status_sex', 'age', 'amount', 'duration']]
                X1_test = X1_test[['personal_status_sex', 'age', 'amount', 'duration']]
                X2_train = X2_train[['personal_status_sex', 'age', 'amount', 'duration']]
                X2_test = X2_test[['personal_status_sex', 'age', 'amount', 'duration']]
            else:
                raise Exception('Not-Implemented-Error')
    
    M1, M2 = None, None
    if modeltype == 'RandomForestClassifier':
        M1 = RandomForestClassifier(n_estimators=10, random_state=0)
        M1.fit(X1_train.to_numpy(dtype=np.float64), y1_train.to_numpy(dtype=np.float64).reshape(-1))
        M2 = RandomForestClassifier(n_estimators=10, random_state=0)
        M2.fit(X2_train.to_numpy(dtype=np.float64), y2_train.to_numpy(dtype=np.float64).reshape(-1))
    elif modeltype == 'XGBClassifier':
        M1 = XGBClassifier(random_state=0)
        M1.fit(X1_train.to_numpy(dtype=np.float64), y1_train.to_numpy(dtype=np.float64).reshape(-1))
        M2 = XGBClassifier(random_state=0)
        M2.fit(X2_train.to_numpy(dtype=np.float64), y2_train.to_numpy(dtype=np.float64).reshape(-1))
        #M1 = XGBClassifier(random_state=0)
        #M1.fit(X1_train, y1_train)
        #M2 = XGBClassifier(random_state=0)
        #M2.fit(X2_train, y2_train)
    elif modeltype == 'LogisticRegression':
        M1 = LogisticRegression(random_state=0, solver='liblinear')
        M1.fit(X1_train.to_numpy(dtype=np.float64), y1_train.to_numpy(dtype=np.float64).reshape(-1))
        M2 = LogisticRegression(random_state=0, solver='liblinear')
        M2.fit(X2_train.to_numpy(dtype=np.float64), y2_train.to_numpy(dtype=np.float64).reshape(-1))
    elif modeltype == 'LinearSVC':
        M1 = LinearSVC(random_state=0)
        M1.fit(X1_train.to_numpy(dtype=np.float64), y1_train.to_numpy(dtype=np.float64).reshape(-1))
        M2 = LinearSVC(random_state=0)
        M2.fit(X2_train.to_numpy(dtype=np.float64), y2_train.to_numpy(dtype=np.float64).reshape(-1))
    elif modeltype == 'MLPClassifier':
        M1 = MLPClassifier(hidden_layer_sizes=kwargs.get('arch', (10,5)), random_state=0)
        M1.fit(X1_train.to_numpy(dtype=np.float64), y1_train.to_numpy(dtype=np.float64).reshape(-1))
        M2 = MLPClassifier(hidden_layer_sizes=kwargs.get('arch', (10,5)), random_state=0)
        M2.fit(X2_train.to_numpy(dtype=np.float64), y2_train.to_numpy(dtype=np.float64).reshape(-1))
    if M1 is None or M2 is None:
        raise Exception('Unsupported Model Type')
    y1_train_pred = M1.predict(X1_train.to_numpy())
    y1_test_pred = M1.predict(X1_test.to_numpy())
    y2_train_pred = M2.predict(X2_train.to_numpy())
    y2_test_pred = M2.predict(X2_test.to_numpy())
    
    M1_training_acc = accuracy_score(y1_train, y1_train_pred)
    M1_test_acc = accuracy_score(y1_test, y1_test_pred)
    M2_training_acc = accuracy_score(y2_train, y2_train_pred)
    M2_test_acc = accuracy_score(y2_test, y2_test_pred)
    
    y2_train_pred_M1 = M1.predict(X2_train.to_numpy())
    y2_test_pred_M1 = M1.predict(X2_test.to_numpy())
    y1_train_pred_M2 = M2.predict(X1_train.to_numpy())
    y1_test_pred_M2 = M2.predict(X1_test.to_numpy())
    
    M1_acc_on_D2_train = accuracy_score(y2_train, y2_train_pred_M1)
    M1_acc_on_D2_test = accuracy_score(y2_test, y2_test_pred_M1)
    M2_acc_on_D1_train = accuracy_score(y1_train, y1_train_pred_M2)
    M2_acc_on_D1_test = accuracy_score(y1_test, y1_test_pred_M2)
    
    cf1, cf2, nor1, nor2 = None, None, None, None
    if method == 'ActionableRecourse':
        if modeltype == 'RandomForestClassifier' or modeltype == 'XGBClassifier' or modeltype == 'MLPClassifier':
            cf1, nor1 = get_recourses(X1_train, M1, lime=True)
            cf2, nor2 = get_recourses(X2_train, M2, lime=True)
        elif modeltype == 'LogisticRegression' or modeltype == 'LinearSVC':
            cf1, nor1 = get_recourses(X1_train, M1)
            cf2, nor2 = get_recourses(X2_train, M2)
    elif method == 'Counterfactuals':
        if modeltype == 'LogisticRegression' or modeltype == 'RandomForestClassifier' or modeltype == 'XGBClassifier' or modeltype == 'MLPClassifier':
            cf1, nor1 = get_counterfactuals(X1_train, M1)
            cf2, nor2 = get_counterfactuals(X2_train, M2)
        elif modeltype == 'LinearSVC':
            cf1, nor1 = get_counterfactuals(X1_train, M1, num_grad=False)
            cf2, nor2 = get_counterfactuals(X2_train, M2, num_grad=False)
    elif method == 'Causal12':
        if modeltype == 'LogisticRegression' or modeltype == 'RandomForestClassifier' or modeltype == 'XGBClassifier' or modeltype == 'MLPClassifier':
            scm1 = GermanSCM(X1_train)
            scm2 = GermanSCM(X2_train)
            #print()
            #print(scm1.f3.coef_)
            #print(scm2.f3.coef_)
            #print()
            cf1, nor1 = get_ccounterfactuals(X1_train, M1, scm1)
            cf2, nor2 = get_ccounterfactuals(X2_train, M2, scm2)
        elif modeltype == 'LinearSVC':
            raise Exception('Not-Implemented-Error')
    elif method == 'Causal22':
        if modeltype == 'LogisticRegression' or modeltype == 'RandomForestClassifier' or modeltype == 'XGBClassifier' or modeltype == 'MLPClassifier':
            scm2 = GermanSCM(X2_train)
            cf1, nor1 = get_ccounterfactuals(X1_train, M1, scm2)
            cf2, nor2 = get_ccounterfactuals(X2_train, M2, scm2)
        elif modeltype == 'LinearSVC':
            raise Exception('Not-Implemented-Error')
    elif method == 'Causal11':
        if modeltype == 'LogisticRegression' or modeltype == 'RandomForestClassifier' or modeltype == 'XGBClassifier' or modeltype == 'MLPClassifier':
            scm1 = GermanSCM(X1_train)
            cf1, nor1 = get_ccounterfactuals(X1_train, M1, scm1)
            cf2, nor2 = get_ccounterfactuals(X2_train, M2, scm1)
        elif modeltype == 'LinearSVC':
            raise Exception('Not-Implemented-Error')
    cf1_inv_by_M2, cf2_inv_by_M1 = np.nan, np.nan
    if len(cf1) >= 1:
        cf1_inv_by_M2 = 100*(1-np.mean(M2.predict(cf1.to_numpy())))
    if len(cf2) >= 1:
        cf2_inv_by_M1 = 100*(1-np.mean(M1.predict(cf2.to_numpy())))
    
    D1_total = len(y1_train)
    D2_total = len(y2_train)
    D1_zeros = D1_total - np.sum(y1_train.to_numpy())
    D2_zeros = D2_total - np.sum(y2_train.to_numpy())
    D1_zero_preds = D1_total - np.sum(M1.predict(X1_train.to_numpy()))
    D2_zero_preds = D2_total - np.sum(M2.predict(X2_train.to_numpy()))
    
    M1_crossacc = cross_val_score(M1, X1_train.to_numpy(dtype=np.float64), y1_train.to_numpy(dtype=np.float64).reshape(-1), cv=10).mean()
    M2_crossacc = cross_val_score(M2, X2_train.to_numpy(dtype=np.float64), y2_train.to_numpy(dtype=np.float64).reshape(-1), cv=10).mean()
    
    
    print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format
(infile, modeltype, method, D1_total, D2_total, D1_zeros, D2_zeros, D1_zero_preds, D2_zero_preds, M1_training_acc, M1_test_acc, M2_training_acc, M2_test_acc, M1_acc_on_D2_train, M1_acc_on_D2_test, M2_acc_on_D1_train, M2_acc_on_D1_test, len(cf1), len(nor1), cf1_inv_by_M2, len(cf2), len(nor2), cf2_inv_by_M1, M1_crossacc, M2_crossacc, comment))

def main():
    np.random.seed(0)
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='temporal')
    parser.add_argument('--classifier', type=str, default='lr')
    parser.add_argument('--method', type=str, default='ar')
    parser.add_argument('--comment', default='None')
    parser.add_argument('--arch', nargs='+', type=int, default=[10, 5])
    args = parser.parse_args()
    dataset = args.dataset
    method = args.method
    comment = args.comment
    if method == 'AR' or method == 'ar' or method == 'berk' or method == 'Berk':
        method = 'ActionableRecourse'
    elif method == 'cf' or method == 'CF' or method == 'wachter' or method == 'Wachter' or method == 'sw' or method == 'SW':
        method = 'Counterfactuals'
    elif method == 'c12':
        method = 'Causal12'
    elif method == 'c11':
        method = 'Causal11'
    elif method == 'c22':
        method = 'Causal22'
    clf = args.classifier
    if clf == 'rf' or clf == 'RF':
        clf = 'RandomForestClassifier'
    elif clf == 'xgb' or clf == 'XGB' or clf == 'gbm' or clf == 'GBM' or clf == 'gbt' or clf == 'GBT':
        clf = 'XGBClassifier'
    elif clf == 'lr' or clf == 'LR':
        clf = 'LogisticRegression'
    elif clf == 'svm' or clf == 'SVM' or clf == 'svc' or clf == 'SVC':
        clf = 'LinearSVC'
    elif clf == 'nn' or clf == 'NN' or clf == 'neural-net' or clf == 'Neural-Net' or clf == 'neural-network' or clf == 'Neural-Network':
        clf = 'MLPClassifier'
        comment = str(comment) + 'ARCH: ' + str(args.arch)
    run(dataset, clf, method, comment=comment, arch=args.arch)

if __name__ == '__main__':
    main()

