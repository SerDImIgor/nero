import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold

from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from SaveDB.SaveDB import ConnectorDBFiles
from os.path import join

import lightgbm as lgb
import xgboost as xgb
import re
import pickle

# Metric model
def model_KNN(X_train,y_train):
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train,y_train)
    return model

# Probability model
def model_NB(X_train,y_train):
    model = GaussianNB()
    model.fit(X_train,y_train)
    return model


# Group Classificator model
def model_LGBM(X_train,y_train):    
    model = lgb.LGBMClassifier(learning_rate = 0.05,max_depth= 4,
                               min_child_weight=4,n_estimators=320,objective='multiclass',reg_lambda=1)
    model.fit(X_train, y_train)
    return model

def model_Xgboost(X_train,y_train):
    model = xgb.XGBClassifier(learning_rate = 0.1,max_depth = 4,min_child_weight = 4,
                              n_estimators = 80,objective='multi:softmax',reg_lambda = 0.1,n_jobs=-1)
    model.fit(X_train,y_train)
    return model

def model_RF(X_train,y_train):
    model = RandomForestClassifier(max_depth = 32,max_features='log2',n_estimators=240,n_jobs=-1)
    model.fit(X_train,y_train) 
    return model


def calckScore(num_classes,X,Y,model):
    num_splits = 3
    mean_score = 0
    mean_precision_score = 0
    mean_recall_score = 0
    mean_f1_score = 0
    con_matrix = np.zeros((num_classes,num_classes))
    kf = KFold(n_splits = num_splits,shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y[train_index].values, Y[test_index].values
        md = model(X_train,y_train)
        y_predict = md.predict(X_test)
        mean_score += accuracy_score(y_test, y_predict)
        mean_precision_score += precision_score(y_test, y_predict,average='micro')
        mean_recall_score += recall_score(y_test, y_predict,average='micro')
        mean_f1_score += f1_score(y_test, y_predict, average='micro')
        
    mean_score = mean_score/num_splits
    mean_precision_score = mean_precision_score/num_splits
    mean_recall_score = mean_recall_score/num_splits
    mean_f1_score = mean_f1_score/num_splits
    #print(mean_matrix)
    return mean_score,mean_precision_score,mean_recall_score,mean_f1_score,md

def getData(db_connect,file_name):
    
    dt = db_connect.load_data(file_name)
    dt = dt.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    Y = dt['target']
    print('Y unique value {}'.format(Y.unique()))
    X = dt.drop(['target','peopleIndex'],axis=1)
    X = X.loc[:,~(X.columns.duplicated())]
    X = X.fillna(-1)
    return X,Y





def create_model_for_atlass(db_connect,file_data,model_folder):
    dicClass = {
        'LGBMClassifier':model_LGBM,
        'XGBClassifier':model_Xgboost,
        'RandomForestClassifier':model_RF,
        'GaussianNB' : model_NB,
        'KNeighborsClassifier':model_KNN
    }
    num_classes = 7
    X,Y = getData(db_connect,file_data)
    Result = []
    for classificator in dicClass.keys():
        print(classificator)
        mean_accuracy_score,mean_precision_score,mean_recall_score,mean_f1_score,model = calckScore(num_classes,X,Y,dicClass[classificator])
        r = {
            'name_classificator' : classificator,
            'accuracy_score' : mean_accuracy_score,
            'mean_precision_score' : mean_precision_score,
            'mean_recall_score' : mean_recall_score,
            'mean_f1_score' : mean_f1_score}
        Result.append(r)
        name_classifcator = '{}_{}'.format(file_data,classificator)
        with open(join(model_folder,name_classifcator), 'wb') as pickle_file:
            pickle.dump(model,pickle_file)
    return pd.DataFrame(Result)

def build_model_for_atlass_harvard_oxford(config):
    name_atlas = dict(config['FEATURES'])
    return build_model_for_atlass(config,'harvard_oxford_featires',name_atlas['harvard_oxford_featires'])

def build_model_for_atlass_craddock(config):
    name_atlas = dict(config['FEATURES'])
    return build_model_for_atlass(config,'atlas_craddock_featires',name_atlas['atlas_craddock_featires'])

def build_model_for_atlas_aal(config):
    name_atlas = dict(config['FEATURES'])
    return build_model_for_atlass(config,'atlas_aal_featires',name_atlas['atlas_aal_featires'])


def build_model_for_atlass(config,name_atlas,name_atlas_file):
    db_connect = ConnectorDBFiles(config['DATACSV']['csv_folder_files'])
    name_model = dict(config['MODEL_FOR_ATLASS'])
    model_folder = config['MODEL']['model_folder']
    print('File name ={}'.format(name_atlas_file))
    res = create_model_for_atlass(db_connect,name_atlas_file,model_folder)
    db_connect.save_data(name_model[name_atlas],res)

    return res.copy()

#def build_model_for_atlass(config,name_atlas):
#    db_connect = ConnectorDBFiles(config['DATACSV']['csv_folder_files'])
#    name_atlas = dict(config['FEATURES'])
#    name_model = dict(config['MODEL_FOR_ATLASS'])
#    model_folder = config['MODEL']['model_folder']
#    atlass = {}
#    for fl in name_atlas:
#        print('File name ={}'.format(name_atlas[fl]))
#        res = create_model_for_atlass(db_connect,name_atlas[fl],model_folder)
#        db_connect.save_data(name_model[fl],res)
#        atlass[fl] =res.copy()
#    return atlass

def predict(config,data,name_model):
    model_folder = config['MODEL']['model_folder']
    with open(join(model_folder,name_model),"rb") as input_file:
        model = pickle.load(input_file)
    return model.predict(data)