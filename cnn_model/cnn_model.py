import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import re
from os.path import join
import pickle
import pandas as pd
from SaveDB.SaveDB import ConnectorDBFiles

def getData(db_connect,file_name):    
    dt = db_connect.load_data(file_name)
    dt = dt.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    Y = dt['target']
    print('Y unique value {}'.format(Y.unique()))
    X = dt.drop(['target','peopleIndex'],axis=1)
    X = X.loc[:,~(X.columns.duplicated())]
    X = X.fillna(-1)
    return X,Y

def calckScore(y_predict, y_test):
    mean_score = accuracy_score(y_test, y_predict)
    mean_precision_score = precision_score(y_test, y_predict,average='micro')
    mean_recall_score = recall_score(y_test, y_predict,average='micro')
    mean_f1_score = f1_score(y_test, y_predict, average='micro')
    return mean_score,mean_precision_score,mean_recall_score,mean_f1_score

def predict(config,data):
    model_folder = config['MODEL']['model_folder']
    with open(join(model_folder,'last_model.hdf5'),"rb") as input_file:
        model = pickle.load(input_file)
    return model.predict(data)

def get_model(input_shape,metric_nm=['accuracy']):

    input_layer = keras.layers.Input(input_shape)
    conv1 = keras.layers.Conv1D(filters=32, kernel_size=3, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)
    
    conv2 = keras.layers.Conv1D(filters=32, kernel_size=3, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv2)
    dns = keras.layers.Dense(49, activation='softmax')(gap_layer)
    output_layer = keras.layers.Dense(7, activation='linear')(dns)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics=metric_nm)
    return model 

def build_model(model_folder,x_train, y_train, x_val, y_val):
    model = get_model(x_train.shape)
    # x_val and y_val are only used to monitor the test loss and NOT for training  
    batch_size = 16
    nb_epochs = 20
    mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

    start_time = time.time() 
    hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
            verbose=True, validation_data=(x_val,y_val))
            
    duration = time.time() - start_time
    model.save(join(model_folder,'last_model.hdf5'))
    model = keras.models.load_model(join(model_folder,'last_model.hdf5'))
    y_pred = model.predict(x_val)
    return y_pred 

def create_model_for_atlass(db_connect,file_data,model_folder):
    X,Y = getData(db_connect,file_data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    y_predict = build_model(model_folder,X_train,y_train,X_test,y_test)
    mean_accuracy_score,mean_precision_score,mean_recall_score,mean_f1_score = calckScore(y_predict,y_test)
    Result = {'accuracy_score' : mean_accuracy_score,'mean_precision_score' : mean_precision_score,
                'mean_recall_score' : mean_recall_score,'mean_f1_score' : mean_f1_score}
    return pd.DataFrame(Result)


def build_model_for_data(config):
    db_connect = ConnectorDBFiles(config['DATACSV']['csv_folder_files'])
    name_atlas = dict(config['FEATURES'])
    model_folder = config['MODEL']['model_folder']
    atlass = {}
    for fl in name_atlas:
        print('File name ={}'.format(name_atlas[fl]))
        res = create_model_for_atlass(db_connect,name_atlas[fl],model_folder)
        db_connect.save_data('best_model.hdf5',res)
        atlass[fl] =res.copy()
    return atlass
