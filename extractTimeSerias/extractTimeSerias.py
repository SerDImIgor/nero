import sys
from SaveDB.SaveDB import ConnectorDBFiles
import logging
logging.basicConfig(level=logging.ERROR)
import configparser
import os
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh import extract_features
import pandas as pn
import numpy as np



list_atlas = []


def ExtractTimeSeries(table_load,save_data,db_connect):
    data = db_connect.load_data(table_load)
    print(data.shape)
    Y_data = []
    peopleIndex_arr = []
    data['time'] = 0
    data['id'] = 0
    un_people = data.peopleIndex.unique()
    un_target = data.target.unique()
    un_id = 0
    for id_people in un_people:
        for tg in un_target:
            size = data.loc[(data['peopleIndex']==id_people) & (data['target']==tg)]
            if size.shape[0] > 0:
                Y_data.append(size['target'].iloc[0])
                peopleIndex_arr.append(size['peopleIndex'].iloc[0])
                arr = np.stack((np.arange(size.shape[0]),un_id * np.ones(size.shape[0]) ),axis=-1)
                data.loc[(data['peopleIndex']==id_people) & (data['target']==tg),['time','id']] = arr
                un_id+=1
    list_features = set(list(data))
    B = {'isLR','peopleIndex','target'}
    list_features = list(list_features.difference(B))
    print('--------------------------------{}----------------------'.format(list_features))
    X_filtered = extract_relevant_features(data[list_features],pn.Series(Y_data), column_id='id', column_sort='time')
    X_filtered['target'] = pn.Series(Y_data)
    X_filtered['peopleIndex'] = pn.Series(peopleIndex_arr)
    print('The atlas is '+str(table_load)+'. The size is '+str(X_filtered.shape))
    return X_filtered


def extract_features_from_ts(config):
    db_connect = ConnectorDBFiles(config['DATACSV']['csv_folder_files'])
    atlas = dict(config['ATLAS'])
    print('---------------------------------------{}-----------------'.format(atlas))

    list_atlas.append({'load':config['ATLAS']['atlas_harvard_oxford'],'save':config['FEATURES']['harvard_oxford_featires']})
    list_atlas.append({'load':config['ATLAS']['atlas_craddock'],'save':config['FEATURES']['atlas_craddock_featires']})
    list_atlas.append({'load':config['ATLAS']['atlas_aal'],'save':config['FEATURES']['atlas_aal_featires']})

    for atlas in list_atlas:
        print('load='+str(atlas['load'])+' save='+str(atlas['save']))
        X_filtered = ExtractTimeSeries(atlas['load'],atlas['save'],db_connect)
        db_connect.save_data(atlas['save'],X_filtered)

