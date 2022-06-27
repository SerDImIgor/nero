#import findspark
#findspark.init()
import configparser
from zipfile import ZipFile
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import glob
import os
import re
import shutil
import numpy as np
from SaveDB.SaveDB import ConnectorDBFiles
import pandas as pd

import sys
import subprocess
import socket
from multiprocessing.pool import Pool


from nilearn import image
from nilearn.connectome import ConnectivityMeasure

import nibabel
import zipfile

from scipy.signal import savgol_filter, medfilt

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import NullType
from pyspark.sql import SparkSession
from pyspark import SparkConf
import happybase



convert_dic = {'EMOTION':0,'GAMBLING':1,'LANGUAGE':2,'MOTOR':3,'RELATIONAL':4,'WM':5,'SOCIAL':6}
extract_folder = ''

#SPARK extract data from archive
def zip_extract(fmri_fl):
    extract_folder ='/data/DSergeev/fmriData/'
    with ZipFile(fmri_fl, 'r') as zip_file:
        file_names = zip_file.namelist()
        files = [fl for fl in file_names  if re.search('[EMOTION_RL|EMOTION_LR|GAMBLING_RL|GAMBLING_LR|LANGUAGE_RL|LANGUAGE_LR|MOTOR_RL|MOTOR_LR|RELATIONAL_RL|RELATIONAL_LR|WM_RL|WM_LR||SOCIAL_RL|SOCIAL_LR ].nii.gz',fl)]
        extract_files = []
        for fl in files:
            val = fl.split('/')
            id_user = val[3].split('_')[1]
            folder_name = os.path.dirname(val[0] +'_' + id_user + '/')
            file_name = val[len(val)-1]
            if not os.path.exists(os.path.join(extract_folder,folder_name)):
                os.mkdir(os.path.join(extract_folder,folder_name))
            source = zip_file.open(fl)
            target = open(os.path.join(extract_folder, os.path.join(folder_name,file_name)), "wb")
            with source, target:
                shutil.copyfileobj(source, target)
            target.close()
            source.close()
            extract_files.append(os.path.join(extract_folder, os.path.join(folder_name,file_name)))
        return extract_files

def cutFiles(lst_files,extract_folder,cnt):
	ls_res = []
	for ls in lst_files:
		aa = re.compile('\d{6,6}')
		res = aa.findall(ls)
		ls_res.append(res[0])
	ls_res = np.unique(ls_res)
	ls_res = ls_res[0:cnt]
	fl_result=[]
	for num in ls_res:
		for ls in lst_files:
			aa = re.compile(num)
			if aa.search(ls):
				fl_result.append(ls)

	return fl_result


def extract_all_files(spark,extract_folder,source_folder):

    fmri_files = [os.path.join(source_folder,f) for f in os.listdir(source_folder) if re.search(r'(EMOTION|GAMBLING|LANGUAGE|MOTOR|RELATIONAL|SOCIAL|WM).*\.zip$', f)]
    fmri_files = cutFiles(fmri_files,extract_folder,200)
    print('COUNT FILES={}'.format(len(fmri_files)))
    res = spark.parallelize(fmri_files).map(zip_extract).collect()

    df = pd.DataFrame.from_records(res)
    df.to_csv(os.path.join(extract_folder,'list_files.csv'),index=False)

def get_list_files(extract_folder):
    fmri_folder = os.listdir(extract_folder)
    LR_files = []
    RL_files = []
    count_drop_dt = 0
    print(len(fmri_folder))
    for fld in fmri_folder:
        fmri_files = glob.glob(os.path.join(extract_folder,fld) + '/*.nii.gz')
        #print(fmri_files)
        if len(fmri_files)!=2:
            count_drop_dt += 1
            continue
        for fl in fmri_files:
            if re.search('LR.nii.gz', fl):
                LR_files.append(fl)
            if re.search('RL.nii.gz', fl):
                RL_files.append(fl)

    print('Count drop files='+str(count_drop_dt))
    data = pd.DataFrame({'0':LR_files,'1':RL_files})
    return data[0:200]
##----------------------------------------------------------------------------work_with_atlas-----------------------------------------------------------
def load_atlas(name_atlas):
    list_atlas = {}
    print(name_atlas)
    for nm in name_atlas.keys():
        if nm=='atlas_harvard_oxford':
            atlas_harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            atlas_harvard_oxford_maps = atlas_harvard_oxford.maps
            print(len(atlas_harvard_oxford.labels))  # 4D data
            list_atlas[nm]={'maps':atlas_harvard_oxford_maps,'regions':atlas_harvard_oxford.labels,'file': name_atlas[nm]}
            print('-------------------Atlass _harvard_oxford--------------------------------------------')

        elif nm=='atlas_craddock':
            atlas_craddock = datasets.fetch_atlas_destrieux_2009()
            atlas_craddock_maps = atlas_craddock.maps
            print(len(atlas_craddock.labels))
            list_atlas[nm]={'maps':atlas_craddock_maps,'regions':atlas_craddock.labels,'file':name_atlas[nm]}
            print('-------------------atlas_craddock--------------------------------------------')

        elif nm=='atlas_aal':
            atlas_aal = datasets.fetch_atlas_aal()
            atlas_aal_maps = atlas_aal.maps
            print(len(atlas_aal.labels))
            list_atlas[nm]={'maps':atlas_aal_maps,'regions':atlas_aal.labels,'file':name_atlas[nm]}
            print('-------------------atlas_aal--------------------------------------------')

    return list_atlas



#SPARK extract time serias
def extract_ROI(attlass,path,id_user,tp_task):
    masker = NiftiLabelsMasker(labels_img=attlass, standardize = False,memory='nilearn_cache')
    print(type(masker))
    r = masker.fit_transform(path)
    arr_1 = np.ones((r.shape[0],1)) * id_user
    arr_2 = np.ones((r.shape[0],1)) * convert_dic[tp_task]
    return np.append(r,np.append(arr_1,arr_2,axis=1),axis=1)

def get_time_serias(file):
    nums = re.compile(r"\d{6}_\w{2,10}")
    prefix_1 = nums.search(file[0]).group(0)
    prefix_1 = prefix_1.split('_')
    prefix_2 = nums.search(file[1]).group(0)
    prefix_2 = prefix_2.split('_')
    if int(prefix_1[0]) != int(prefix_2[0]):
        return ('key1',1)
    arr1 = extract_ROI(file[2],file[0],int(prefix_1[0]),prefix_1[1])
    arr2 = extract_ROI(file[2],file[1],int(prefix_1[0]),prefix_1[1])
    time_series = np.vstack((arr1,arr2))
    m_columns = ['x_' + str(i) for i in range(time_series.shape[1])]
    m_columns[len(m_columns)-2] = 'peopleIndex'
    m_columns[len(m_columns)-1] = 'target'
    return ('key',pd.DataFrame(time_series,columns = m_columns))



def norm_data_by_lenght(dt):
    val = dt.groupby(['peopleIndex','target']).count().reset_index(drop=True)['x_0']
    min_lenght = np.min(val)
    print('Norm value = '+str(min_lenght))
    unic_user = dt.peopleIndex.unique()
    unic_target = dt.target.unique()

    data_norm = pd.DataFrame()
    user_uid = 0
    for u in unic_user:
        for tg in unic_target:
            val = dt.loc[(dt['peopleIndex']==u) & (dt['target']==tg)].reset_index(drop=True)
            if val.shape[0] > 0:
                val = val[0:min_lenght].reset_index(drop=True)
                if data_norm.shape[0] == 0:
                    data_norm = val
                else:
                    data_norm = data_norm.append(val,ignore_index=True)
                user_uid = user_uid + 1
    return data_norm

def concat_data(dt1,dt2):
    return pd.concat([dt1, dt2])

def extract_atlas(spark,data,atlass_name,db_connect):
    print(data.shape)
    list_atlas = load_atlas(atlass_name)
    for ls in list_atlas.keys():
        print(list_atlas[ls]['file'])
        address = np.array( [list_atlas[ls]['maps']]* data.shape[0] ).reshape((-1,1))
        address = np.hstack((data,address))

        print(address.shape)
 
        res = spark.parallelize(address).map(get_time_serias).reduceByKey(concat_data).collect()
        print(res[0][1].shape)
        dt = norm_data_by_lenght(res[0][1])
        db_connect.save_data(list_atlas[ls]['file'],dt)
        print('Atlass-------------------------------{}'.format(list_atlas[ls]['file']))


def extract_time_series_from_fmrt(spark,config):
    global extract_folder
    db_connect = ConnectorDBFiles(config['DATACSV']['csv_folder_files'])
    extract_folder = os.path.dirname(config['PATH']['extract_folder']) # extract fmri files
    source_folder = os.path.dirname(config['PATH']['source_folder']) # source fmri files
    print(extract_folder)
    #ATLAS
    atlas = dict(config['ATLAS'])

    is_extract_files = config['EXTRACTFILES']['is_extract_files']
    if is_extract_files == 'True':
        #extract_all_files()
        extract_all_files(spark,extract_folder,source_folder)
    else:
        data = get_list_files(extract_folder)
        extract_atlas(spark,data.values,atlas,db_connect)
 
 #--------------------------------------Egor------------------------------------------------------

def get_ROI(path, masker, delete_confounds):
    if delete_confounds:
        confounds = image.high_variance_confounds(path, n_confounds = 5)
        return masker.transform(path, confounds = confounds)
    else:
        return masker.transform(path)
    

def get_masker(atlas_name, atlas_path):
    if atlas_name=='harvard_oxford':
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', data_dir=atlas_path)
    elif atlas_name=='juelich':
        atlas = datasets.fetch_atlas_juelich('maxprob-thr0-1mm', data_dir=atlas_path)
    else:
        raise OSError('there is no atlas {0}'.format(atlas_name))
    atlas_filename = atlas.maps

    masker = NiftiLabelsMasker(labels_img=atlas_filename, 
                               memory='nilearn_cache')
    masker.fit()
    return masker

      
def extract_archive_HCP(archive_path, path_to_save):
    archive = zipfile.ZipFile(archive_path)
    people_number = int(archive_path.split('/')[-1].split('_')[0])
    rest_number = archive_path.split('/')[-1].split('_')[3]
    path_in_archive = '{0}/MNINonLinear/Results/rfMRI_{1}_{2}/rfMRI_{1}_{2}.nii.gz'
    pathes = []
    for file in archive.namelist():
        if file.startswith(path_in_archive.format(people_number, rest_number, 'LR')) or file.startswith(path_in_archive.format(people_number, rest_number, 'RL')):
            archive.extract(file, path_to_save)
            pathes.append(path_to_save+'/'+str(file))
    return pathes


def write_to_hbase(data, atlas, table_name, person, person_number):
    connection = happybase.Connection()
    table = connection.table(table_name)

    data = data.to_json().encode('utf-8')
    row = 'HCP_'+person
    row = row.encode('utf-8')
    table.put(row, {b'data:person': person_number.encode('utf-8'),
                    b'data:dataset': b'HCP',
                    b'data:atlas': atlas.encode('utf-8'),
                    b'data:roi': data})


def extract_ROI_atlass(config):

    conf = SparkConf()
    conf.setAppName('Test')
    conf.set("spark.hadoop.yarn.resourcemanager.hostname", config['SPARK_CONFIGURATION']['resourcemanager_hostname'])
    # conf.set("spark.hadoop.yarn.resourcemanager.address", config['SPARK_CONFIGURATION']['resourcemanager_address'])
    conf.set('spark.executor.memoryOverhead', config['SPARK_CONFIGURATION']['memoryOverhead'])
    conf.set('spark.executor.instances', config['SPARK_CONFIGURATION']['instances'])
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    data_path = config['EXTRACTION']['data_path']
    atlas_name = config['EXTRACTION']['extraction_method']

    pathes = glob.glob(data_path + '/*REST1_preproc.zip') + glob.glob(data_path + '/*REST2_preproc.zip')
    data = pd.DataFrame(pathes, columns=['path'])
    data['people_number'] = data['path'].apply(lambda x: os.path.split(x)[-1].split('_')[0])
    # data = pd.read_csv(data_path)
    data = data[:1]
    print(data.shape)
    data_spark = spark.createDataFrame(data)
    
    @pandas_udf("path string, people_number long", PandasUDFType.GROUPED_MAP)
    def extract_ROI_spark(data):
        people_path = data['path']
        print(1111111111111111111111111, data.shape)
        people_number = np.unique(data['people_number'])[0]

        try:
            os.mkdir('/tmp/nilearn_data')
        except Exception as e:
            a = 1

        atlas_path = config['EXTRACTION']['path_to_method_data']
        masker = get_masker(atlas_name, atlas_path)
        j = 0

        for person in zip(people_path, [people_number]*data.shape[1]):
            print(2)
            pathes = extract_archive_HCP(person[0], '/tmp/nilearn_data')
            j+=1

            for person_path in pathes:
                print(3)
                data_person = get_ROI(person_path, masker, True)
                columns = ['x' + str(i) for i in range(data_person.shape[1])]
                print(4)

                data_person = pd.DataFrame(data_person, columns=columns)
                save_path = config['EXTRACTION']['save_path'] + str(people_number)+'_{}.csv'.format(j)
                j+=1
                #if config['LOAD_REGIONS']['where_to_write']=='hdd':
                data_person.to_csv(save_path, index=False)
                # write_to_hbase(data_person, 
                #                atlas_name, 
                #                'test_table_neuro', 
                #                str(person), 
                #                str(person)+str(j))

                
                os.remove(person_path)
        shutil.rmtree('/tmp/nilearn_data/'+str(people_number))
        temp = pd.DataFrame([[pathes[0],people_number]], columns=['path', 'people_number'])
        return temp

    df1 = data_spark.groupby("people_number").apply(extract_ROI_spark)
    print(df1.toPandas()['path'][0])
