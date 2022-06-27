#import findspark
#findspark.init()
import configparser
from zipfile import ZipFile
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import glob
import os
import io
import re
import shutil
import pandas as pn
import numpy as np
import sys
from SaveDB.SaveDB import ConnectorDBFiles






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

    df = pn.DataFrame.from_records(res)
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
    data = pn.DataFrame({'0':LR_files,'1':RL_files})
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
    return ('key',pn.DataFrame(time_series,columns = m_columns))



def norm_data_by_lenght(dt):
    val = dt.groupby(['peopleIndex','target']).count().reset_index(drop=True)['x_0']
    min_lenght = np.min(val)
    print('Norm value = '+str(min_lenght))
    unic_user = dt.peopleIndex.unique()
    unic_target = dt.target.unique()

    data_norm = pn.DataFrame()
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
    return pn.concat([dt1, dt2])

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
 