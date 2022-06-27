import configparser
import sys

from cls_model.cls_model import build_model_for_atlass_harvard_oxford,build_model_for_atlass_craddock,build_model_for_atlas_aal
from cls_model.create_img import imgs_for_attlass
from cnn_model.cnn_model import build_model_for_data


from extract_ROI.extract_ROI import extract_time_series_from_fmrt
from SaveDB.SaveDB import ConnectorDBFiles
from extractTimeSerias.extractTimeSerias import extract_features_from_ts

import findspark
findspark.init()


import pyspark
from pyspark import SparkConf
######configuration spark


conf = pyspark.SparkConf()
conf.setAppName('DS_Test')
conf.set('spark.executor.memory', '40G')
conf.set('spark.driver.memory', '80G')
conf.set('spark.driver.maxResultSize', '80G')

spark = pyspark.SparkContext(conf=conf)



from os import listdir
from os.path import isfile, join


if __name__ == '__main__':

    config_path = './config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)

    #extract_time_series_from_fmrt(spark,config)
    print('Extract data finished')

    #extract_features_from_ts(config)
    print('Extract extract_time_series finished')
    build_model_for_atlass_harvard_oxford(config)
    build_model_for_atlass_craddock(config)
    build_model_for_atlas_aal(config)
    print('Build model')
    imgs_for_attlass(config)
    print('Create img')


#sudo PYTHON=/root/.pyenv/versions/3.6.9/bin/python PYSPARK_PYTHON=/root/.pyenv/versions/3.6.9/bin/python PYSPARK_DRIVER_PYTHON=/root/.pyenv/versions/3.6.9/bin/python spark-submit ./main.py
#sudo PYTHON=/root/.pyenv/versions/3.6.9/bin/python PYSPARK_PYTHON=/root/.pyenv/versions/3.6.9/bin/python PYSPARK_DRIVER_PYTHON=/root/.pyenv/versions/3.6.9/bin/python spark-submit ./main.py --py-files ./extract_ROI/extract_ROI.py,./SaveDB/SaveDB.py,./extractTimeSerias/extractTimeSerias.py 