import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys

from SaveDB.SaveDB import ConnectorDBFiles
from os.path import join

def create_img(df,img_path,prefix):
    metric = list(set(df.columns) - set(['name_classificator'],))
    sns.set(rc={'figure.figsize':(12,9)})
    for m in metric:
        sns_plot = sns.barplot(data=df,x='name_classificator',y=m)
        sns_plot.set_ylim([0.8, 1])
        sns_plot.set_title(m)
        fig = sns_plot.get_figure()
        name = '{}_{}.png'.format(prefix,m)
        path = join(img_path,name)
        fig.savefig(path,dpi=600)

def imgs_for_attlass(config):
    db_connect = ConnectorDBFiles(config['DATACSV']['csv_folder_files'])
    name_model = dict(config['MODEL_FOR_ATLASS'])
    for val in name_model.values(): 
        df = db_connect.load_data(val)
        create_img(df,config['IMG']['img_folder'],val) 