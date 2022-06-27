import pandas as pn
import os

class ConnectorDBFiles:
    def __init__(self,prefix):
        self.prefix = os.path.dirname(prefix)
    def load_data(self,name_table):
        return pn.read_csv(os.path.join(self.prefix,name_table+'.csv'))
    def save_data(self,name_table,data):
        data.to_csv(os.path.join(self.prefix,name_table+'.csv'),index=False)