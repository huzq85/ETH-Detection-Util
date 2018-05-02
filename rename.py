# -*- coding: utf-8 -*-
"""
Created on Tue May  1 21:37:33 2018

@author: huzq85
"""

import os
def rename_files(f_path):
    for root, parent, files in os.walk(f_path):
        for file in files:
            if (file.endswith('png')):
                if file.split('.')[0].endswith('_0'):
                    new_file = file[0:len(file.split('.')[0])-2] + '.png'
                    print('Renaming: ' + file + '-->' + new_file)
                    os.rename(os.path.join(root, file), os.path.join(root, new_file))
                    
                    
if __name__ == '__main__':
    path = r'E:\5-DataSets\ETH\Setup1\BAHNHOF\left'
    rename_files(path)

