'''
base_services.py
================================================================================
The module containing base services

Copyright 2020 - The IoTwins Project Consortium, Alma Mater Studiorum
Università di Bologna. All rights reserved.
'''
#!/usr/bin/python3.6
import pandas as pd
import re, pickle, sys
import os.path

def store_csv_test(df_fname):
    '''
    base_services.store_csv_test
    '''
    print(df_fname)
    df = pd.read_csv(df_fname)
    dest_df_fname = 'stored_{}'.format(df_fname)
    df.to_csv(dest_df_fname)
    print(dest_df_fname)

if __name__ == '__main__':
    service_type = sys.argv[1]
    service_args = sys.argv[2:]
    if service_type == 0:
        store_csv_test(service_args)
    else:
        print('Unsupported option')


