#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:55:17 2021

@author: heegulee
"""
import numpy as np
import pandas as pd

from scipy import signal
import poledb as PDB
import ast
from scipy.signal import detrend
PDB.poledb_init()
import openpyxl
import os
import csv
#import win32com.client as win32
import re

global in_num
global out_num

def Signal_Info_out(poleid,stype,num):
    
    global out_x
    global out_y
    global out_z
    
    print('mum',poleid,stype,num)
    sig_x=PDB.get_meas_data(poleid,num,stype,'x')
    sig_y=PDB.get_meas_data(poleid,num,stype,'y')
    sig_z=PDB.get_meas_data(poleid,num,stype,'z')
    
    out_x = sig_x
    out_y = sig_y
    out_z = sig_z
    
def Signal_Info_in(poleid,stype,num):
    
    global sig_x
    
    sig_x=PDB.get_meas_data(poleid,num,stype,'x')
    index_ch=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
    
    sig_x_ptp=[]
    sig_Scan_points=0
    x=0
      
def pole_Info(poleid):
    
    global in_num
    global out_num
    global re_out
    global re_in
    
    pole_sgn_count=PDB.get_meas_result_count(poleid)
    
    re_out=PDB.get_meas_result(poleid, 'OUT')
    num_sig_out=re_out.shape[0]
    
    re_in=PDB.get_meas_result(poleid, 'IN')
    num_sig_in=re_in.shape[0]
    
    in_num = num_sig_in
    out_num = num_sig_out
        
    
    return num_sig_out,num_sig_in,re_out,re_in
    
def conf_file_open(dir):
    os.mkdir('polelist/' + dir + ' 원본데이터')

    Pole_list = PDB.get_pole_list(dir)

    for poleid in Pole_list['poleid']:
        dir_path = 'polelist/' + dir + ' 원본데이터'

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        pole_info = pole_Info(poleid)
        
        print(pole_info)

        combined_data = pd.DataFrame()  # 빈 DataFrame을 생성하여 데이터를 합칠 준비

        for kk in range(in_num):
            stype = 'IN'
            num = int(re_in['measno'][kk])
            in_x = PDB.get_meas_data(poleid, num, stype, 'x')
            combined_data[f"IN-X-{kk+1}"] = in_x.values.flatten()

        for kk in range(out_num):
            stype = 'OUT'
            num = int(re_out['measno'][kk])
            out_z = PDB.get_meas_data(poleid, num, stype, 'z')
            combined_data[f"OUT-Z-{kk+1}"] = out_z.values.flatten()  # 1차원으로 변환하여 추가

        # DataFrame을 CSV 파일로 저장
        combined_data.to_csv(os.path.join(dir_path, f"{poleid}.csv"), index=False)

    return Pole_list
        
name = "부산울산중부산-202403"
conf_file_open(name)