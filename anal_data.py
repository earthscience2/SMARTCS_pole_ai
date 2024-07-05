import bottleneck as bn
import numpy as np
import pandas as pd
from scipy import signal
import logger
# 문제가 되는 import를 제거
# import magpylib_simulation_m as mag_m
import poledb as PDB
import ast
from scipy.signal import detrend
import pole_auto_analy_command as pole_lib
import argparse
import datetime
import pandas as pd
import json
import slack
from collections import OrderedDict
from pytz import timezone
from datetime import datetime
import maintime  
import time
import os
import traceback
import shutil
import mysqldb
import poleconf
import export_data
PDB.poledb_init()
poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
poledb_conn.connect()
        
#====================================================================
# 1차 분석 // 
#==================================================================== 
def first_anal(group_name_list):
    (nowday, nowhour, nowminute, simplenowday, nowtime) = maintime.maintime()
    simple = simplenowday.split('-')
    ssimplenowday = simple[0] + simple[1] +simple[2]

    with open('./pin.json') as m:
        data = m.read()
        input_conf = json.loads(data)
    IN_Info = pd.json_normalize(input_conf)

    if not os.path.exists("./AUTO_analysis/"+simplenowday+"_result"):
        os.makedirs("./AUTO_analysis/"+simplenowday+"_result")
        
    project_list = ['전북서부B-202405']

    for j in range(len(project_list)):
        slack.slack(str(j+1) + ". " + project_list[j])
        if not os.path.exists("./AUTO_analysis/"+simplenowday+"_result/"+project_list[j]+'/output'):
            os.makedirs("./AUTO_analysis/"+simplenowday+"_result/"+project_list[j]+'/output')

        if not os.path.exists("./AUTO_analysis/"+simplenowday+"_result/"+project_list[j]+'/input'):
            os.makedirs("./AUTO_analysis/"+simplenowday+"_result/"+project_list[j]+'/input')

        if not os.path.exists("./AUTO_analysis/"+simplenowday+"_result/"+project_list[j]+'/output/ref'):
            os.makedirs("./AUTO_analysis/"+simplenowday+"_result/"+project_list[j]+'/output/ref')
        with open('./AUTO_analysis/'+simplenowday+'_result/'+project_list[j]+'/input/'+project_list[j]+'_pole_auto_conf.json', 'w', encoding='utf-8') as make_file:
            json.dump(input_conf, make_file, ensure_ascii = False, indent='\t')
                    
    for k in range(len(project_list)):
        print("프로젝트 " + project_list[k] + " 스캔시작")
        print(project_list[k])
        Pole_list=PDB.get_pole_list(project_list[k])
        print(Pole_list)
        print("프로젝트 검사")

        Pole_list=Pole_list.loc[Pole_list['diagstate']!='-'].reset_index()

        Signal_Analy_result=pole_lib.pole_State_Scan(Pole_list,IN_Info,project_list[k])
        Signal_Analy_result.to_csv("./AUTO_analysis/"+simplenowday+"_result/"+project_list[k]+'/output/Polelist_scan-'+project_list[k]+'_MF_'+ssimplenowday+'.csv') 

        print(project_list[k]+' 스캔결과============')
        print('분석 대상 전주 : '+str(len(Pole_list))+' 개')

        if str(len(Pole_list)) == '0':
            print('분석 대상 전주 없음')
            if os.path.exists("./AUTO_analysis/"+simplenowday+"_result/"+project_list[k]):
                shutil.rmtree("./AUTO_analysis/"+simplenowday+"_result/"+project_list[k])

        else:
            print("프로젝트 " + project_list[k] + " 분석시작")
            result=pole_lib.pole_State_Analysis_all(IN_Info,Pole_list,project_list[k],simplenowday)
            result.to_csv("./AUTO_analysis/"+simplenowday+"_result/"+project_list[k]+'/output/result_detail-'+project_list[k]+'_MF_'+ssimplenowday+'.csv') 
            print("프로젝트 " + project_list[k] + " 분석완료")
                
            
group_name_list = ["경기안산-202208"]
first_anal(group_name_list)
#group_name_list = export_data.groupname_info()
#first_anal(group_name_list)