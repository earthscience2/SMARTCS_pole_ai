#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:55:17 2021

@author: heegulee
"""
import numpy as np
import pandas as pd
import os
from scipy import signal
import poledb as PDB
import ast
import mysqldb
from scipy.signal import detrend
PDB.poledb_init()
import openpyxl
import poleconf
import time
import csv
import slack

#====================================================================
# 해당 그룹의 2차 분석 종료 후 파단되 전주 가져오기  // list
#====================================================================
def make_ai_break_data_set(dir):
    poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
    poledb_conn.connect()
    query = "select * from tb_pole"
    result = poledb_conn.do_select_pd(query)
    
    pole_list = []
    initial_data_list = []
    middle_data_list = []
    final_data_list = []
    
    for i in range(len(result)):
        if result['groupname'][i] == dir:
            pole_list.append(str(result['poleid'][i]))
            
    if pole_list == []:
        print(dir + " 프로젝트는 존재하지 않습니다.")
        return
    
    # 해당 프로젝트의 2차 분석된 전주 정보 수집 
    else:
        print(dir + " 프로젝트 기본 데이터 수집중")
        num = 0
        for k in pole_list:
            num += 1
            per = (num / len(pole_list)) * 100 
            print(dir + " 프로젝트 기본 데이터 수집중 " + str(round(per, 2)) + "%")
            query = f"select * from tb_anal_result where poleid = '{k}'"
            result = poledb_conn.do_select_pd(query)
            
            for j in range(len(result)):
                analstep = str(result['analstep'][j])
                if analstep == '2':
                    initial_data_list.append({"breakstate": result['breakstate'][j],
                                              "breakheight": result['breakheight'][j],
                                              "breakdegree": result['breakdegree'][j],
                                              "groupname": result['groupname'][j],
                                              "poleid": result['poleid'][j]})    
    
        if initial_data_list == []:
            print(dir + " 프로젝트에서 2차 분석된 전주가 존재하지 않습니다.")
            return
            
        # 2차 분석된 전주의 데이터 수집
        else:
            num = 0
            for i in initial_data_list:
                num += 1
                per = (num / len(initial_data_list)) * 100 
                print(dir + " 프로젝트 중간 데이터 수집중 " + str(round(per, 2)) + "%")
                poleid = i['poleid']
                
                query = f"select * from tb_diag_pole_meas_data where poleid = '{poleid}'"
                result = poledb_conn.do_select_pd(query)
                
                old_devicetype = ""
                old_measno = ""
                
                for j in range(len(result)):
                    if result['devicetype'][j] != old_devicetype or result['measno'][j] != old_measno:
                        middle_data_list.append({"breakstate": i['breakstate'],
                                                 "groupname": i['groupname'],
                                                 "poleid": i['poleid'],
                                                 "devicetype": result['devicetype'][j],
                                                 "measno": result['measno'][j],
                                                 "breakheight": i['breakheight'],
                                                 "breakdegree": i['breakdegree']})
                        old_devicetype = result['devicetype'][j]
                        old_measno = result['measno'][j]
                
        # 2차 분석된 전주의 추가 데이터 수집 
        print(dir + " 프로젝트 마지막 데이터 수집시작")
        num = 0
        for i in middle_data_list:
            num += 1
            per = (num / len(middle_data_list)) * 100 
            print(dir + " 프로젝트 마지막 데이터 수집중 " + str(round(per, 2)) + "%")
            poleid = i['poleid']
            devicetype = i['devicetype']
            measno = i['measno']
            
            query = f"select * from tb_diag_pole_meas_result where poleid = '{poleid}' and devicetype = '{devicetype}' and measno = '{measno}'" 
            breakstate_result = poledb_conn.do_select_pd(query)
            for j in range(len(breakstate_result)):
                breakstate = i['breakstate']
                stdegree = breakstate_result['stdegree'][j]
                eddegree = breakstate_result['eddegree'][j]
                stheight = breakstate_result['stheight'][j]
                edheight = breakstate_result['edheight'][j]
            
            for j in ['ch1', 'ch2','ch3','ch4','ch5','ch6','ch7','ch8']:
                if i['devicetype'] == 'OUT' and breakstate == 'B' and i['breakdegree'] != None:
                    ch_num = (float(i['breakdegree']) - stdegree) / 10
                    try:
                        ch_num = round(ch_num, 0)
                        ch_num = int(ch_num)
                        ch_str = 'ch' + str(ch_num)
                    except:
                        ch_str = 'ch0'
                    for k in ['x', 'y', 'z']:
                        query = f"select * from tb_diag_pole_meas_data where poleid = '{poleid}' and devicetype = '{devicetype}' and measno = '{measno}' and axis = '{k}'" 
                        result = poledb_conn.do_select_pd(query)
                        re_result = result['ch1']
                        if k == 'x':
                            x_data = list(re_result)
                        elif k == 'y':
                            y_data = list(re_result)
                        elif k == 'z':
                            z_data = list(re_result)

                        result = None
                    if breakstate == 'B':
                        final_data_list.append({"breakstate": breakstate,
                                                "groupname": i['groupname'],
                                                "poleid": i['poleid'],
                                                "devicetype": i['devicetype'],
                                                "measno": i['measno'],
                                                "breakheight": i['breakheight'],
                                                "breakdegree": i['breakdegree'],
                                                "ch": j,
                                                "x": x_data, 
                                                "y": y_data,
                                                "z": z_data})
    return final_data_list

#====================================================================
# 해당 그룹의 2차 분석 and 후크형 and 파단 데이터 추출 // list
#====================================================================
def make_ai_break_data_set(dir):
    poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
    poledb_conn.connect()
    query = "select * from tb_pole"
    result = poledb_conn.do_select_pd(query)
    
    pole_list = []
    initial_data_list = []
    middle_data_list = []
    final_data_list = []
    
    for i in range(len(result)):
        if result['groupname'][i] == dir:
            pole_list.append(str(result['poleid'][i]))
            
    if pole_list == []:
        print(dir + " 프로젝트는 존재하지 않습니다.")
        return
    
    # 해당 프로젝트의 2차 분석된 전주 정보 수집 
    else:
        print(dir + " 프로젝트 기본 데이터 수집중")
        num = 0
        for k in pole_list:
            num += 1
            per = (num / len(pole_list)) * 100 
            print(dir + " 프로젝트 기본 데이터 수집중 " + str(round(per, 2)) + "%")
            query = f"select * from tb_anal_result where poleid = '{k}'"
            result = poledb_conn.do_select_pd(query)
            
            for j in range(len(result)):
                analstep = str(result['analstep'][j])
                if analstep == '2':
                    initial_data_list.append({"breakstate": result['breakstate'][j],
                                              "breakheight": result['breakheight'][j],
                                              "breakdegree": result['breakdegree'][j],
                                              "groupname": result['groupname'][j],
                                              "poleid": result['poleid'][j]})    
    
        if initial_data_list == []:
            print(dir + " 프로젝트에서 2차 분석된 전주가 존재하지 않습니다.")
            return
            
        # 2차 분석된 전주의 데이터 수집
        else:
            num = 0
            for i in initial_data_list:
                num += 1
                per = (num / len(initial_data_list)) * 100 
                print(dir + " 프로젝트 중간 데이터 수집중 " + str(round(per, 2)) + "%")
                poleid = i['poleid']
                
                query = f"select * from tb_diag_pole_meas_data where poleid = '{poleid}'"
                result = poledb_conn.do_select_pd(query)
                
                old_devicetype = ""
                old_measno = ""
                
                for j in range(len(result)):
                    if result['devicetype'][j] != old_devicetype or result['measno'][j] != old_measno:
                        middle_data_list.append({"breakstate": i['breakstate'],
                                                 "groupname": i['groupname'],
                                                 "poleid": i['poleid'],
                                                 "devicetype": result['devicetype'][j],
                                                 "measno": result['measno'][j],
                                                 "breakheight": i['breakheight'],
                                                 "breakdegree": i['breakdegree']})
                        old_devicetype = result['devicetype'][j]
                        old_measno = result['measno'][j]
                
        # 2차 분석된 전주의 추가 데이터 수집 
        print(dir + " 프로젝트 마지막 데이터 수집시작")
        num = 0
        for i in middle_data_list:
            num += 1
            per = (num / len(middle_data_list)) * 100 
            print(dir + " 프로젝트 마지막 데이터 수집중 " + str(round(per, 2)) + "%")
            poleid = i['poleid']
            devicetype = i['devicetype']
            measno = i['measno']
            
            query = f"select * from tb_diag_pole_meas_result where poleid = '{poleid}' and devicetype = '{devicetype}' and measno = '{measno}'" 
            breakstate_result = poledb_conn.do_select_pd(query)
            for j in range(len(breakstate_result)):
                breakstate = i['breakstate']
                stdegree = breakstate_result['stdegree'][j]
                eddegree = breakstate_result['eddegree'][j]
                stheight = breakstate_result['stheight'][j]
                edheight = breakstate_result['edheight'][j]
            
            for j in ['ch1', 'ch2','ch3','ch4','ch5','ch6','ch7','ch8']:
                if i['devicetype'] == 'OUT' and breakstate == 'B' and i['breakdegree'] != None:
                    ch_num = (float(i['breakdegree']) - stdegree) / 10
                    try:
                        ch_num = round(ch_num, 0)
                        ch_num = int(ch_num)
                        ch_str = 'ch' + str(ch_num)
                    except:
                        ch_str = 'ch0'
                    for k in ['x', 'y', 'z']:
                        query = f"select * from tb_diag_pole_meas_data where poleid = '{poleid}' and devicetype = '{devicetype}' and measno = '{measno}' and axis = '{k}'" 
                        result = poledb_conn.do_select_pd(query)
                        re_result = result['ch1']
                        if k == 'x':
                            x_data = list(re_result)
                        elif k == 'y':
                            y_data = list(re_result)
                        elif k == 'z':
                            z_data = list(re_result)

                        result = None
                    print(breakstate, j, ch_str)
                    if breakstate == 'B' and j == ch_str:
                        print("hello")
                        final_data_list.append({"breakstate": breakstate,
                                                "groupname": i['groupname'],
                                                "poleid": i['poleid'],
                                                "devicetype": i['devicetype'],
                                                "measno": i['measno'],
                                                "breakheight": i['breakheight'],
                                                "breakdegree": i['breakdegree'],
                                                "ch": j,
                                                "x": x_data, 
                                                "y": y_data,
                                                "z": z_data})
    return final_data_list
        
#====================================================================
# 리스트 형식을 csv파일로 변환  // csv
#====================================================================
def list_to_csv(list, folder_path, dir):
    if list:
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, dir + '.csv')
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list[0].keys())
            writer.writeheader()
            writer.writerows(list)
    else:
        print("데이터 없음")
    
#=====================================================================
"""
N : 정상
B : 파단
U : 보류
"""
folder_path = './ai_data_set/'
groupname = "서울마포용산-202110"

result = make_ai_break_data_set(groupname)
#result = make_ai_normal_data_set(groupname)
list_to_csv(result, folder_path, groupname)

    