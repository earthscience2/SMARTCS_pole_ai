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

""" 코드 정보============================
측정관련
- : 미측정
MF : 측정완료
AP : 분석중
AF : 분석완료

측정결과
N : 정상
B : 파단
U : 보류
X : 측정불가

분석관련
NA : 미대상
- : 미분석
AP : 분석중
AF : 분석완료

분석결과
N : 정상
B : 파단
U : 보류
"""
#====================================================================
# 팀이름 정보 가져오기 // list
#====================================================================
def teamname_info():
    poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
    poledb_conn.connect()
    query = "select teamname from tb_team"
    result = poledb_conn.do_select_pd(query)
    poledb_conn.close()
    df = pd.DataFrame(result)
    data_list = df['teamname'].tolist()
    return data_list

#====================================================================
# 그룹이름 정보 가져오기 // list
#====================================================================
def groupname_info():
    poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
    poledb_conn.connect()
    query = "select groupname from tb_pole_group"
    result = poledb_conn.do_select_pd(query)
    poledb_conn.close()
    df = pd.DataFrame(result)
    data_list = df['groupname'].tolist()
    return data_list

#====================================================================
# 해당 그룹의 전주번호 정보 가져오기 // list
#====================================================================
def group_polename_info(groupname):
    poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
    poledb_conn.connect()
    query = f"select * from tb_pole where groupname = '{groupname}'"
    result = poledb_conn.do_select_pd(query)
    poledb_conn.close()
    df = pd.DataFrame(result)
    data_list = df['poleid'].tolist()
    return data_list

#====================================================================
# 해당 그룹의 현재 측정 진행도 가져오기 // dict
#====================================================================
def group_diag_progress_info(groupname):
    poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
    poledb_conn.connect()
    query = f"select * from tb_pole where groupname = '{groupname}'"
    result = poledb_conn.do_select_pd(query)
    poledb_conn.close()
    df = pd.DataFrame(result)
    data_list = df.to_dict('records')
    
    none = 0 # 미측정
    MF = 0 # 측정완료
    AP = 0 # 분석중
    AF = 0 # 분석완료
    el = 0 # 기타
    for i in data_list:
        if i["diagstate"] == "-":
            none = none + 1
        elif i["diagstate"] == "MF":
            MF = MF + 1
        elif i["diagstate"] == "AP":
            AP = AP + 1
        elif i["diagstate"] == "AF":
            AF = AF + 1
        else:
            print("기타 정보 발생")
            el = el + 1
    result = {"total" : len(data_list), "-" : none, "MF" : MF, "AP" : AP, "AF" : AF, "el" : el}
    return result

#====================================================================
# 해당 그룹의 현재 측정 결과 가져오기 // dict
#====================================================================
def group_diag_result_info(groupname):
    poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
    poledb_conn.connect()
    query = f"select * from tb_diag_state where groupname = '{groupname}'"
    result = poledb_conn.do_select_pd(query)
    poledb_conn.close()
    df = pd.DataFrame(result)
    data_list = df.to_dict('records')
    
    N = 0 # 정상
    B = 0 # 파단
    U = 0 # 보류
    X = 0 # 측정불가
    el = 0 # 기타
    for i in data_list:
        if i["breakstate"] == "N":
            N = N + 1
        elif i["breakstate"] == "B":
            B = B + 1
        elif i["breakstate"] == "U":
            U = U + 1
        elif i["breakstate"] == "X":
            X = X + 1
        else:
            print("기타 정보 발생")
            el = el + 1
    result = {"total" : len(data_list), "N" : N, "B" : B, "U" : U, "X" : X, "el" : el}
    return result

#====================================================================
# 해당 그룹의 현재 분석 진행도 가져오기 // dict
#====================================================================
def group_anal_progress_info(groupname):
    poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
    poledb_conn.connect()
    query = f"select * from tb_anal_state where groupname = '{groupname}'"
    result = poledb_conn.do_select_pd(query)
    poledb_conn.close()
    df = pd.DataFrame(result)
    data_list = df.to_dict('records')
    
    none = 0 # 미분석
    anal1 = 0 # 1차분석 완료
    anal2 = 0 # 2차분석 완료
    for i in data_list:
        if i["anal1finyn"] != None:
            anal1 = anal1 + 1
        if i["anal2finyn"] != None:
            anal2 = anal2 + 1
        if i["anal1finyn"] == None and i["anal2finyn"] == None:
            none = none + 1
    result = {"total" : len(data_list), "anal1" : anal1, "anal2" : anal2, "none" : none}
    return result

#====================================================================
# 해당 그룹의 현재 분석 결과 가져오기 // dict
#====================================================================
def group_anal_progress_info(groupname):
    poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
    poledb_conn.connect()
    query = f"select * from tb_anal_state where groupname = '{groupname}'"
    result = poledb_conn.do_select_pd(query)
    poledb_conn.close()
    df = pd.DataFrame(result)
    data_list = df.to_dict('records')
    
    none = 0 # 미분석
    anal1 = 0 # 1차 분석 완료
    anal1_N = 0 # 1차 분석 정상
    anal1_B = 0 # 1차 분석 파단
    anal1_U = 0 # 1차 분석 보류
    anal1_X = 0 # 1차 분석 불가
    anal1_el = 0 # 1차 분석 기타
    anal2 = 0 # 2차 분석 완료
    anal2_N = 0 # 2차 분석 정상
    anal2_B = 0 # 2차 분석 파단
    anal2_U = 0 # 2차 분석 보류
    anal2_X = 0 # 1차 분석 불가
    anal2_el = 0 # 2차 분석 기타
    for i in data_list:
        if i["anal1finyn"] != None:
            anal1 = anal1 + 1
            if i["anal1result"] == "N":
                anal1_N = anal1_N + 1
            elif i["anal1result"] == "B":
                anal1_B = anal1_B + 1
            elif i["anal1result"] == "U":
                anal1_U = anal1_U + 1
            elif i["anal1result"] == "X":
                anal1_X = anal1_X + 1
            else:
                anal1_el = anal1_el + 1
                print(i)
                
        if i["anal2finyn"] != None:
            anal2 = anal2 + 1
            if i["anal2result"] == "N":
                anal2_N = anal2_N + 1
            elif i["anal2result"] == "B":
                anal2_B = anal2_B + 1
            elif i["anal2result"] == "U":
                anal2_U = anal2_U + 1
            elif i["anal2result"] == "X":
                anal2_X = anal2_X + 1
            else:
                anal2_el = anal2_el + 1
                
        if i["anal1finyn"] == None and i["anal2finyn"] == None:
            none = none + 1
            
    result = {"total" : len(data_list), 
              "anal1" : anal1, "anal1_N" : anal1_N, "anal1_B" : anal1_B, "anal1_U" : anal1_U, "anal1_X" : anal1_X, "anal1_el" : anal1_el,
              "anal2" : anal2, "anal2_N" : anal2_N, "anal2_B" : anal2_B, "anal2_U" : anal2_U, "anal2_X" : anal2_X, "anal2_el" : anal2_el,
              "none" : none}
    print(result)
    return result

#====================================================================
# 해당 그룹의 2차 분석 결과 파단or정상("B"or"N") 전주 가져오기 // list
#====================================================================
def group_anal_type_pole(groupname, anal_type):
    poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
    poledb_conn.connect()
    query = f"select * from tb_anal_result where groupname = '{groupname}' and analstep = 2 and breakstate = '{anal_type}'"
    result = poledb_conn.do_select_pd(query)
    poledb_conn.close()
    df = pd.DataFrame(result)
    data_list = df['poleid'].tolist()
    return data_list

#====================================================================
# 해당 전주의 측정 횟수 구하기(measno) 가져오기 // list(dict)
#====================================================================
def pole_add_measno_info(poleid, devicetype):
    re_result = []
    if not poleid == []:
        poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
        poledb_conn.connect()
        for i in poleid:
            query = f"select * from tb_diag_pole_meas_result where poleid = '{i}' and devicetype = '{devicetype}'"
            result = poledb_conn.do_select_pd(query)
            df = pd.DataFrame(result)
            data_list = df.to_dict('records')
            if not data_list == []:
                re_result.append({"poleid" : i, "measno" : data_list, "devicetype" : devicetype, "stdegree" : stdegree} )
        poledb_conn.close()
    return re_result

#====================================================================
# 해당 전주의 상세 데이터 가져오기 // list(dict)
#====================================================================
def pole_detail_data_info(groupname, poleid_measno_list):
    fianl_data_list = []
    x_data = []
    y_data = []
    z_data = []
    if not poleid_measno_list == []:
        poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
        poledb_conn.connect()
        for i in poleid_measno_list:
            for j in i["measno"]:
                for k in ['ch1', 'ch2','ch3','ch4','ch5','ch6','ch7','ch8']:
                    for l in ['x', 'y', 'z']:
                        query = f"select * from tb_diag_pole_meas_data where poleid = '{i['poleid']}' and devicetype = '{i['devicetype']}' and measno = '{j}' and axis = '{l}'"
                        result = poledb_conn.do_select_pd(query)
                        df = pd.DataFrame(result)
                        data_list = df[k].tolist()
                        if l == 'x':
                            x_data = data_list
                        elif l == 'y':
                            y_data = data_list
                        elif l == 'z':
                            z_data = data_list
                        data_list = []
                    print(j, k, l)
                    fianl_data_list.append({"groupname": groupname,
                                            "poleid" : i['poleid'],
                                            "devicetype" : i['devicetype'],
                                            "measno" : j,
                                            "ch" : k,
                                            "x" : x_data, 
                                            "y" : y_data,
                                            "z" : z_data})
    return fianl_data_list

#====================================================================
# 리스트 형식을 csv파일로 변환 // csv
#====================================================================
def list_to_csv(data_list, folder_path, groupname):
    if data_list:
        if all(isinstance(item, dict) for item in data_list):
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, groupname + '.csv')
            
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data_list[0].keys())
                writer.writeheader()
                writer.writerows(data_list)
    else:
        print("데이터 없음")

#=====================================================================
#anal_type = "B"
#groupname = "서울마포용산-202110"
#folder_path = './ai_data_set/'

#result = group_anal_type_pole(groupname, anal_type)
#result = pole_add_measno_info(result, "OUT")
#result = pole_detail_data_info(groupname, result)
#list_to_csv(result, folder_path, groupname)