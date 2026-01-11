from . import mysqldb
from . import poleconf
import logger
import pandas as pd
import os
import csv

logger = logger.get_logger()
poledb_conn = None

# diagstate
#  -	미측정
# MF	측정완료
# AP	분석중
# AF	분석완료​
# breatestate
# N	정상
# B	파단
# U	보류
# X	측정불가
# -	없음

#====================================================================
# 데이터베이스 접속
#====================================================================
def poledb_init(server):
    global poledb_conn
    poledb_host, poledb_dbname, poledb_user, poledb_pwd = poleconf.db(server)
    poledb_conn = mysqldb.Mysqlhandler(poledb_host, poledb_user, poledb_pwd, poledb_dbname)
    poledb_conn.connect()

#====================================================================
# 데이터베이스 접속확인
#====================================================================
def ping():
    poledb_conn.do_sql('select now()')
    logger.info('[DB] ping success')
    return

#====================================================================
# 측정데이터 가져오기
#====================================================================
def get_meas_data(poleid, measno, devicetype, axis):
    data = [ poleid, measno, devicetype, axis ]
    sql_str = 'SELECT * FROM tb_diag_pole_meas_data WHERE poleid=%s and measno=%s and devicetype=%s and axis=%s order by idx;'
    #logger.info('sql_str={} data={}'.format(sql_str, data))
    result = None
    try:
        result = poledb_conn.do_select_pd(sql_str, data)
    except Exception as e:
        logger.error(str(e))
        result = None

    return result

#====================================================================
# 측정데이터 결과데이터 가져오기
#====================================================================
def get_meas_result(poleid, dtype):
    data = [ poleid, dtype ]
    sql_str = 'SELECT * FROM tb_diag_pole_meas_result WHERE poleid=%s and devicetype=%s order by idx;'
    #logger.info('sql_str={} data={}'.format(sql_str, data))
    result = None
    try:
        result = poledb_conn.do_select_pd(sql_str, data)
    except Exception as e:
        logger.error(str(e))
        result = None

    return result

#====================================================================
# 측정데이터 결과 수 가져오기
#====================================================================
def get_meas_result_count(poleid):
    data = [ poleid ]
    sql_str = 'select devicetype, count(*) as cnt from tb_diag_pole_meas_result tdpmr  where poleid=%s group by devicetype;'
    #logger.info('sql_str={} data={}'.format(sql_str, data))
    result = None
    try:
        result = poledb_conn.do_select_pd(sql_str, data)
    except Exception as e:
        logger.error(str(e))
        result = None

    return result

#====================================================================
# 해당 그룹의 전주목록(특정정보) 가져오기
#====================================================================
def get_pole_list(group_name):
    data = [ group_name ]
    sql_str = 'select tp.poleid, tp.regdate, tp.diagstate, tds.endtime, tds.breakstate , tds.teamid from tb_pole tp join tb_diag_state tds on tds.poleid = tp.poleid where tds.groupname=%s and diagstate in ("MF", "AP", "AF");'
    #logger.info('sql_str={} data={}'.format(sql_str, data))
    result = None
    try:
        result = poledb_conn.do_select_pd(sql_str, data)
    except Exception as e:
        logger.error(str(e))
        result = None

    return result

#====================================================================
# 해당 그룹의 전주목록(전체) 가져오기
#====================================================================
def get_pole_list_a(group_name):
    data = [ group_name ]
    print(data)
    sql_str = 'select * from tb_pole where groupname=%s;'
    #logger.info('sql_str={} data={}'.format(sql_str, data))
    result = None
    try:
        result = poledb_conn.do_select_pd(sql_str, data)
    except Exception as e:
        logger.error(str(e))
        result = None

    return result

#====================================================================
# 전주목록(전체) 가져오기
#====================================================================
def get_pole_list_all():
    sql_str = 'select * from tb_pole'
    result = None
    try:
        result = poledb_conn.do_select_pd(sql_str)
    except Exception as e:
        logger.error(str(e))
        result = None
    print(result)

    return result

#====================================================================
# 해당 그룹의 현재 측정 결과 가져오기 // dict
#====================================================================
def group_diag_result_info_2(groupname):
    query = f"select * from tb_diag_state where groupname = '{groupname}'"
    result = None
    try:
        result = poledb_conn.do_select_pd(query)
    except Exception as e:
        logger.error(str(e))
        result = None
    df = pd.DataFrame(result)
    data_list = df.to_dict('records')
    return data_list


#====================================================================
# 팀이름 정보 가져오기 // list
#====================================================================
def teamname_info():
    query = "select teamname from tb_team"
    result = None
    try:
        result = poledb_conn.do_select_pd(query)
    except Exception as e:
        logger.error(str(e))
        result = None
    df = pd.DataFrame(result)
    data_list = df['teamname'].tolist()
    return data_list

#====================================================================
# 그룹이름 정보 가져오기 // list
#====================================================================
def groupname_info():
    query = 'select groupname from tb_pole_group'
    result = None
    try:
        result = poledb_conn.do_select_pd(query)
    except Exception as e:
        logger.error(str(e))
        result = None
    df = pd.DataFrame(result)
    data_list = df['groupname'].tolist()
    return data_list
        
#====================================================================
# 해당 그룹의 전주번호 정보 가져오기 // list
#====================================================================
def group_polename_info(groupname):
    query = f"select * from tb_pole where groupname = '{groupname}'"
    result = None
    try:
        result = poledb_conn.do_select_pd(query)
    except Exception as e:
        logger.error(str(e))
        result = None
    df = pd.DataFrame(result)
    data_list = df['poleid'].tolist()
    return data_list

#====================================================================
# 해당 그룹의 전주번호 정보 가져오기 // list
#====================================================================
def group_polename_info_2(groupname):
    query = f"select * from tb_pole where groupname = '{groupname}'"
    result = None
    try:
        result = poledb_conn.do_select_pd(query)
    except Exception as e:
        logger.error(str(e))
        result = None
    df = pd.DataFrame(result)
    data_list = df.to_dict('records')
    return data_list

#====================================================================
# 해당 그룹의 현재 측정 결과 가져오기 // dict
#====================================================================
def group_diag_result_info_2(groupname):
    query = f"select * from tb_diag_state where groupname = '{groupname}'"
    result = None
    try:
        result = poledb_conn.do_select_pd(query)
    except Exception as e:
        logger.error(str(e))
        result = None
    df = pd.DataFrame(result)
    data_list = df.to_dict('records')
    return data_list


#====================================================================
# 해당 그룹의 현재 측정 진행도 가져오기 // dict
#====================================================================
def group_diag_progress_info(groupname):
    query = f"select * from tb_pole where groupname = '{groupname}'"
    result = None
    try:
        result = poledb_conn.do_select_pd(query)
    except Exception as e:
        logger.error(str(e))
        result = None
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
    query = f"select * from tb_diag_state where groupname = '{groupname}'"
    result = None
    try:
        result = poledb_conn.do_select_pd(query)
    except Exception as e:
        logger.error(str(e))
        result = None
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
    query = f"select * from tb_anal_state where groupname = '{groupname}'"
    result = None
    try:
        result = poledb_conn.do_select_pd(query)
    except Exception as e:
        logger.error(str(e))
        result = None
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
    query = f"select * from tb_anal_state where groupname = '{groupname}'"
    result = None
    try:
        result = poledb_conn.do_select_pd(query)
    except Exception as e:
        logger.error(str(e))
        result = None
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
    query = f"select * from tb_anal_result where groupname = '{groupname}' and analstep = 2 and breakstate = '{anal_type}'"
    result = None
    try:
        result = poledb_conn.do_select_pd(query)
    except Exception as e:
        logger.error(str(e))
        result = None
    df = pd.DataFrame(result)
    if not df.empty:
        data_list = df['poleid'].tolist()
    else:
        data_list = []
    return data_list

#====================================================================
# 해당 그룹의 2차 분석 결과 파단or정상("B"or"N") 전주 가져오기 // list
#====================================================================
def group_anal_type_pole_2(groupname, anal_type):
    query = f"select * from tb_anal_result where groupname = '{groupname}' and analstep = 2 and breakstate = '{anal_type}'"
    result = None
    try:
        result = poledb_conn.do_select_pd(query)
    except Exception as e:
        logger.error(str(e))
        result = None
    df = pd.DataFrame(result)
    data_list = df.to_dict('records')
    return data_list

#====================================================================
# 해당 전주의 측정 횟수 구하기(measno) 가져오기 // list(dict)
#====================================================================
def pole_add_measno_info(poleid, devicetype):
    re_result = []
    if not poleid == []:
        for i in poleid:
            query = f"select * from tb_diag_pole_meas_result where poleid = '{i}' and devicetype = '{devicetype}'"
            result = None
            try:
                result = poledb_conn.do_select_pd(query)
            except Exception as e:
                logger.error(str(e))
                result = None
            df = pd.DataFrame(result)

            data_list = df['poleid'].tolist()
            if not data_list == []:
                re_result.append({"poleid" : i, "measno" : data_list, "devicetype" : devicetype} )

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
        for i in poleid_measno_list:
            for j in i["measno"]:
                for k in ['ch1', 'ch2','ch3','ch4','ch5','ch6','ch7','ch8']:
                    for l in ['x', 'y', 'z']:
                        #query = f"select * from tb_diag_pole_meas_data where poleid = '{i["poleid"]}' and devicetype = '{i["devicetype"]}' and measno = '{j}' and axis = '{l}'" 
                        query = f"select * from tb_diag_pole_meas_data where poleid = '{i['poleid']}' and devicetype = '{i['devicetype']}' and measno = '{j}' and axis = '{l}'"
                        result = None
                        try:
                            result = poledb_conn.do_select_pd(query)
                        except Exception as e:
                            logger.error(str(e))
                            result = None
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
