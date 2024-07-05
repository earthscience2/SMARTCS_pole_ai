import bottleneck as bn
import numpy as np
import pandas as pd
from scipy import signal
import logger
import magpylib_simulation_m as mag_m
import poledb as PDB
import ast
from scipy.signal import detrend
import pole_auto_analy_command as pole_lib
import argparse
import datetime
import pandas as pd
import json
import slack
import name
from collections import OrderedDict
from pytz import timezone
from datetime import datetime
import mysqldb
import poleconf

def pageupgrade(): 
    file_data = OrderedDict()
    PDB.poledb_init()
    poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
    poledb_conn.connect()
    #변수 설정================================================================
    bf_pole ='-'
    n = 0
    m = 0
    p = 0
    f = 0
    no = 0
    br = 0
    f_num = 1
    s_num = 1
    e_num = 1
    n_num = 1
    total_polelistnum = 0
    break_polelistnum = 0
    
    done_polelistnum = 0
    first_polelistnum = 0
    second_polelistnum = 0
    none_polelistnum = 0
    
    f_file_data = {}
    s_file_data = {}
    e_file_data = {}
    n_file_data = {}
    rate_data = {}
    #시간 설정================================================================
    KST = datetime.now(timezone('Asia/Seoul'))
    nowday = KST.strftime("%d")
    nowhour = KST.strftime("%H")
    nowminute = KST.strftime("%M")
    simplenowday = KST.strftime("%Y-%m-%d" )
    nowtime = KST.strftime("%H:%M:%S")
    Pole_list = pd.DataFrame()
    name.find_name()
    sql_str = 'select * from tb_pole'
    result = poledb_conn.do_select_pd(sql_str)
    Pole_list = result
    sql_str = 'select * from tb_anal_result'
    break_file = poledb_conn.do_select_pd(sql_str)
    break_list = break_file 
    
    with open("/workspace/SMART_CS/PAGE/name.json", "r", encoding="utf8") as a:
        contents = a.read()
        name_data = json.loads(contents)
        
    f_file_data["day"] = str(simplenowday)
    f_file_data["time"] = str(nowtime)
    
    for a in range(99):
        f_file_data["f_project_"+str(a+1)] = ''
        f_file_data["f_name_"+str(a+1)] = ''
        f_file_data["f_total_"+str(a+1)] = ''
        f_file_data["f_no_ck_"+str(a+1)] = ''
        f_file_data["f_do_ck_"+str(a+1)] = ''
        f_file_data["f_fi_ck_"+str(a+1)] = ''
        f_file_data["f_se_ck_"+str(a+1)] = ''
        f_file_data["f_ok_ck_"+str(a+1)] = ''
        f_file_data["f_br_ck_"+str(a+1)] = ''
        f_file_data["f_check_"+str(a+1)] = ''
        
        s_file_data["s_project_"+str(a+1)] = ''
        s_file_data["s_name_"+str(a+1)] = ''
        s_file_data["s_total_"+str(a+1)] = ''
        s_file_data["s_no_ck_"+str(a+1)] = ''
        s_file_data["s_do_ck_"+str(a+1)] = ''
        s_file_data["s_fi_ck_"+str(a+1)] = ''
        s_file_data["s_se_ck_"+str(a+1)] = ''
        s_file_data["s_ok_ck_"+str(a+1)] = ''
        s_file_data["s_br_ck_"+str(a+1)] = ''
        s_file_data["s_check_"+str(a+1)] = ''
        
        e_file_data["e_project_"+str(a+1)] = ''
        e_file_data["e_name_"+str(a+1)] = ''
        e_file_data["e_total_"+str(a+1)] = ''
        e_file_data["e_no_ck_"+str(a+1)] = ''
        e_file_data["e_do_ck_"+str(a+1)] = ''
        e_file_data["e_fi_ck_"+str(a+1)] = ''
        e_file_data["e_se_ck_"+str(a+1)] = ''
        e_file_data["e_ok_ck_"+str(a+1)] = ''
        e_file_data["e_br_ck_"+str(a+1)] = ''
        e_file_data["e_check_"+str(a+1)] = ''
        
        n_file_data["n_project_"+str(a+1)] = ''
        n_file_data["n_name_"+str(a+1)] = ''
        n_file_data["n_total_"+str(a+1)] = ''
        n_file_data["n_no_ck_"+str(a+1)] = ''
        n_file_data["n_do_ck_"+str(a+1)] = ''
        n_file_data["n_fi_ck_"+str(a+1)] = ''
        n_file_data["n_se_ck_"+str(a+1)] = ''
        n_file_data["n_ok_ck_"+str(a+1)] = ''
        n_file_data["n_br_ck_"+str(a+1)] = ''
        n_file_data["n_check_"+str(a+1)] = ''

    for i in name_data.keys():
        print(i)
        n = 0
        m = 0
        p = 0
        f = 0
        no = 0
        br = 0
        t_num = 0
        for j in range(len(Pole_list)):
            if Pole_list['officename'][j] == str([i][0]):
                if Pole_list['diagstate'][j] == '-':
                    n = n + 1
                elif Pole_list['diagstate'][j] == 'MF':
                    m = m + 1
                elif Pole_list['diagstate'][j] == 'AP':
                    p = p + 1
                elif Pole_list['diagstate'][j] == 'AF':
                    f = f + 1
                t_num = t_num + 1
                for k in range(len(break_list)):
                    if Pole_list['poleid'][j] == break_list['poleid'][k]:
                        if break_list['breakstate'][k] == 'B':
                            if bf_pole != break_list['poleid'][k]:
                                br = br + 1
                                bf_pole = break_list['poleid'][k]
        no = m+p+f-br
        #print(t_num)
        if t_num > 0:      
            if t_num == f:
                e_file_data["e_project_"+str(e_num)] = str(name_data[i])
                e_file_data["e_name_"+str(e_num)] = str([i][0])
                e_file_data["e_check_"+str(e_num)] = "완료"
                e_file_data["e_total_"+str(e_num)] = str(t_num)
                e_file_data["e_no_ck_"+str(e_num)] = str(n)
                e_file_data["e_do_ck_"+str(e_num)] = str(m+p+f)
                e_file_data["e_fi_ck_"+str(e_num)] = str(p)
                e_file_data["e_se_ck_"+str(e_num)] = str(f)
                e_file_data["e_ok_ck_"+str(e_num)] = str(no)
                e_file_data["e_br_ck_"+str(e_num)] = str(br)
                e_num = e_num + 1
                done_polelistnum = done_polelistnum + 1
                break_polelistnum = break_polelistnum + br
                total_polelistnum = total_polelistnum + t_num
            elif t_num != p and f > 0:
                s_file_data["s_project_"+str(s_num)] = str(name_data[i])
                s_file_data["s_name_"+str(s_num)] = str([i][0])
                s_file_data["s_check_"+str(s_num)] = "2차 분석중"
                s_file_data["s_total_"+str(s_num)] = str(t_num)
                s_file_data["s_no_ck_"+str(s_num)] = str(n)
                s_file_data["s_do_ck_"+str(s_num)] = str(m+p+f)
                s_file_data["s_fi_ck_"+str(s_num)] = str(p)
                s_file_data["s_se_ck_"+str(s_num)] = str(f)
                s_file_data["s_ok_ck_"+str(s_num)] = str(no)
                s_file_data["s_br_ck_"+str(s_num)] = str(br)
                s_num = s_num + 1
                second_polelistnum = second_polelistnum + 1
            elif f == 0 and t_num == p:
                s_file_data["s_project_"+str(s_num)] = str(name_data[i])
                s_file_data["s_name_"+str(s_num)] = str([i][0])
                s_file_data["s_check_"+str(s_num)] = "2차 분석 대기중"
                s_file_data["s_total_"+str(s_num)] = str(t_num)
                s_file_data["s_no_ck_"+str(s_num)] = str(n)
                s_file_data["s_do_ck_"+str(s_num)] = str(m+p+f)
                s_file_data["s_fi_ck_"+str(s_num)] = str(p)
                s_file_data["s_se_ck_"+str(s_num)] = str(f)
                s_file_data["s_ok_ck_"+str(s_num)] = str(no)
                s_file_data["s_br_ck_"+str(s_num)] = str(br)
                s_num = s_num + 1
                second_polelistnum = second_polelistnum + 1
            elif t_num != p and p > 0:
                f_file_data["f_project_"+str(f_num)] = str(name_data[i])
                f_file_data["f_name_"+str(f_num)] = str([i][0])
                f_file_data["f_check_"+str(f_num)] = "1차 분석중"
                f_file_data["f_total_"+str(f_num)] = str(t_num)
                f_file_data["f_no_ck_"+str(f_num)] = str(n)
                f_file_data["f_do_ck_"+str(f_num)] = str(m+p+f)
                f_file_data["f_fi_ck_"+str(f_num)] = str(p)
                f_file_data["f_se_ck_"+str(f_num)] = str(f)
                f_file_data["f_ok_ck_"+str(f_num)] = str(no)
                f_file_data["f_br_ck_"+str(f_num)] = str(br)
                f_num = f_num + 1
                first_polelistnum = first_polelistnum + 1
            elif p == 0 and m > 0:
                f_file_data["f_project_"+str(f_num)] = str(name_data[i])
                f_file_data["f_name_"+str(f_num)] = str([i][0])
                f_file_data["f_check_"+str(f_num)] = "1차 분석 대기중"
                f_file_data["f_total_"+str(f_num)] = str(t_num)
                f_file_data["f_no_ck_"+str(f_num)] = str(n)
                f_file_data["f_do_ck_"+str(f_num)] = str(m+p+f)
                f_file_data["f_fi_ck_"+str(f_num)] = str(p)
                f_file_data["f_se_ck_"+str(f_num)] = str(f)
                f_file_data["f_ok_ck_"+str(f_num)] = str(no)
                f_file_data["f_br_ck_"+str(f_num)] = str(br)
                f_num = f_num + 1
                first_polelistnum = first_polelistnum + 1
            elif t_num == n:
                n_file_data["n_project_"+str(n_num)] = str(name_data[i])
                n_file_data["n_name_"+str(n_num)] = str([i][0])
                n_file_data["n_check_"+str(n_num)] = "미측정"
                n_file_data["n_total_"+str(n_num)] = str(t_num)
                n_file_data["n_no_ck_"+str(n_num)] = str(n)
                n_file_data["n_do_ck_"+str(n_num)] = str(m+p+f)
                n_file_data["n_fi_ck_"+str(n_num)] = str(p)
                n_file_data["n_se_ck_"+str(n_num)] = str(f)
                n_file_data["n_ok_ck_"+str(n_num)] = str(no)
                n_file_data["n_br_ck_"+str(n_num)] = str(br)
                n_num = n_num + 1
                none_polelistnum = none_polelistnum + 1
            else:
                n_file_data["n_project_"+str(n_num)] = str(name_data[i])
                n_file_data["n_name_"+str(n_num)] = str([i][0])
                n_file_data["n_check_"+str(n_num)] = "기타"
                n_file_data["n_total_"+str(n_num)] = str(t_num)
                n_file_data["n_no_ck_"+str(n_num)] = str(n)
                n_file_data["n_do_ck_"+str(n_num)] = str(m+p+f)
                n_file_data["n_fi_ck_"+str(n_num)] = str(p)
                n_file_data["n_se_ck_"+str(n_num)] = str(f)
                n_file_data["n_ok_ck_"+str(n_num)] = str(no)
                n_file_data["n_br_ck_"+str(n_num)] = str(br)
                n_num = n_num + 1
                
    rate_data["total_project"] = str(len(name_data))
    
    if total_polelistnum == 0:
        rate_data["break_rate"] = "0.00"
    else:
        rate_data["break_rate"] = "%.2f" % ((break_polelistnum / total_polelistnum)*100)
        
    rate_data["done_project"] = str(done_polelistnum)
    
    if done_polelistnum == 0:
        rate_data["done_project_rate"] = "0.00"
    else:
        rate_data["done_project_rate"] = "%.2f" % ((done_polelistnum / (done_polelistnum + second_polelistnum + first_polelistnum + none_polelistnum))*100)
        
    rate_data["first_project"] = str(first_polelistnum)
    
    if first_polelistnum == 0:
        rate_data["first_project_rate"] = "0.00"
    else:
        rate_data["first_project_rate"] = "%.2f" % ((first_polelistnum / (done_polelistnum + second_polelistnum + first_polelistnum + none_polelistnum))*100)
        
        
    rate_data["second_project"] = str(second_polelistnum)
    
    if second_polelistnum == 0:
        rate_data["second_project_rate"] = "0.00"
    else:
        rate_data["second_project_rate"] = "%.2f" % ((second_polelistnum / (done_polelistnum + second_polelistnum + first_polelistnum + none_polelistnum))*100)
        
    rate_data["none_project"] = str(none_polelistnum)
    
    if none_polelistnum == 0:
        rate_data["none_project_rate"] = "0.00"
    else:
        rate_data["none_project_rate"] = "%.2f" % ((none_polelistnum / (done_polelistnum + second_polelistnum + first_polelistnum + none_polelistnum))*100)
        
    with open('/workspace/SMART_CS/PAGE/f_data.json', 'w', encoding='utf-8') as make_file:
        json.dump(f_file_data, make_file, ensure_ascii = False, indent='\t')
        
    with open('/workspace/SMART_CS/PAGE/s_data.json', 'w', encoding='utf-8') as make_file:
        json.dump(s_file_data, make_file, ensure_ascii = False, indent='\t')
        
    with open('/workspace/SMART_CS/PAGE/e_data.json', 'w', encoding='utf-8') as make_file:
        json.dump(e_file_data, make_file, ensure_ascii = False, indent='\t')
        
    with open('/workspace/SMART_CS/PAGE/n_data.json', 'w', encoding='utf-8') as make_file:
        json.dump(n_file_data, make_file, ensure_ascii = False, indent='\t')  
        
    with open('/workspace/SMART_CS/PAGE/rate.json', 'w', encoding='utf-8') as make_file:
        json.dump(rate_data, make_file, ensure_ascii = False, indent='\t')
        
    slack.slack("전주 업데이트 완료")