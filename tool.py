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
PDB.poledb_init()
poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
poledb_conn.connect()

def find_group():
    file_data = OrderedDict()
    gropnames = []
    gropname = []
    grop = []
    project_list = []
    query = "select * from tb_pole"
    bf_name = '-'
    check = 0

    result = poledb_conn.do_select_pd(query)

    for i in range(len(result)):
        gropnames.append(result['groupname'][i])
        gropnames.sort()

    for j in range(len(gropnames)):
        part_name = gropnames[j]

        if part_name != bf_name:
            gropname.append(gropnames[j])
            bf_name = part_name

    gropname.sort()
    gropname = gropname[0:-3]
    for o in range(len(gropname)):
        part = gropname[o].split('-')
        if len(part[1]) == 6:
            project_list.append([gropname[o], part[1]])

    project_list = sorted(project_list, key=lambda x:x[1])

    for m in range(len(project_list)):
        file_data['gropname'+str(m+1)] = project_list[len(project_list) - m - 1][0]

    with open('/workspace/SMART_CS/PAGE/grop_data.json', 'w', encoding='utf-8') as make_file:
        json.dump(file_data, make_file, ensure_ascii = False, indent='\t')   
        
def find_name():
    bf_name = '-'
    name_data = {}
    with open("/workspace/SMART_CS/PAGE/grop_data.json", "r", encoding="utf8") as f:
            contents = f.read()
            json_data = json.loads(contents)

    poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
    poledb_conn.connect()

    sql_str = 'select * from tb_pole'

    result = poledb_conn.do_select_pd(sql_str)

    for i in range(len(json_data)):

        for j in range(len(result['poleid'])):

            if result['groupname'][j] == json_data["gropname"+str(i+1)]:

                if bf_name != result['officename'][j]:

                    name_data[result['officename'][j]] = result['groupname'][j]
                    bf_name = result['officename'][j]

    with open('/workspace/SMART_CS/PAGE/office_data.json', 'w', encoding='utf-8') as make_file:
        json.dump(name_data, make_file, ensure_ascii = False, indent='\t')
        
def upgrade():
    KST = datetime.now(timezone('Asia/Seoul'))
    nowday = KST.strftime("%d")
    nowhour = KST.strftime("%H")
    nowminute = KST.strftime("%M")
    simplenowday = KST.strftime("%Y-%m-%d" )
    nowtime = KST.strftime("%H:%M:%S")
    Pole_list = pd.DataFrame()
    #변수 설정================================================================
    bf_pole ='-'
    n = 0
    m = 0
    p = 0
    f = 0
    no = 0
    br = 0
    one_num = 1
    two_num = 1
    done_num = 1
    none_num = 1
    miss_num = 1
    other_num = 1
    list_num = 0
    total_polelistnum = 0
    break_polelistnum = 0
    
    done_polelistnum = 0
    first_polelistnum = 0
    second_polelistnum = 0
    none_polelistnum = 0
    miss_polelistnum = 0
    
    Pole_list = pd.DataFrame()
    sql_str = 'select * from tb_pole'
    result = poledb_conn.do_select_pd(sql_str)
    Pole_list = result
    sql_str = 'select * from tb_anal_result'
    break_file = poledb_conn.do_select_pd(sql_str)
    break_list = break_file 
    miss_data = {}
    none_data = {}
    one_data = {}
    two_data = {}
    done_data = {}
    other_data = {}
        
    miss_data["day"] = str(simplenowday)
    miss_data["time"] = str(nowtime)
    
    for a in range(99):
        miss_data["miss_project_"+str(a+1)] = ''
        miss_data["miss_name_"+str(a+1)] = ''
        miss_data["miss_total_"+str(a+1)] = ''
        miss_data["miss_no_ck_"+str(a+1)] = ''
        miss_data["miss_do_ck_"+str(a+1)] = ''
        miss_data["miss_fi_ck_"+str(a+1)] = ''
        miss_data["miss_se_ck_"+str(a+1)] = ''
        miss_data["miss_ok_ck_"+str(a+1)] = ''
        miss_data["miss_br_ck_"+str(a+1)] = ''
        miss_data["miss_check_"+str(a+1)] = ''
        
        none_data["none_project_"+str(a+1)] = ''
        none_data["none_name_"+str(a+1)] = ''
        none_data["none_total_"+str(a+1)] = ''
        none_data["none_no_ck_"+str(a+1)] = ''
        none_data["none_do_ck_"+str(a+1)] = ''
        none_data["none_fi_ck_"+str(a+1)] = ''
        none_data["none_se_ck_"+str(a+1)] = ''
        none_data["none_ok_ck_"+str(a+1)] = ''
        none_data["none_br_ck_"+str(a+1)] = ''
        none_data["none_check_"+str(a+1)] = ''
        
        one_data["one_project_"+str(a+1)] = ''
        one_data["one_name_"+str(a+1)] = ''
        one_data["one_total_"+str(a+1)] = ''
        one_data["one_no_ck_"+str(a+1)] = ''
        one_data["one_do_ck_"+str(a+1)] = ''
        one_data["one_fi_ck_"+str(a+1)] = ''
        one_data["one_se_ck_"+str(a+1)] = ''
        one_data["one_ok_ck_"+str(a+1)] = ''
        one_data["one_br_ck_"+str(a+1)] = ''
        one_data["one_check_"+str(a+1)] = ''
        
        two_data["two_project_"+str(a+1)] = ''
        two_data["two_name_"+str(a+1)] = ''
        two_data["two_total_"+str(a+1)] = ''
        two_data["two_no_ck_"+str(a+1)] = ''
        two_data["two_do_ck_"+str(a+1)] = ''
        two_data["two_fi_ck_"+str(a+1)] = ''
        two_data["two_se_ck_"+str(a+1)] = ''
        two_data["two_ok_ck_"+str(a+1)] = ''
        two_data["two_br_ck_"+str(a+1)] = ''
        two_data["two_check_"+str(a+1)] = ''
        
        done_data["done_project_"+str(a+1)] = ''
        done_data["done_name_"+str(a+1)] = ''
        done_data["done_total_"+str(a+1)] = ''
        done_data["done_no_ck_"+str(a+1)] = ''
        done_data["done_do_ck_"+str(a+1)] = ''
        done_data["done_fi_ck_"+str(a+1)] = ''
        done_data["done_se_ck_"+str(a+1)] = ''
        done_data["done_ok_ck_"+str(a+1)] = ''
        done_data["done_br_ck_"+str(a+1)] = ''
        done_data["done_check_"+str(a+1)] = ''
        
        other_data["other_project_"+str(a+1)] = ''
        other_data["other_name_"+str(a+1)] = ''
        other_data["other_total_"+str(a+1)] = ''
        other_data["other_no_ck_"+str(a+1)] = ''
        other_data["other_do_ck_"+str(a+1)] = ''
        other_data["other_fi_ck_"+str(a+1)] = ''
        other_data["other_se_ck_"+str(a+1)] = ''
        other_data["other_ok_ck_"+str(a+1)] = ''
        other_data["other_br_ck_"+str(a+1)] = ''
        other_data["other_check_"+str(a+1)] = ''

    with open("/workspace/SMART_CS/PAGE/none_check_data.json", "r", encoding="utf8") as b:
        contents = b.read()
        none_check_data = json.loads(contents)

    with open("/workspace/SMART_CS/PAGE/one_check_data.json", "r", encoding="utf8") as c:
        contents = c.read()
        one_check_data = json.loads(contents)

    with open("/workspace/SMART_CS/PAGE/two_check_data.json", "r", encoding="utf8") as d:
        contents = d.read()
        two_check_data = json.loads(contents)
    
    with open("/workspace/SMART_CS/PAGE/miss_check_data.json", "r", encoding="utf8") as e:
        contents = e.read()
        miss_check_data = json.loads(contents)
        
    with open("/workspace/SMART_CS/PAGE/done_check_data.json", "r", encoding="utf8") as f:
        contents = f.read()
        done_check_data = json.loads(contents)
        
    with open("/workspace/SMART_CS/PAGE/other_check_data.json", "r", encoding="utf8") as g:
        contents = g.read()
        other_check_data = json.loads(contents)
        
    anal_list = [miss_check_data, none_check_data, one_check_data, two_check_data, done_check_data, other_check_data]
    for group in anal_list:
        list_num = list_num + 1
        print(group)
        print(type(group))
        for i in group.keys():
            print(i)
            n = 0
            m = 0
            p = 0
            f = 0
            no = 0
            br = 0
            t_num = 0
            for j in range(len(Pole_list)):
                if Pole_list['officename'][j] == str(i):
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

            if t_num > 0: 
                
                if list_num == 1:
                    miss_data["miss_project_"+str(miss_num)] = str(group[i])
                    miss_data["miss_name_"+str(miss_num)] = str(i)
                    miss_data["miss_check_"+str(miss_num)] = "미분류"
                    miss_data["miss_total_"+str(miss_num)] = str(t_num)
                    miss_data["miss_no_ck_"+str(miss_num)] = str(n)
                    miss_data["miss_do_ck_"+str(miss_num)] = str(m+p+f)
                    miss_data["miss_fi_ck_"+str(miss_num)] = str(p)
                    miss_data["miss_se_ck_"+str(miss_num)] = str(f)
                    miss_data["miss_ok_ck_"+str(miss_num)] = str(no)
                    miss_data["miss_br_ck_"+str(miss_num)] = str(br)
                    miss_num = miss_num + 1

                elif list_num == 2:
                    none_data["none_project_"+str(none_num)] = str(group[i])
                    none_data["none_name_"+str(none_num)] = str(i)
                    none_data["none_check_"+str(none_num)] = "미측정"
                    none_data["none_total_"+str(none_num)] = str(t_num)
                    none_data["none_no_ck_"+str(none_num)] = str(n)
                    none_data["none_do_ck_"+str(none_num)] = str(m+p+f)
                    none_data["none_fi_ck_"+str(none_num)] = str(p)
                    none_data["none_se_ck_"+str(none_num)] = str(f)
                    none_data["none_ok_ck_"+str(none_num)] = str(no)
                    none_data["none_br_ck_"+str(none_num)] = str(br)
                    none_num = none_num + 1

                elif list_num == 3:
                    one_data["one_project_"+str(one_num)] = str(group[i])
                    one_data["one_name_"+str(one_num)] = str(i)
                    one_data["one_check_"+str(one_num)] = "1차 분석중"
                    one_data["one_total_"+str(one_num)] = str(t_num)
                    one_data["one_no_ck_"+str(one_num)] = str(n)
                    one_data["one_do_ck_"+str(one_num)] = str(m+p+f)
                    one_data["one_fi_ck_"+str(one_num)] = str(p)
                    one_data["one_se_ck_"+str(one_num)] = str(f)
                    one_data["one_ok_ck_"+str(one_num)] = str(no)
                    one_data["one_br_ck_"+str(one_num)] = str(br)
                    one_num = one_num + 1
                    
                elif list_num == 4:
                    two_data["two_project_"+str(two_num)] = str(group[i])
                    two_data["two_name_"+str(two_num)] = str(i)
                    two_data["two_check_"+str(two_num)] = "2차 분석중"
                    two_data["two_total_"+str(two_num)] = str(t_num)
                    two_data["two_no_ck_"+str(two_num)] = str(n)
                    two_data["two_do_ck_"+str(two_num)] = str(m+p+f)
                    two_data["two_fi_ck_"+str(two_num)] = str(p)
                    two_data["two_se_ck_"+str(two_num)] = str(f)
                    two_data["two_ok_ck_"+str(two_num)] = str(no)
                    two_data["two_br_ck_"+str(two_num)] = str(br)
                    two_num = two_num + 1
                    
                elif list_num == 5:
                    done_data["done_project_"+str(done_num)] = str(group[i])
                    done_data["done_name_"+str(done_num)] = str(i)
                    done_data["done_check_"+str(done_num)] = "분석완료"
                    done_data["done_total_"+str(done_num)] = str(t_num)
                    done_data["done_no_ck_"+str(done_num)] = str(n)
                    done_data["done_do_ck_"+str(done_num)] = str(m+p+f)
                    done_data["done_fi_ck_"+str(done_num)] = str(p)
                    done_data["done_se_ck_"+str(done_num)] = str(f)
                    done_data["done_ok_ck_"+str(done_num)] = str(no)
                    done_data["done_br_ck_"+str(done_num)] = str(br)
                    done_num = done_num + 1

                else:
                    other_data["other_project_"+str(other_num)] = str(group[i])
                    other_data["other_name_"+str(other_num)] = str(i)
                    other_data["other_check_"+str(other_num)] = "측정제외"
                    other_data["other_total_"+str(other_num)] = str(t_num)
                    other_data["other_no_ck_"+str(other_num)] = str(n)
                    other_data["other_do_ck_"+str(other_num)] = str(m+p+f)
                    other_data["other_fi_ck_"+str(other_num)] = str(p)
                    other_data["other_se_ck_"+str(other_num)] = str(f)
                    other_data["other_ok_ck_"+str(other_num)] = str(no)
                    other_data["other_br_ck_"+str(other_num)] = str(br)
                    other_num = other_num + 1
                    
    with open('/workspace/SMART_CS/PAGE/miss_data.json', 'w', encoding='utf-8') as make_file:
        json.dump(miss_data, make_file, ensure_ascii = False, indent='\t') 
        
    with open('/workspace/SMART_CS/PAGE/none_data.json', 'w', encoding='utf-8') as make_file:
        json.dump(none_data, make_file, ensure_ascii = False, indent='\t')
        
    with open('/workspace/SMART_CS/PAGE/one_data.json', 'w', encoding='utf-8') as make_file:
        json.dump(one_data, make_file, ensure_ascii = False, indent='\t')
        
    with open('/workspace/SMART_CS/PAGE/two_data.json', 'w', encoding='utf-8') as make_file:
        json.dump(two_data, make_file, ensure_ascii = False, indent='\t')
        
    with open('/workspace/SMART_CS/PAGE/done_data.json', 'w', encoding='utf-8') as make_file:
        json.dump(done_data, make_file, ensure_ascii = False, indent='\t')
        
    with open('/workspace/SMART_CS/PAGE/other_data.json', 'w', encoding='utf-8') as make_file:
        json.dump(other_data, make_file, ensure_ascii = False, indent='\t') 
        
    print("업그레이드 완료")
upgrade()

def find_category():
    Pole_list = pd.DataFrame()
    sql_str = 'select * from tb_pole'
    result = poledb_conn.do_select_pd(sql_str)
    Pole_list = result
    bf_name = "none"
    pole_num = 0
    pole_break = 0
    data = {}
    
    with open("/workspace/SMART_CS/PAGE/office_data.json", "r", encoding="utf8") as a:
        contents = a.read()
        office_data = json.loads(contents)
        
    with open("/workspace/SMART_CS/PAGE/other_check_data.json", "r", encoding="utf8") as b:
        contents = b.read()
        other_check_data = json.loads(contents)
        
    with open("/workspace/SMART_CS/PAGE/done_check_data.json", "r", encoding="utf8") as c:
        contents = c.read()
        done_check_data = json.loads(contents)
        
    with open("/workspace/SMART_CS/PAGE/none_check_data.json", "r", encoding="utf8") as d:
        contents = d.read()
        none_check_data = json.loads(contents)
        
    with open("/workspace/SMART_CS/PAGE/one_check_data.json", "r", encoding="utf8") as f:
        contents = f.read()
        one_check_data = json.loads(contents)
    
    for i in  office_data.keys():
        for j in range(len(Pole_list)):
            if Pole_list['officename'][j] == i:
                pole_num = pole_num + 1
                print(pole_num)
                if Pole_list['diagstate'][j] == "AF" or Pole_list['diagstate'][j] == "AP":
                    pole_break = pole_break + 1
                    print(pole_break)
                    
        if pole_num == pole_break:
            print("분석 제외대상 프로젝트 : ", office_data[i] , i)
            data[i] = office_data[i]
        pole_num  = 0
        pole_break = 0
                
    for n in other_check_data.keys():
        if n in data: 
            print("중복 삭제")
            del data[n]
            
    for m in done_check_data.keys():
        if m in data: 
            print("중복 삭제")
            del data[m]
        
    for k in none_check_data.keys():
        if k in data: 
            print("중복 삭제")
            del data[k]
            
    for h in one_check_data.keys():
        if h in data: 
            print("중복 삭제")
            del data[h]
            
    with open('/workspace/SMART_CS/pole_anal/ggggggggggggg.json', 'w', encoding='utf-8') as make_file:
        json.dump(data, make_file, ensure_ascii = False, indent='\t') 