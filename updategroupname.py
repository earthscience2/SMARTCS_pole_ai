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

file_data = OrderedDict()
PDB.poledb_init()
logger=logger.get_logger()
poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
poledb_conn.connect()
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
print(gropname)
for o in range(len(gropname)):
    part = gropname[o].split('-')
    print(len(part[1]))
    if len(part[1]) == 6:
        project_list.append([gropname[o], part[1]])

project_list = sorted(project_list, key=lambda x:x[1])

for m in range(len(project_list)):
    file_data['gropname'+str(m+1)] = project_list[len(project_list) - m - 1][0]
    
with open('/workspace/SMART_CS/PAGE/grop.json', 'w', encoding='utf-8') as make_file:
    json.dump(file_data, make_file, ensure_ascii = False, indent='\t')   