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



poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
poledb_conn.connect()

query = "select * from tb_pole_group"
groupname_data = poledb_conn.do_select_pd(query)
groupname_data = groupname_data['groupname'].tolist()
#print((groupname_data))

query = "select * from tb_diag_pole_info where groupname = '서울동대문중랑-202210'"
pole_id_data = poledb_conn.do_select_pd(query)
print(pole_id_data)


query = "select * from tb_diag_pole_meas_data info where poleid = '0126G191'"
pole_meas_data = poledb_conn.do_select_pd(query)

query = "select * from tb_diag_pole_meas_result info where poleid = '0126G191'"
pole_result_data = poledb_conn.do_select_pd(query)

pole_meas_data = pole_meas_data.values.tolist()
