import mysqldb
import poleconf
import logger
import json
import pandas as pd

logger = logger.get_logger()
poledb_conn = None

def find_name():
    bf_name = '-'
    name_data = {}
    with open("/workspace/SMART_CS/PAGE/grop.json", "r", encoding="utf8") as f:
            contents = f.read()
            json_data = json.loads(contents)

    poledb_conn = mysqldb.Mysqlhandler(poleconf.poledb_host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
    poledb_conn.connect()

    sql_str = 'select * from tb_pole'

    result = poledb_conn.do_select_pd(sql_str)
    print(len(result))

    for i in range(len(json_data)):

        for j in range(len(result['poleid'])):

            if result['groupname'][j] == json_data["gropname"+str(i+1)]:

                if bf_name != result['officename'][j]:

                    name_data[result['officename'][j]] = result['groupname'][j]
                    bf_name = result['officename'][j]
                    print(bf_name)

    with open('/workspace/SMART_CS/PAGE/total_data.json', 'w', encoding='utf-8') as make_file:
        json.dump(name_data, make_file, ensure_ascii = False, indent='\t')
