
import mysqldb
import poleconf
import logger

logger = logger.get_logger()
poledb_conn = None

def poledb_init(server=None):
    """
    데이터베이스 연결 초기화
    
    Args:
        server: 서버 이름 ('main', 'is', 'kh', 'jt'). None이면 기본 설정 사용
    """
    global poledb_conn
    
    # 서버별 호스트 설정
    server_hosts = {
        'main': '210.105.85.3',
        'is': 'smartpole-is.iptime.org:33306',
        'kh': 'smartpole-kh.iptime.org:33306',
        'jt': 'smartpole-jt.iptime.org:33306'
    }
    
    # 서버가 지정된 경우 해당 호스트 사용, 아니면 기본 설정 사용
    if server and server in server_hosts:
        host = server_hosts[server]
    else:
        host = poleconf.poledb_host
    
    poledb_conn = mysqldb.Mysqlhandler(host, poleconf.poledb_user, poleconf.poledb_pwd, poleconf.poledb_dbname)
    poledb_conn.connect()

def ping():
    poledb_conn.do_sql('select now()')
    logger.info('[DB] ping success')
    return

def get_meas_data(poleid, measno, devicetype, axis):
    data = [ poleid, measno, devicetype, axis ]
    sql_str = 'SELECT * FROM tb_diag_pole_meas_data WHERE poleid=%s and measno=%s and devicetype=%s and axis=%s order by idx;'

    logger.info('sql_str={} data={}'.format(sql_str, data))

    result = None
    try:
        result = poledb_conn.do_select_pd(sql_str, data)
    except Exception as e:
        logger.error(str(e))
        result = None

    return result

def get_meas_result(poleid, dtype):
    data = [ poleid, dtype ]
    sql_str = 'SELECT * FROM tb_diag_pole_meas_result WHERE poleid=%s and devicetype=%s order by idx;'

    logger.info('sql_str={} data={}'.format(sql_str, data))

    result = None
    try:
        result = poledb_conn.do_select_pd(sql_str, data)
    except Exception as e:
        logger.error(str(e))
        result = None

    return result

def get_meas_result_count(poleid):
    data = [ poleid ]
    sql_str = 'select devicetype, count(*) as cnt from tb_diag_pole_meas_result tdpmr  where poleid=%s group by devicetype;'

    logger.info('sql_str={} data={}'.format(sql_str, data))

    result = None
    try:
        result = poledb_conn.do_select_pd(sql_str, data)
    except Exception as e:
        logger.error(str(e))
        result = None

    return result


#DB 업데이트
def update_anal_result(poleid, anal_result, break_cnt, break_loc_height_list, break_loc_degree_list, anal_comment):
    logger.info('update_anal_result) poleid={} anal_result={} break_cnt={} break_loc_height_list={} break_loc_degree_list={} anal_commnet={} '.format(poleid, anal_result, str(break_cnt), break_loc_height_list, break_loc_degree_list, anal_comment)) 

    # tb_anal_state
    # poleid, anal1userid, anal1finyn, anal1result, anal1breakcnt
    data = [ poleid, 'pole', 'Y', anal_result, break_cnt ]
    sql = 'REPLACE INTO tb_anal_state (poleid, anal1userid, anal1finyn, anal1result, anal1breakcnt, regdate) \
            VALUES(%s, %s, %s, %s, %s, now());'

    try:
        poledb_conn.execute(sql, data)
    except Exception as e:
        logger.error('tb_anal_state')
        logger.error(str(e))
        return False

    # tb_anal_result
    data = [ poleid, 1 ]
    sql = 'DELETE FROM tb_anal_result where poleid=%s and analstep=%s'
    try:
        poledb_conn.execute(sql, data)
    except Exception as e:
        logger.error('tb_anal_result1')
        logger.error(str(e))
        return False

    if anal_result == 'B' and break_cnt > 0:
        data = []
        for i in range(break_cnt):
            break_data = [ poleid, 'pole', 1, anal_result, break_loc_height_list[i], break_loc_degree_list[i] ]
            data.append(break_data)
        sql = 'INSERT INTO tb_anal_result(poleid, userid, analstep, breakstate, breakheight, breakdegree, regdate) ' \
                ' VALUES(%s, %s, %s, %s, %s, %s, now())'
        try:
            poledb_conn.executemany(sql, data)
        except Exception as e:
            logger.error('tb_anal_result2')
            logger.error(str(e))
            return False
    else:
        break_data = [ poleid, 'pole', 1, anal_result ]
        sql = 'INSERT INTO tb_anal_result(poleid, userid, analstep, breakstate, regdate) ' \
                ' VALUES(%s, %s, %s, %s, now())'
        try:
            poledb_conn.execute(sql, break_data)
        except Exception as e:
            logger.error('tb_anal_result2')
            logger.error(str(e))
            return False

    # tb_anal_comment
    data = [ poleid, 'pole', 1, anal_comment ]
    sql = 'REPLACE INTO tb_anal_comment (poleid, userid, analstep, comment, regdate) \
            VALUES(%s, %s, %s, %s, now());'

    try:
        poledb_conn.execute(sql, data)
    except Exception as e:
        logger.error('tb_anal_commnet')
        logger.error(str(e))
        return False

    data = [ 'AP', poleid ]
    sql = 'update tb_pole set diagstate=%s where poleid=%s'

    try:
        poledb_conn.execute(sql, data)
    except Exception as e:
        logger.error('tb_anal_commnet')
        logger.error(str(e))
        return False

    return True

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


def get_pole_list(group_name):
    data = [ group_name ]
    print(data)
    sql_str = 'select tp.poleid, tp.regdate, tp.diagstate, tds.endtime, tds.breakstate , tds.teamid from tb_pole tp join tb_diag_state tds on tds.poleid = tp.poleid where tds.groupname=%s and diagstate in ("MF", "AP", "AF");'

    logger.info('sql_str={} data={}'.format(sql_str, data))

    result = None
    try:
        result = poledb_conn.do_select_pd(sql_str, data)

    except Exception as e:
        logger.error(str(e))
        result = None

    return result

def get_pole_list_a(group_name):
    data = [ group_name ]
    print(data)
    sql_str = 'select * from tb_pole where groupname=%s;'

    logger.info('sql_str={} data={}'.format(sql_str, data))

    result = None
    try:
        result = poledb_conn.do_select_pd(sql_str, data)
    except Exception as e:
        logger.error(str(e))
        result = None

    return result

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

def groupname_info():
    """
    모든 프로젝트(groupname) 목록을 조회
    
    Returns:
        list: 프로젝트 이름 목록
    """
    sql_str = 'SELECT DISTINCT groupname FROM tb_anal_state WHERE groupname IS NOT NULL ORDER BY groupname'
    
    logger.info('sql_str={}'.format(sql_str))
    
    result = None
    try:
        result = poledb_conn.do_select_pd(sql_str)
        if result is not None and not result.empty:
            return result['groupname'].tolist()
        return []
    except Exception as e:
        logger.error(str(e))
        return []

def group_anal_type_pole_2(groupname, anal_type):
    """
    2차 분석 결과를 타입별로 조회
    
    Args:
        groupname: 프로젝트 이름
        anal_type: 분석 타입 ('B': 파단, 'N': 정상)
    
    Returns:
        list: 분석 결과 목록 (dict 형태)
    """
    data = [groupname, 2, anal_type]  # analstep=2 (2차 분석)
    sql_str = """
        SELECT tar.poleid, tar.breakstate, tar.breakheight, tar.breakdegree, tas.groupname
        FROM tb_anal_result tar
        JOIN tb_anal_state tas ON tar.poleid = tas.poleid
        WHERE tas.groupname = %s
        AND tar.analstep = %s
        AND tar.breakstate = %s
        ORDER BY tar.poleid
    """
    
    logger.info('sql_str={} data={}'.format(sql_str, data))
    
    result = None
    try:
        result = poledb_conn.do_select_pd(sql_str, data)
        if result is not None and not result.empty:
            return result.to_dict('records')
        return []
    except Exception as e:
        logger.error(str(e))
        return []

#get_pole_list_all())

'''
poledb_init()
re_out=get_meas_result('9320A112', 'OUT')
re_in=get_meas_result('9420R141', 'IN')


group_name="GWANGMYEONG-202110"
kkk=get_pole_list(group_name)

print(kkk)
'''