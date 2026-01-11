import pymysql
from pymysql.constants import CLIENT
import logger
import pandas as pd
from . import poleconf

logger = logger.get_logger()

class Mysqlhandler:
    def __init__(self, host_with_port, user, password, db_name):
        self.host, self.port = self._parse_host_port(host_with_port)
        self.user = user
        self.password = password
        self.db_name = db_name
        self.conn = None

    def _parse_host_port(self, host_with_port):
        if ':' in host_with_port:
            host, port = host_with_port.split(':')
            return host, int(port)
        else:
            return host_with_port, 3306

    def connect(self):
        try:
            self.conn = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                db=self.db_name,
                port=self.port,
                client_flag=pymysql.constants.CLIENT.MULTI_STATEMENTS,
                charset='utf8'
            )
            logger.info('mysql) host={} db_name={} port={}'.format(self.host, self.db_name, self.port))
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.conn = None

    def close(self):
        if self.conn:
            self.conn.close()

    def do_select(self, query_str, params=None):
        if query_str == '':
            logger.info('Empty Query String')
            return []

        curs = self.conn.cursor()
        if params is None:
            curs.execute(query_str)
        else:
            curs.execute(query_str, params)
        rows = curs.fetchall()
        curs.close()
        return rows

    def do_select_pd(self, query_str, params=None):
        if query_str == '':
            logger.info('Empty Query String')
            return []
        
        curs = self.conn.cursor(pymysql.cursors.DictCursor)
        if params is None:
            curs.execute(query_str)
        else:
            curs.execute(query_str, params)
        rows = curs.fetchall()
        raws_pd = pd.DataFrame(rows)
        curs.close()
        return raws_pd

    def do_sql(self, sql):
        if sql == '':
            logger.info('Empty Query String')
            return
        logger.info(sql)
        curs = self.conn.cursor()
        curs.execute(sql)
        self.conn.commit()
        curs.close()

    def execute(self, sql, data):
        if sql == '':
            logger.info('Empty Query String')
            return
        logger.info(sql)
        curs = self.conn.cursor()
        curs.execute(sql, data)
        self.conn.commit()
        curs.close()

    def executemany(self, sql, data):
        if sql == '':
            logger.info('Empty Query String')
            return
        logger.info(sql)
        curs = self.conn.cursor()
        curs.executemany(sql, data)
        self.conn.commit()
        curs.close()

def poledb_init():
    global poledb_conn

    poledb_conn = Mysqlhandler(
        poleconf.poledb_host,  # 호스트와 포트가 함께 있는 문자열
        poleconf.poledb_user,
        poleconf.poledb_pwd,
        poleconf.poledb_dbname
    )
    poledb_conn.connect()

def fetch_tables():
    tables = poledb_conn.do_select("SHOW TABLES")
    
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        
        table_structure = poledb_conn.do_select(f"DESCRIBE {table_name}")
        
        for column in table_structure:
            print(f"  {column}")



