from configparser import ConfigParser

parser = ConfigParser()
parser.read('pole.ini')

#poledb_host = parser.get('DB', 'db_host')
#poledb_dbname = parser.get('DB', 'db_name')
#poledb_user = parser.get('DB', 'db_user')
#poledb_pwd = parser.get('DB', 'db_pwd')

poledb_host = 'smartpole-is.iptime.org:33306'
poledb_dbname = 'polediagdb'
poledb_user = 'polediag'
poledb_pwd = 'moricon001'

print('[POLE-DB] host={} dbname={} user={} pwd={}'.format(poledb_host, poledb_dbname, poledb_user, poledb_pwd))
