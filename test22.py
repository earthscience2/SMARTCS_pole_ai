import mysqldb

# 데이터베이스에 연결
db_connection = mysqldb.connect(
    host="smartpole-is.iptime.org",
    user="root",
    passwd="password"
)

# 커서 생성
cursor = db_connection.cursor()

# 데이터베이스 목록 가져오기
cursor.execute("SHOW DATABASES")
databases = cursor.fetchall()

# 각 데이터베이스의 테이블 목록 및 구조 조회
for database in databases:
    db_name = database[0]
    print(f"\nDatabase: {db_name}")
    
    # 데이터베이스 사용
    cursor.execute(f"USE {db_name}")
    
    # 테이블 목록 가져오기
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        print(f"\n  Table: {table_name}")
        
        # 테이블 구조 조회
        cursor.execute(f"DESCRIBE {table_name}")
        table_structure = cursor.fetchall()
        
        for column in table_structure:
            print(f"    {column}")

# 연결 종료
cursor.close()
db_connection.close()

