import os
import json
from tqdm import tqdm  
import poledb as PDB
import shutil
import os


#====================================================================
# json 파일 만들기
#====================================================================
def make_json():
    normal_file = 'raw_data_ai_json/normal/'
    break_file = 'raw_data_ai_json/break/'
    normal_num = 0
    break_num = 0
    
    if os.path.exists(normal_file):
        shutil.rmtree(normal_file)
    os.makedirs(normal_file)
        
    if os.path.exists(break_file):
        shutil.rmtree(break_file)
    os.makedirs(break_file)
                
    server_list = ["main", "is", "kh"]
    for server in server_list:
        PDB.poledb_init(server)
        print(server, "서버 // 데이터 수집중")
        groupname_list = PDB.groupname_info()
        for i in tqdm(groupname_list, desc="Processing poles"):
            normal_data_list = PDB.group_anal_type_pole_2(i, "N")
            break_data_list = PDB.group_anal_type_pole_2(i, "B")
            normal_info = []
            break_info = []
            
            for j1 in normal_data_list:
                normal_info.append(j1['poleid'])

            for j2 in break_data_list:
                try:
                    break_info.append([j2['poleid'], str(float(j2['breakheight'])), str(float(j2['breakdegree']))])
                except:
                    continue
        
            if not os.path.exists(normal_file + i + '.json'):
                with open(normal_file + i + '.json', 'w', encoding='utf-8') as json_file:
                    json.dump(normal_info, json_file, ensure_ascii=False, indent=4)
                    normal_num = normal_num + 1
                    
            if not os.path.exists(break_file + i + '.json'):
                with open(break_file + i + '.json', 'w', encoding='utf-8') as json_file:
                    json.dump(break_info, json_file, ensure_ascii=False, indent=4)
                    break_num = break_num + 1
                    
    print("데이터 수집 완료 // 정상 " + str(normal_num) + " 개 / 파단 " + str(break_num) + " 개" )
#====================================================================
# json파일 만들기
#====================================================================
make_json() 