import pageupgrade
import slack
import time
import tool
import json

slack.slack("페이지 자동 업데이트 시작")
check = 0     
tool.find_group()
tool.find_name()

with open("/workspace/SMART_CS/PAGE/office_data.json", "r", encoding="utf8") as a:
    contents = a.read()
    office_data = json.loads(contents)

with open("/workspace/SMART_CS/PAGE/none_check_data.json", "r", encoding="utf8") as b:
    contents = b.read()
    none_check_data = json.loads(contents)
    
with open("/workspace/SMART_CS/PAGE/one_check_data.json", "r", encoding="utf8") as c:
    contents = c.read()
    one_check_data = json.loads(contents)
    
with open("/workspace/SMART_CS/PAGE/two_check_data.json", "r", encoding="utf8") as d:
    contents = d.read()
    two_check_data = json.loads(contents)
    
with open("/workspace/SMART_CS/PAGE/other_check_data.json", "r", encoding="utf8") as e:
    contents = e.read()
    other_check_data = json.loads(contents)
    
with open("/workspace/SMART_CS/PAGE/done_check_data.json", "r", encoding="utf8") as f:
    contents = f.read()
    done_check_data = json.loads(contents)
    
miss_check_data = office_data

for i in none_check_data.keys():
    del miss_check_data[i]
    
for j in one_check_data.keys():
    del miss_check_data[j]
    
for k in two_check_data.keys():
    del miss_check_data[k]
    
for n in other_check_data.keys():
    del miss_check_data[n]
    
for m in done_check_data.keys():
    del miss_check_data[m]

with open('/workspace/SMART_CS/PAGE/miss_check_data.json', 'w', encoding='utf-8') as make_file:
    json.dump(miss_check_data, make_file, ensure_ascii = False, indent='\t')

print("done")
time.sleep(100)

while True:
    try:
        tool.find_group()
        tool.find_name()
        
        time.sleep(10)
        check = check + 1
        
        if check == 10:
            slack.slack("페이지 정상 작동중")
            check = 0
            
    except Exception as e:
        slack.slack("페이지 오류 : " + str(e))  