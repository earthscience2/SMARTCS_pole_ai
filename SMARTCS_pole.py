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
from collections import OrderedDict
from pytz import timezone
from datetime import datetime
import maintime  
import time
import os
import traceback
import shutil

project_list = []
project_list_a = []
bf_project = '-'
bff_project = '-'
start = 0
(nowday, nowhour, nowminute, simplenowday, nowtime) = maintime.maintime()
simple = simplenowday.split('-')
ssimplenowday = simple[0] + simple[1] +simple[2]
while True: 
    try:
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
        from collections import OrderedDict
        from pytz import timezone
        from datetime import datetime
        import maintime  
        import time
        import os
        import traceback
        import shutil
        (nowday, nowhour, nowminute, simplenowday, nowtime) = maintime.maintime()
        simple = simplenowday.split('-')
        ssimplenowday = simple[0] + simple[1] +simple[2]
        #----------------------------------------------------------------------------
        slack.slack("----------------------전주 분석 시작----------------------")
        project_list = []
        project_list_a = []
        Pole_list = None
        bf_project = '-'
        bff_project = '-'
        if not os.path.exists("/workspace/SMART_CS/pole_anal/AUTO_analysis/"+simplenowday+"_result"):
            os.makedirs("/workspace/SMART_CS/pole_anal/AUTO_analysis/"+simplenowday+"_result")

        with open("/workspace/SMART_CS/PAGE/one_data.json", "r", encoding="utf8") as f:
            contents = f.read()
            project = json.loads(contents)

        with open('/workspace/SMART_CS/pole_anal/pin.json') as m:
            data = m.read()
            input_conf = json.loads(data)
            
        IN_Info = pd.json_normalize(input_conf)
        
        for i in range(99):
            if project["one_project_"+str(i+1)] != "":
                project_list_a.append(project["one_project_"+str(i+1)])
        
        for a in range(len(project_list_a)):
            if bff_project != project_list_a[a]:
                project_list.append(project_list_a[a])
                bff_project = project_list_a[a]
                
        slack.slack("현재 시간 : " + nowtime)
        slack.slack("분석할 프로젝트============ ")

        for j in range(len(project_list)):
            slack.slack(str(j+1) + ". " + project_list[j])
            if not os.path.exists("/workspace/SMART_CS/pole_anal/AUTO_analysis/"+simplenowday+"_result/"+project_list[j]+'/output'):
                 os.makedirs("/workspace/SMART_CS/pole_anal/AUTO_analysis/"+simplenowday+"_result/"+project_list[j]+'/output')
                    
            if not os.path.exists("/workspace/SMART_CS/pole_anal/AUTO_analysis/"+simplenowday+"_result/"+project_list[j]+'/input'):
                 os.makedirs("/workspace/SMART_CS/pole_anal/AUTO_analysis/"+simplenowday+"_result/"+project_list[j]+'/input')
                    
            if not os.path.exists("/workspace/SMART_CS/pole_anal/AUTO_analysis/"+simplenowday+"_result/"+project_list[j]+'/output/ref'):
                 os.makedirs("/workspace/SMART_CS/pole_anal/AUTO_analysis/"+simplenowday+"_result/"+project_list[j]+'/output/ref')
                    
            with open('/workspace/SMART_CS/pole_anal/AUTO_analysis/'+simplenowday+'_result/'+project_list[j]+'/input/'+project_list[j]+'_pole_auto_conf.json', 'w', encoding='utf-8') as make_file:
                json.dump(input_conf, make_file, ensure_ascii = False, indent='\t')  


        slack.slack("프로젝트 분석 시작============ ")

        for k in range(len(project_list)):
            slack.slack("프로젝트 " + project_list[k] + " 스캔시작")
            
            Pole_list=PDB.get_pole_list(project_list[k])
            
            slack.slack("프로젝트 검사")
            
            Pole_list=Pole_list.loc[Pole_list['diagstate']=='MF'].reset_index()
            
            for h in range(len(Pole_list)):
                print(Pole_list['teamid'][h])
                if Pole_list['teamid'][h] == 'daegun' or Pole_list['teamid'][h] == 'KEOSAN':
                    Pole_list = Pole_list.drop(h, axis=0)
                    
            Signal_Analy_result=pole_lib.pole_State_Scan(Pole_list,IN_Info,project_list[k])
            Signal_Analy_result.to_csv("/workspace/SMART_CS/pole_anal/AUTO_analysis/"+simplenowday+"_result/"+project_list[k]+'/output/Polelist_scan-'+project_list[k]+'_MF_'+ssimplenowday+'.csv') 

            slack.slack(project_list[k]+' 스캔결과============')
            slack.slack('분석 대상 전주 : '+str(len(Pole_list))+' 개')
            
            if str(len(Pole_list)) == '0':
                slack.slack('분석 대상 전주 없음')
                if os.path.exists("/workspace/SMART_CS/pole_anal/AUTO_analysis/"+simplenowday+"_result/"+project_list[k]):
                    shutil.rmtree("/workspace/SMART_CS/pole_anal/AUTO_analysis/"+simplenowday+"_result/"+project_list[k])
            
            else:
                slack.slack("프로젝트 " + project_list[k] + " 분석시작")
                result=pole_lib.pole_State_Analysis_all(IN_Info,Pole_list,project_list[k],simplenowday)
                result.to_csv("/workspace/SMART_CS/pole_anal/AUTO_analysis/"+simplenowday+"_result/"+project_list[k]+'/output/result_detail-'+project_list[k]+'_MF_'+ssimplenowday+'.csv') 
                slack.slack("프로젝트 " + project_list[k] + " 분석완료")
            
        while int(nowhour) != 1 or int(nowminute) != 59:
            nowhour, nowminute = maintime.sptime()
            if start == 0:
                slack.slack("금일 분석완료")
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
            from collections import OrderedDict
            from pytz import timezone
            from datetime import datetime
            import maintime  
            import time
            import os
            import traceback
            import shutil
            time.sleep(1)    
            start = 1 
            (nowday, nowhour, nowminute, simplenowday, nowtime) = maintime.maintime()
            simple = simplenowday.split('-')
            ssimplenowday = simple[0] + simple[1] +simple[2]
        start = 0
    
    except Exception as e:
        time.sleep(60)
        slack.slack(e)
