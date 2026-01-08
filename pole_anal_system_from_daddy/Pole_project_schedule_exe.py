# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:10:30 2021

@author: 스마트제어계측
"""
import schedule
import time
import subprocess
import datetime
import pandas as pd

def work():
 pole_project_list=pd.read_csv('analysis/Pole_project_list.csv', engine='python') 
 for i,Project in  enumerate(pole_project_list['project']):
     print(Project)
     if pole_project_list['state'][i]=='ing': 
        subprocess.call("python pole_analy_main_schedule.py "+Project+" all MF now 1", shell=True)
 
    
schedule.every().day.at("21:43").do(work)
while True:
  
        # Checks whether a scheduled task 
        # is pending to run or not
        schedule.run_pending()
        time.sleep(1)