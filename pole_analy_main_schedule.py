# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:37:29 2021

@author: 스마트제어계측
"""

import pole_auto_analy_command as pole_lib
import argparse
import datetime
import pandas as pd


def work(Dir,method,option,start_date,end_date):
    
    if start_date=='now':

        now = datetime.datetime.now()

        now_string = now.strftime("%Y%m%d")
        print("date and time =", now_string )   
    
        pre_now = now-datetime.timedelta(days=int(end_date)-1)
    
        pre_now_string=pre_now.strftime("%Y%m%d")
    
        print("date and time =", pre_now_string)   
        
        start_date=pre_now_string
        end_date=now_string
    
    start = datetime.datetime.strptime(start_date, '%Y%m%d')
    end = datetime.datetime.strptime(end_date, '%Y%m%d')
    step = datetime.timedelta(days=1)
    
    df = pd.DataFrame()
    days=[]
    
    
    while start <= end:
       days.append(start.strftime('%Y-%m-%d'))
       start += step
    total_day=len(days)
    
    
    
    for day in days:
               
        IN_Info,Pole_list=pole_lib.conf_file_open(Dir) 
        
        if option=='MF':
            Pole_list=Pole_list.loc[Pole_list['diagstate']=='MF'].reset_index()
        elif option=='AP':
            Pole_list=Pole_list.loc[Pole_list['diagstate']=='AP'].reset_index()
        elif option=='AF':
            Pole_list=Pole_list.loc[Pole_list['diagstate']=='AF'].reset_index()
        elif option=='all':
            Pole_list=Pole_list.loc[Pole_list['diagstate']!='-'].reset_index()
        
        
        
        for i,poleid in enumerate(Pole_list['poleid']): 
            hh=Pole_list['endtime'][i].strftime('%Y-%m-%d')
            if hh!=day:
               Pole_list=Pole_list.drop([i]) 
        
        if (method=='scan' or method=='all') and len(Pole_list)>0:
            Signal_Analy_result=pole_lib.pole_State_Scan(Pole_list,IN_Info,Dir)
            
            Signal_Analy_result.to_csv('analysis/'+Dir+'/output/schedule/scan/'+'Polelist_scan-'+Dir+'_'+option+'_'+day+'.csv')    
        
        
        if (method=='analysis' or method=='all')and len(Pole_list)>0:    
            
            result=pole_lib.pole_State_Analysis_all(IN_Info,Pole_list,Dir)
            result.to_csv('analysis/'+Dir+'/output/schedule/result/'+'result_detail_'+Dir+'_'+option+'_'+day+'.csv')
            
            

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Pole Analysis : 2021')
    parser.add_argument('DIR', type=str, default='2021\거산전설\남인천지사', help='ex> 2021\거산전설\남인천지사')
    parser.add_argument('method', type=str, default='scan',help='ex) scan : 전주 신호 적합성 분석  Analysis: 전주 신호 파단여부분석   all: scan+anaysis')
    parser.add_argument('option', type=str, default='all',help='ex) all:모든 전주  MF:측정완료(분석된 것 제외)  AP:1차분석된 전주  AF:2차 분석 완료된 전주') 
    parser.add_argument('start_date', type=str, default='now',help='Input the start date: 20201110 or now')
    parser.add_argument('end_date', type=str,default='10', help='Input the end date : 20201120 or 10')
    

    args = parser.parse_args()
    
    
    import time
    start = time.time()  # 시작 시간 저장

    work(args.DIR, args.method,args.option,args.start_date,args.end_date)
    print("runtime :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간