
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:37:29 2021

@author: 스마트제어계측
"""

import pole_auto_analy_command as pole_lib
import argparse
import datetime
import pandas as pd


def work(Dir,method,option):
    
        IN_Info,Pole_list=pole_lib.conf_file_open(Dir) 
        #diagstate
        #  -	미측정
        # MF	측정완료
        # AP	분석중(1차분석 완료)
        # AF	분석완료
        if option=='MF':
            Pole_list=Pole_list.loc[Pole_list['diagstate']=='MF'].reset_index()
        elif option=='AP':
            Pole_list=Pole_list.loc[Pole_list['diagstate']=='AP'].reset_index()
        elif option=='AF':
            Pole_list=Pole_list.loc[Pole_list['diagstate']=='AF'].reset_index()
        elif option=='all':
            Pole_list=Pole_list.loc[Pole_list['diagstate']!='-'].reset_index()
            
        print(Pole_list)
        d_today = datetime.date.today()
        today_str=d_today.strftime("%Y%m%d")
        
        if method=='scan' or method=='all' :
            #pole_lib.pole_State_Scan(Pole_list,IN_Info,Dir)
            
            Signal_Analy_result=pole_lib.pole_State_Scan(Pole_list,IN_Info,Dir)
            
            Signal_Analy_result.to_csv('analysis/'+Dir+'/output/'+'Polelist_scan-'+Dir+'_'+option+'_'+today_str+'.csv')   
        
        if method=='analysis' or method=='all':    
            
            result=pole_lib.pole_State_Analysis_all(IN_Info,Pole_list,Dir)
            result.to_csv('analysis/'+Dir+'/output/'+'result_detail-'+Dir+'_'+option+'_'+today_str+'.csv')
        
        

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Pole Analysis : 2021')
    parser.add_argument('DIR', type=str, default='', help='ex> CHUNGBUK-202110')
    parser.add_argument('method', type=str, default='scan',help='ex) scan : 전주 신호 적합성 분석  Analysis: 전주 신호 파단여부분석   all: scan+anaysis')
    parser.add_argument('option', type=str, default='all',help='ex) all:모든 전주  MF:측정완료(분석된 것 제외)  AP:1차분석된 전주  AF:2차 분석 완료된 전주')    
    args = parser.parse_args()
    
    
    import time
    start = time.time()  # 시작 시간 저장

    work(args.DIR, args.method,args.option)
    print("runtime :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간