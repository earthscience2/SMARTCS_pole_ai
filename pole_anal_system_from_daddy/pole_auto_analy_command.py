# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 17:53:04 2021

@author: 스마트제어계측
"""
import bottleneck as bn

import numpy as np
import pandas as pd

from scipy import signal
import logger
import magpylib_simulation_m as mag_m
import poledb as PDB
import slack
import ast
import maintime
from scipy.signal import detrend
PDB.poledb_init()
logger=logger.get_logger()

def butter_lowpass(cutoff, fs, order=5): 
    nyq = 0.5 * fs 
    normal_cutoff = cutoff / nyq 
    b, a = signal.butter(order, normal_cutoff, btype='lowpass',analog=False) 
    return b, a
  
    
def butter_lowpass_filter(data, cutoff, fs, order=5): 
    b, a = butter_lowpass(cutoff, fs, order=order) 
    y = signal.lfilter(b, a, data) 
    return y 

def rollavg_bottlneck(a,n):
    gg= bn.move_mean(a, window=n,min_count = None)
    
    return np.nan_to_num(gg)

def mac(phi1, phi2):
    """
    Modal assurance criterion number from comparison of two mode shapes.

    Arguments
    ---------------------------
    phi1 : double
        first mode
    phi2 : double
        second mode

    Returns
    ---------------------------
    mac_value : boolean
        MAC number
    """

    mac_value = np.real(np.abs(np.dot(phi1.T,phi2))**2 / np.abs((np.dot(phi1.T, phi1) * np.dot(phi2.T, phi2))))
    return mac_value

def normalize_phi(phi):
    """
    Normalize all complex-valued (or real-valued) mode shapes in modal transformation matrix.

    Arguments
    ---------------------------
    phi : double
        complex-valued (or real-valued) modal transformation matrix (column-wise stacked mode shapes)

    Returns
    ---------------------------
    phi_n : boolean
        modal transformation matrix, with normalized (absolute value of) mode shapes
    mode_scaling : 
        the corresponding scaling factors used to normalize, i.e., phi_n[:,n] * mode_scaling[n] = phi[n]

    """       
    phi_n = phi*0
    n_modes = np.shape(phi)[1]
    mode_scaling = np.zeros([n_modes])
    for mode in range(0, n_modes):
        mode_scaling[mode] = max(abs(phi[:, mode]))
        sign = np.sign(phi[np.argmax(abs(phi[:, mode])), mode])
        phi_n[:, mode] = phi[:, mode]/mode_scaling[mode]*sign

    return phi_n, mode_scaling


def pole_Info(poleid):
    pole_sgn_count=PDB.get_meas_result_count(poleid)
    
    re_out=PDB.get_meas_result(poleid, 'OUT')
    num_sig_out=re_out.shape[0]
    
    re_in=PDB.get_meas_result(poleid, 'IN')
    num_sig_in=re_in.shape[0]
    
    return num_sig_out,num_sig_in,re_out,re_in  # 전주 정보 신호 개수


def Signal_Info_out(poleid,stype,num,tpara):
    print('mum',poleid,stype,num)
    sig_x=PDB.get_meas_data(poleid,num,stype,'x')
    sig_y=PDB.get_meas_data(poleid,num,stype,'y')
    sig_z=PDB.get_meas_data(poleid,num,stype,'z')
    
    
    index_ch=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
    
    sig_x_ptp=[]
    sig_y_ptp=[]
    sig_z_ptp=[]
    sig_Scan_points=0
    x=0
    y=0
    z=0

    if len(sig_x)>0:
        for index,i in enumerate(index_ch):
            
            sig_x[i]=tpara*sig_x[i]
            sig_y[i]=tpara*sig_y[i]
            sig_z[i]=tpara*sig_z[i]
              
            sig_x_ptp.append(abs(sig_x[i].max()-sig_x[i].min()))
            sig_y_ptp.append(abs(sig_y[i].max()-sig_y[i].min()))
            sig_z_ptp.append(abs(sig_z[i].max()-sig_z[i].min()))
            
        sig_Scan_points=len(sig_x['ch1'])  
        
        if len(sig_x_ptp)>0: 
            x=max(sig_x_ptp)
            y=max(sig_y_ptp)
            z=max(sig_z_ptp)

    return x,y,z,sig_Scan_points

def Signal_Info_in(poleid,stype,num,tpara):
    sig_x=PDB.get_meas_data(poleid,num,stype,'x')
    index_ch=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
    
    sig_x_ptp=[]
    sig_Scan_points=0
    x=0
        
    
    if len(sig_x)>0:
        for index,i in enumerate(index_ch):
            sig_x[i]=tpara*sig_x[i]
            
            
            sig_x_ptp.append(abs(sig_x[i].max()-sig_x[i].min()))
            
        sig_Scan_points=len(sig_x['ch1'])  
        if len(sig_x_ptp)>0: 
            x=max(sig_x_ptp)
    
    return x,sig_Scan_points
    



def Signal_Info_select_data_out(poleid,stype,num,limit_x,limit_y,limit_z,tpara):
    sig_x=PDB.get_meas_data(poleid,num,stype,'x')
    sig_y=PDB.get_meas_data(poleid,num,stype,'y')
    sig_z=PDB.get_meas_data(poleid,num,stype,'z')
    index_ch=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
    
    sig_x_ptp=[]
    sig_y_ptp=[]
    sig_z_ptp=[]
    select_x_data=[]
    select_y_data=[]
    select_z_data=[]
    
    select_ch=[]
    
    
    if len(sig_x)>0:
        for index,i in enumerate(index_ch):
            
            sig_x[i]=tpara*sig_x[i]
            sig_y[i]=tpara*sig_y[i]
            sig_z[i]=tpara*sig_z[i]
            
            
            sig_x_ptp.append(abs(sig_x[i].max()-sig_x[i].min()))
            sig_y_ptp.append(abs(sig_y[i].max()-sig_y[i].min()))
            sig_z_ptp.append(abs(sig_z[i].max()-sig_z[i].min()))

        
        for index,i in enumerate(index_ch):
            sig_x[i]=detrend(sig_x[i])
            sig_x[i]=butter_lowpass_filter(sig_x[i],5, 30)
            sig_x[i]=np.gradient(sig_x[i],1)
            sig_x[i]=rollavg_bottlneck(sig_x[i],5)
            
            sig_y[i]=detrend(sig_y[i])
            sig_y[i]=butter_lowpass_filter(sig_y[i],5, 30)
            sig_y[i]=np.gradient(sig_y[i],1)
            sig_y[i]=rollavg_bottlneck(sig_y[i],5)
            
            sig_z[i]=detrend(sig_z[i])
            sig_z[i]=butter_lowpass_filter(sig_z[i],5, 30)
            sig_z[i]=np.gradient(sig_z[i],1)
            sig_z[i]=rollavg_bottlneck(sig_z[i],5)
            
            
            if sig_x_ptp[index]>=limit_x and sig_y_ptp[index]>=limit_y and sig_z_ptp[index]>=limit_z: 
                select_x_data.append(sig_x[i])
                select_y_data.append(sig_y[i])
                select_z_data.append(sig_z[i])
                select_ch.append(index+1) 
            
    return select_x_data,select_y_data,select_z_data,select_ch


def Signal_Info_select_data_in(poleid,stype,num,limit_x,tpara):
    sig_x=PDB.get_meas_data(poleid,num,stype,'x')
    index_ch=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
    
    sig_x_ptp=[]
    select_x_data=[]
    
    select_ch=[]

    if len(sig_x)>0:
        for index,i in enumerate(index_ch):
            
            sig_x[i]=tpara*sig_x[i]
            sig_x_ptp.append(abs(sig_x[i].max()-sig_x[i].min()))
            

        
        for index,i in enumerate(index_ch):
            sig_x[i]=detrend(sig_x[i])
            sig_x[i]=butter_lowpass_filter(sig_x[i],5, 30)
            sig_x[i]=np.gradient(sig_x[i],1)
            sig_x[i]=rollavg_bottlneck(sig_x[i],5)
            
            
            
            if sig_x_ptp[index]>=limit_x : 
                select_x_data.append(sig_x[i])
                select_ch.append(index+1) 
            
    return select_x_data,select_x_data,select_x_data,select_ch




def pole_State_Scan(Pole_list,IN_Info,dir):
        
        Signal_Analy_result= pd.DataFrame(columns=['poleid','s_num','stype','PtP_x','PtP_y','PtP_z','Scan_len','Scan_points','Scan_V','breakstate','State'])
        
        
        for poleid in Pole_list['poleid']:
                    
            num_sig_out,num_sig_in,re_out,re_in=pole_Info(poleid)
            
            print('### 조사 전주  {}  : 관통형 측정신호 개수는 {}, 후크형 측정신호 개수는 {} '.format(poleid,num_sig_in,num_sig_out)) 
            
            
            breakstate=Pole_list['breakstate'][Pole_list.loc[Pole_list['poleid']==poleid].index].to_list()[0]
            
            for kk in range(num_sig_out+num_sig_in): # 측정 신호 세트 스캔
                    if kk <num_sig_out:
                        stype='OUT'
                        num=int(re_out['measno'][kk])
                        sig_x_ptp,sig_y_ptp,sig_z_ptp,sig_Scan_points=Signal_Info_out(poleid,stype,num,IN_Info['Out_Trans_para'][0])
                        Scan_len=(re_out['edheight'][kk]-re_out['stheight'][kk])*1000
                        
                        Scan_vel=sig_Scan_points*10/Scan_len
                        
                        if Scan_vel>IN_Info['Limit_Scan_Vel'][0]: 
                            state='OK'
                        else:
                            state='NG'
                        temp_new_data={'poleid':poleid,'s_num':num,'stype':stype, 'PtP_x':sig_x_ptp,'PtP_y':sig_y_ptp,'PtP_z':sig_z_ptp, 
                                       'Scan_len':Scan_len,'Scan_points':sig_Scan_points,'Scan_V':round(Scan_vel,2),'breakstate':breakstate,'State':state}
                    
                    else:
                        stype='IN'
                        num=int(re_in['measno'][(kk-num_sig_out)])
                        sig_x_ptp,sig_Scan_points=Signal_Info_in(poleid,stype,num,IN_Info['IN_Trans_para'][0])
                        Scan_len=(re_in['stheight'][(kk-num_sig_out)]+re_in['depth'][(kk-num_sig_out)])*1000
                        
                        Scan_vel=sig_Scan_points*10/Scan_len
                        
                        if Scan_vel>IN_Info['Limit_Scan_Vel'][0]: 
                            state='OK'
                        else:
                            state='NG'
                        
                        temp_new_data={'poleid':poleid,'s_num':num,'stype':stype, 'PtP_x':sig_x_ptp, 
                                       'Scan_len':Scan_len,'Scan_points':sig_Scan_points,'Scan_V':round(Scan_vel,2),'breakstate':breakstate,'State':state}
                        
                    
                    Signal_Analy_result=Signal_Analy_result.append(temp_new_data,ignore_index=True)
        
        #d_today = datetime.date.today()
        #today_str=d_today.strftime("%Y%m%d")
        
        #Signal_Analy_result.to_csv('analysis/'+dir+'/output/'+today_str+'_Polelist_scan.csv')    
        return Signal_Analy_result          
###############################################


           
def pole_State_Analysis(IN_Info,poleid,current_num,total_num,dir,breakstate,simplenowday):
              
        detect_num=0
        
        limit_MAC_X=IN_Info['limit_MAC_X'][0]
        limit_MAC_Y=IN_Info['limit_MAC_Y'][0]
        limit_MAC_Z=IN_Info['limit_MAC_Z'][0]
        
        limit_corrcoef_X=IN_Info['limit_corrcoef_X'][0]
        limit_corrcoef_Y=IN_Info['limit_corrcoef_Y'][0]
        limit_corrcoef_Z=IN_Info['limit_corrcoef_Z'][0]
        
        Main_mag_s=IN_Info['Main_mag_s'][0]
        HallSensor_h=IN_Info['HallSensor_h'][0]
        Gap_size=IN_Info['Gap_size'][0]
        
        Win_Start=IN_Info['Win_Start'][0]
        Win_End=IN_Info['Win_End'][0]
        Win_Interval=IN_Info['Win_Interval'][0]

        Cut_loc_Start=IN_Info['Cut_loc_Start'][0]
        Cut_loc_End=IN_Info['Cut_loc_End'][0]
        Cut_loc_Interval=IN_Info['Cut_loc_Interval'][0]

        Sensor_Line_Start=IN_Info['Sensor_Line_Start'][0]
        Sensor_Line_End=IN_Info['Sensor_Line_End'][0]
        Sensor_Line_Interval=IN_Info['Sensor_Line_Interval'][0]
        
        num_sig_out,num_sig_in,re_out,re_in=pole_Info(poleid)
        
        print('### 해석 전주  {}..{}/{}  : 관통형 측정신호 개수는 {}, 후크형 측정신호 개수는 {} '.format(poleid,current_num,total_num,num_sig_in,num_sig_out)) 
        
        Signal_Analy_result= pd.DataFrame(columns=['poleid','s_num','stype','CH','simu_rate','S_position','X_location','mac_x_v','mac_y_v','mac_z_v',
                                                   'corrcoef_x','corrcoef_y','corrcoef_z',
                                                'total','X_point_X','X_point_Y','prediction'])

        Temp_signal_Analy_result_SET= pd.DataFrame(columns=['poleid','s_num','stype','CH','simu_rate','S_position','X_location','mac_x_v','mac_y_v','mac_z_v',
                                                                    'corrcoef_x','corrcoef_y','corrcoef_z','X_point_X','X_point_Y','total','prediction'])
        
        slack.slack(str(current_num)+'/'+str(total_num)+'_____전주 '+poleid+' 분석중')
        
        for kk in range(num_sig_out+num_sig_in): # 측정 신호 세트 스캔
            
                
                if kk <num_sig_out:
                    stype='OUT'
                    num=int(re_out['measno'][kk])
                    
                    limit_x=IN_Info['limit_out_x'][0]
                    limit_y=IN_Info['limit_out_y'][0]
                    limit_z=IN_Info['limit_out_z'][0]
                    if breakstate=='B':
                        limit_x=0
                        limit_y=0
                        limit_z=0
                    select_x_data,select_y_data,select_z_data,select_ch=Signal_Info_select_data_out(poleid,stype,num,limit_x,limit_y,limit_z,IN_Info['Out_Trans_para'][0])
                    
                    # 파단위치 계산을 위한 측정 위치 정보--------
                    start_point=re_out['stheight'][kk]
                    start_angle=re_out['stdegree'][kk]
                    end_angle=re_out['eddegree'][kk]
                    if start_angle<end_angle:
                        angle_interval=(end_angle-start_angle)/7
                    else:
                        angle_interval=((360-start_angle)+end_angle)/7
                    #---------------------------------------------
                    
                    
                    
                    scan_len=(re_out['edheight'][kk]-re_out['stheight'][kk])*1000
                    if len(select_x_data)>0:
                        print('    -- 분석 신호 후크형 신호 :  {}/{} , 측정길이 : {} mm , 신호 개수 : {} ,  속도 : {} points/cm'.format(num,num_sig_out,scan_len,len(select_x_data[0]),round(len(select_x_data[0])*10/scan_len,3)))
                    
                else:
                    stype='IN'
                    
                    num=int(re_in['measno'][(kk-num_sig_out)])
                    limit_x=IN_Info['limit_in_x'][0]
                    limit_y=IN_Info['limit_in_y'][0]
                    limit_z=IN_Info['limit_in_z'][0]
                    if breakstate=='B':
                        limit_x=0
                        limit_y=0
                        limit_z=0
                        
                    select_x_data,select_y_data,select_z_data,select_ch=Signal_Info_select_data_in(poleid,stype,num,limit_x,IN_Info['IN_Trans_para'][0])
                    
                    # 파단위치 계산을 위한 측정 위치 정보--------
                    start_point=re_in['depth'][(kk-num_sig_out)]
                    start_angle=re_in['stdegree'][(kk-num_sig_out)]
                    end_angle=start_angle+360
                    angle_interval=360/8
                    
                    #---------------------------------------------    
                    
                    scan_len=(re_in['stheight'][(kk-num_sig_out)]+re_in['depth'][(kk-num_sig_out)])*1000
                    if len(select_x_data)>0:
                        print('    -- 분석 신호 관통형 신호 :  {}/{} , 측정길이 : {} mm , 신호 개수 : {} ,  속도 : {} points/cm'.format(num,num_sig_in,scan_len,len(select_x_data[0]),round(len(select_x_data[0])*10/scan_len,3)))
                
                for i ,index in enumerate(select_ch):  # 선택죈 채널 스캔 
                    
                    Temp_signal_Analy_result= pd.DataFrame(columns=['poleid','s_num','stype','CH','simu_rate','S_position','X_location','mac_x_v','mac_y_v','mac_z_v',
                                                                    'corrcoef_x','corrcoef_y','corrcoef_z','X_point_X','X_point_Y','total','prediction'])
                    
                    for tt in range(Win_Start,Win_End,Win_Interval):  #시물레이션 신호 폭 변경(측정 길이를 임의적으로 조정) => simu_rate
                        tt=tt/100
                        
                                  
                        for k in range(Cut_loc_End,Cut_loc_Start,-Cut_loc_Interval):   # 파단 위치 스캔  X_position
                            Bs=mag_m.pole_simulation(Main_mag_s,Gap_size,k,len(select_x_data[0]),scan_len*tt,HallSensor_h)
                            # 파단 위치 계산을 위한 ---------------------
                            X_point_Y=0   #파단위치(길이 하->상)
                            X_point_X=0   #파단위치(각도)
                                       
                            if stype=='OUT':
                                   X_point_Y=round((start_point*1000+scan_len*k/100)/1000,2)
                            if stype=='IN':
                                   
                                if scan_len*tt<start_point*1000:
                                    X_point_Y=-round((start_point*1000-scan_len*k/100)/1000,2)
                                else:
                                    X_point_Y=-round((scan_len*k/100-start_point*1000)/1000,2)
                            X_point_X=start_angle+angle_interval*(index-1)
                            
                            if X_point_X > 360:
                                X_point_X =X_point_X -360 
                            #---------------------------------------------------     
                            for hh in [2]:
                                for j in range(Sensor_Line_Start,Sensor_Line_End,Sensor_Line_Interval):  # 센서 위치 스캔  S_location
                                    
                                    if hh==0:
                                        data_x = np.array([[l,w] for l,w in zip((Bs[:,j,0]),select_x_data[i])])
                                        data_y = np.array([[l,w] for l,w in zip((Bs[:,j,1]),select_y_data[i])])
                                        data_z = np.array([[l,w] for l,w in zip((Bs[:,j,2]),select_z_data[i])])
                                    elif hh==1:
                                        data_x = np.array([[-l,w] for l,w in zip((Bs[:,j,0]),select_x_data[i])])
                                        data_y = np.array([[-l,w] for l,w in zip((Bs[:,j,1]),select_y_data[i])])
                                        data_z = np.array([[-l,w] for l,w in zip((Bs[:,j,2]),select_z_data[i])])
                                    elif hh==2:
                                        data_x = np.array([[l,w] for l,w in zip(np.gradient(Bs[:,j,0],1),select_x_data[i])])
                                        data_y = np.array([[l,w] for l,w in zip(np.gradient(Bs[:,j,1],1),select_y_data[i])])
                                        data_z = np.array([[l,w] for l,w in zip(np.gradient(Bs[:,j,2],1),select_z_data[i])])
                                    elif hh==3:
                                        data_x = np.array([[-l,w] for l,w in zip(np.gradient(Bs[:,j,0],1),select_x_data[i])])
                                        data_y = np.array([[-l,w] for l,w in zip(np.gradient(Bs[:,j,1],1),select_y_data[i])])
                                        data_z = np.array([[-l,w] for l,w in zip(np.gradient(Bs[:,j,2],1),select_z_data[i])])
                                        
                                    phi_n_x, mode_scaling=normalize_phi(data_x)
                                    mac_x_v=mac(phi_n_x[:,0],phi_n_x[:,1])
                                    
                                    phi_n_y, mode_scaling=normalize_phi(data_y)
                                    mac_y_v=mac(phi_n_y[:,0],phi_n_y[:,1])
                                    
                                    phi_n_z, mode_scaling=normalize_phi(data_z)
                                    mac_z_v=mac(phi_n_z[:,0],phi_n_z[:,1])
                                    
                                    
                                    
                                    corr_x = signal.correlate(phi_n_x[:,0],phi_n_x[:,1])
                                    corr_x /= np.max(corr_x)
                                    corrcoef_x=np.corrcoef(phi_n_x[:,0],phi_n_x[:,1])[0, 1]
                                    
                                    corr_y = signal.correlate(phi_n_y[:,0],phi_n_y[:,1])
                                    corr_y /= np.max(corr_y)
                                    corrcoef_y=np.corrcoef(phi_n_y[:,0],phi_n_y[:,1])[0, 1]
                                    
                                    
                                    corr_z = signal.correlate(phi_n_z[:,0],phi_n_z[:,1])
                                    corr_z /= np.max(corr_z)
                                    corrcoef_z=np.corrcoef(phi_n_z[:,0],phi_n_z[:,1])[0, 1]
                                    
                                    
                                    
                                    total_v=round(mac_x_v+mac_y_v+mac_z_v+abs(corrcoef_x)+abs(corrcoef_y)+abs(corrcoef_z),2)
                                    
                                              
                                    
                                    if mac_x_v > limit_MAC_X and mac_y_v > limit_MAC_Y and mac_z_v > limit_MAC_Z and abs(corrcoef_x)>limit_corrcoef_X and abs(corrcoef_y)>limit_corrcoef_Y and abs(corrcoef_z)>limit_corrcoef_Z: 
                                        temp_new_data={'poleid':poleid,'s_num':num,'stype':stype, 'simu_rate':tt,'X_location':k,'CH':select_ch[i],'S_position':j,'mac_x_v':round(mac_x_v,3),
                                                   'mac_y_v':round(mac_y_v,3),'mac_z_v':round(mac_z_v,3),
                                                   'corrcoef_x':round(corrcoef_x,3),'corrcoef_y':round(corrcoef_y,3),'corrcoef_z':round(corrcoef_z,3),
                                                   'X_point_X':round(X_point_X,0),'X_point_Y':round(X_point_Y,2),'total':total_v,'prediction':'NG'}
                                        
                                        Temp_signal_Analy_result=Temp_signal_Analy_result.append(temp_new_data,ignore_index=True)
                                        detect_num=detect_num+1
                
                    if len(Temp_signal_Analy_result)>0 :                       
                        Temp_signal_Analy_result=Temp_signal_Analy_result.sort_values(by=['total'], axis=0,ascending=False)
                        Temp_signal_Analy_result_SET=Temp_signal_Analy_result_SET.append(Temp_signal_Analy_result.loc[0],ignore_index=True)
                        
                # data set  결과 임시 저장 Dataframe
        if len(Temp_signal_Analy_result_SET)>0:
                        
            Temp_signal_Analy_result_SET=Temp_signal_Analy_result_SET.sort_values(by=['total'], axis=0,ascending=False)
            Signal_Analy_result=Signal_Analy_result.append(Temp_signal_Analy_result_SET)
        # 2021_11_2 추가
        elif breakstate=='X':
            temp_new_data={'poleid':poleid,'prediction':'X'}
            Signal_Analy_result=Signal_Analy_result.append(temp_new_data,ignore_index=True)
        ####################################################################################     
        else:
            temp_new_data={'poleid':poleid,'prediction':'OK'}
            Signal_Analy_result=Signal_Analy_result.append(temp_new_data,ignore_index=True)
            #Signal_Analy_result.append()                   
        if detect_num>0:
            Signal_Analy_result.to_csv('./AUTO_analysis/'+simplenowday+"_result/"+dir+'/output/ref/'+poleid+'.csv')
            
        return Signal_Analy_result                    


       
import json
import datetime

def pole_State_Analysis_all(IN_Info,Pole_list,dir,simplenowday): 
    
    
        d_today = datetime.date.today()
        today_str=d_today.strftime("%Y%m%d")
        
        
        result= pd.DataFrame(columns=['poleid','s_num','stype','CH','simu_rate','S_position','X_location','mac_x_v','mac_y_v','mac_z_v','corrcoef_x','corrcoef_y','corrcoef_z',
                                                'total','X_point_X','X_point_Y','prediction'])
        
        Pole_list=Pole_list.reset_index()
        for index,poleid in enumerate(Pole_list['poleid']):
            
                   kk=pole_State_Analysis(IN_Info,poleid,index+1,len(Pole_list['poleid']),dir,Pole_list['breakstate'][index],simplenowday)
                   result=result.append(kk)    
        
        result['breakstate']=np.NaN
        for i,pid in enumerate(result['poleid']):
            result['breakstate'][i]=Pole_list['breakstate'][Pole_list.loc[Pole_list['poleid']==pid].index].to_list()[0]
            
        return  result
       

def conf_file_open(dir):
    # reading the data from the file
    with open('./pole_anal/analysis/'+dir+'/input/'+dir+'_pole_auto_conf.json') as f:
        data = f.read()
    input_conf = json.loads(data)
    
    # diagstate
    #  -	미측정
    # MF	측정완료
    # AP	분석중(1차분석 완료)
    # AF	분석완료
    # breatestate
    # N	정상
    # B	파단
    # U	보류
    # X	측정불가
    # -	없음

    Pole_list=PDB.get_pole_list(dir)
    
    
    
    
    
    #Pole_list= pd.read_csv('analysis/'+dir+'/input/pole_list.csv', engine='python')
    return IN_Info,Pole_list
        
def conf_file_open_schedule(dir,day):
    
    # reading the data from the file
    with open('analysis/'+dir+'/input/'+dir+'_pole_auto_conf.json') as f:
        data = f.read()
    input_conf = json.loads(data)
       

    IN_Info = pd.json_normalize(input_conf)
    
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

    Pole_list=PDB.get_pole_list(dir)
    
    for i,poleid in enumerate(Pole_list['poleid']): 
        hh=Pole_list['endtime'][i].strftime('%Y-%m-%d')
        if hh!=day:
            Pole_list=Pole_list.drop([i]) 
    
    #Pole_list= pd.read_csv('analysis/'+dir+'/input/pole_list.csv', engine='python')
    
    return IN_Info,Pole_list

def update_anal_result(df_anal_result):

    anal_result_list = []
    anal_result_success = True

    # poleid, anal_result, break_cnt, break_loc_height_list, break_loc_degree_list, anal_comment

    # checking result
    for idx, row in df_anal_result.iterrows():
        poleid = row['poleid']
        prediction = row['prediction']
        break_num = row['break_num']
        break_loc = row['break_loc']
        opinion = row['opinion']

        try:
            if prediction == 'NG':
                anal_result = 'B'

                if break_num <= 0:
                    logger.error('update_anal_result) break_num <= 0  (poleid={})'.format(poleid))
                    anal_result_success = False
                    break

                break_loc_list = ast.literal_eval(break_loc)
                if len(break_loc_list) != 2:
                    logger.error('update_anal_result) break_loc malformed (poleid={})'.format(poleid))
                    anal_result_success = False
                    break

                break_loc_height_list = break_loc_list[1]
                break_loc_degree_list = break_loc_list[0]
                anal_comment = opinion
                break_cnt = len(break_loc_height_list)

                if len(break_loc_height_list) != len(break_loc_degree_list):
                    logger.error('update_anal_result) break_cnt mismatched (poleid={})'.format(poleid))
                    anal_result_success = False
                    break

                if break_num != len(break_loc_height_list):
                    logger.error('update_anal_result) break_cnt mismatched (poleid={})'.format(poleid))
                    anal_result_success = False
                    break
            else:
                anal_result = 'N'
                break_loc_height_list = []
                break_loc_degree_list = []
                anal_comment = ''
                break_cnt = 0

            tmp = [ poleid, anal_result, break_cnt, break_loc_height_list, break_loc_degree_list, anal_comment ]
            anal_result_list.append(tmp)
        except Exception as e:
            logger.error('update_anal_result) exception occurred (poleid={})'.format(poleid))
            logger.error(str(e))
            anal_result_success = False
            break

    if anal_result_success == False:
        logger.error('anal_result check failed')
        return

    logger.info('----------------------------')
    logger.info('anal_result checking success')
    logger.info('----------------------------')

    for i in range(len(anal_result_list)):
        tmp = anal_result_list[i]
        logger.info(tmp)

        result = PDB.update_anal_result(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5])
        if result is False:
            logger.error('update_anal_result failed (poleid={})'.format(poleid))
            break
        else:
            logger.info('update_anal_result success (poleid={})'.format(poleid))

    return
#IN_Info,Pole_list=conf_file_open() 

#pole_State_Analysis_all(IN_Info,Pole_list)

#pole_State_Scan(Pole_list)

