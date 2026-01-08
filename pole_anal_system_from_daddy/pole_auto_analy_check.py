# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 17:53:04 2021

@author: 스마트제어계측
"""
import bottleneck as bn
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtCore import Qt
import numpy as np
import pandas as pd
import pyqtgraph as pg
import sys,os
import random
import csv
import peakutils
import scipy
import time
from time import sleep
from scipy import signal
import math
import threading

import magpylib_simulation_m as mag_m
import poledb as PDB
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
import pandas as pd
from scipy.signal import detrend
import pole_auto_analy_command as pole_lib
PDB.poledb_init()

import json






class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        global curve,curve1,FD_table_1,Bridge_information
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle('Smart c&S...File load for display and analysis v0.1')
        self.central_widget = QtGui.QStackedWidget()
        self.setCentralWidget(self.central_widget)
        self.login_widget = LoginWidget(self)
        
        self.login_widget.label_image.setPixmap(QtGui.QPixmap("smartcs.png"))
        
        self.login_widget.button_file_open.clicked.connect(self.pushButtonClicked_file_open)
        
        
        curve=self.login_widget.plot3.plot(pen='y')
        
        curve1=self.login_widget.plot4.plot(pen='y')
        
        self.login_widget.button_Analysis.clicked.connect(self.pushButtonClicked_Analysis)
        
        self.login_widget.button_Analysis_back.clicked.connect(self.pushButtonClicked_Analysis_back)
        
        self.login_widget.button_Analysis_all.clicked.connect(self.pushButtonClicked_Analysis_delete)
        
        self.login_widget.button_plot_clear.clicked.connect(self.pushButtonClicked_maker_file)
        
        self.login_widget.button_file_upload.clicked.connect(self.pushButtonClicked_file_upload)
        
        self.central_widget.addWidget(self.login_widget)
        
       
        
        
    def pushButtonClicked_file_upload(self):     
        
        
        fname = QtGui.QFileDialog.getOpenFileName(self)
        
        final_file= pd.read_csv(fname[0], engine='python')
       
        print(final_file)
        
        pole_lib.update_anal_result(final_file)
        
        
        
    def pushButtonClicked_maker_file(self):

        # 임시로 저장        
        result= DATA
        breakstate= 'N'
        breakstate_desc=''
        final_result= pd.DataFrame(columns=['poleid','breakstate','prediction','break_num','break_loc',"opinion"])
        for index,poleid in enumerate(result['poleid']):
            
            #if Pole_list['진단상태'][index]!='미측정':
              
               
               temp_pd=result.loc[result['poleid']==poleid].reset_index()

               prediction=temp_pd['prediction'][0]
                            
               if len(temp_pd)>0 and prediction=='NG':
                  break_num=len(temp_pd)
               else:
                  break_num=0 
               
               break_loc=[]
               opinion=''
               if len(temp_pd)>0 and prediction=='NG':
                   break_loc=[(temp_pd['X_point_X'].tolist()),temp_pd['X_point_Y'].tolist()]
                   temp_DATA=GDATA.loc[(GDATA['poleid']==poleid) & (GDATA['breakstate']=='B')].reset_index()

                   if len(temp_DATA)>0:
                       breakstate= 'B'
                       breakstate_desc='Site opinion : Break'
                   else:
                       breakstate= 'N'
                       breakstate_desc=''
                   opinion=breakstate_desc+'\n'
                   for i in range(len(temp_pd)):
                       
                       opinion=opinion+'data set :'+str(temp_pd['s_num'][i])+'  type :'+temp_pd['stype'][i]+'  CH : '+str(temp_pd['CH'][i])+'\n'
                   print(opinion)
                   temp_new_data={'poleid':poleid,'breakstate':breakstate,'prediction':prediction,'break_num':break_num,'break_loc':break_loc,"opinion":opinion}
               else:
                   
                   temp_new_data={'poleid':poleid,'breakstate':breakstate,'prediction':prediction,'break_num':break_num,'break_loc':break_loc,"opinion":opinion}
                   
               final_result=final_result.append(temp_new_data,ignore_index=True)
        final_result=final_result.drop_duplicates(['poleid'])
        final_result.to_csv(file_dir+'final_result_modify-'+Onlyfilename)    
        
    
    def pushButtonClicked_Analysis_delete(self): 
        
       
        poleid=self.login_widget.ComboBox_D_ID.currentText()    
        
        index=self.login_widget.ComboBox_D_ID.currentIndex()
        
        temp_DATA=DATA.loc[(DATA['poleid']==poleid)].reset_index()
        
        if len(temp_DATA)>1:
            
            DATA.drop(DATA.loc[(DATA['poleid']==poleid) & (DATA['s_num']==num) & (DATA['stype']==stype) & (DATA['CH']==CH+1) ].index, inplace=True)
            DATA.reset_index(drop=True, inplace=True)
        else:
            ii=DATA.index[DATA['poleid']==poleid].tolist()
            DATA['prediction'][ii[0]]='OK'
            print('#########################',ii)
        
        ##########################
        data.drop(data.loc[(data['poleid']==poleid) & (data['s_num']==num) & (data['stype']==stype) & (data['CH']==CH+1) ].index, inplace=True)
        data.reset_index(drop=True, inplace=True)
        
        print(data)
        self.login_widget.ComboBox_D_ID.removeItem(index)
        
        self.login_widget.label_Analysis_pole_total_num.setText(str(len(data['poleid'])))
        ##############################
        
        print(DATA)
        
        break_pole=data['poleid']
        break_pole=set(break_pole)
        break_pole=list(break_pole)
        total_pole_num=DATA['poleid']
        total_pole_num=set(total_pole_num)
        total_pole_num=list(total_pole_num)
        
        break_pole_list='# 총 분석 전 주 개수 : '+ str(len(total_pole_num))+'\n'+'# 파단 예측 전주: \n'
        for item in break_pole:
            break_pole_list=break_pole_list+item+'\n'
        break_pole_list=break_pole_list+'....'+str(len(break_pole))+'...'+str(round(len(break_pole)/len(total_pole_num)*100,1)) 
        self.login_widget.label_current_pole.setText(break_pole_list)
        
        self.login_widget.plot1.clear()   
        self.login_widget.plot2.clear()   
        self.login_widget.plot3.clear()
        self.login_widget.plot4.clear()   
        self.login_widget.plot5.clear()   
        self.login_widget.plot6.clear()
        self.login_widget.plot7.clear()   
        self.login_widget.plot8.clear()   
        self.login_widget.plot9.clear()
            
    def pushButtonClicked_Analysis(self):
        
        global poleid, stype,num,CH
        
        
        index=self.login_widget.ComboBox_D_ID.currentIndex()
        
        if index == len(data['poleid'])-1:
           index=0
        else:
           index=index+1 
           
        
        self.login_widget.ComboBox_D_ID.setCurrentIndex(index)
        
        
        poleid=self.login_widget.ComboBox_D_ID.currentText()
        
            
        print('index',index)
        stype=data['stype'][index]
        num=int(data['s_num'][index])
        CH=int(data['CH'][index])-1
        
        index_ch=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
            
        self.login_widget.plot1.clear()   
        self.login_widget.plot2.clear()   
        self.login_widget.plot3.clear()
        
        if stype=='OUT':
            select_x_data,select_y_data,select_z_data,select_ch=pole_lib.Signal_Info_select_data_out(poleid,stype,num,0,0,0,IN_Info['Out_Trans_para'][0])
            
            sig_x=PDB.get_meas_data(poleid,num,stype,'x')
            sig_y=PDB.get_meas_data(poleid,num,stype,'y')
            sig_z=PDB.get_meas_data(poleid,num,stype,'z')
            index_ch=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
            
            for index1,i in enumerate(index_ch):
                
                sig_x[i]=IN_Info['Out_Trans_para'][0]*sig_x[i]
                sig_y[i]=IN_Info['Out_Trans_para'][0]*sig_y[i]
                sig_z[i]=IN_Info['Out_Trans_para'][0]*sig_z[i]
                self.login_widget.plot1.plot(sig_x[i] ,pen=(100,index1*20,index1*20)) 
                self.login_widget.plot2.plot(sig_y[i], pen=(index1*20,100,index1*20)) 
                self.login_widget.plot3.plot(sig_z[i], pen=(index1*20,index1*20,100)) 
        elif stype=='IN':
            select_x_data,select_y_data,select_z_data,select_ch=pole_lib.Signal_Info_select_data_in(poleid,stype,num,0,IN_Info['IN_Trans_para'][0])
            sig_x=PDB.get_meas_data(poleid,num,stype,'x')
            sig_y=PDB.get_meas_data(poleid,num,stype,'x')
            sig_z=PDB.get_meas_data(poleid,num,stype,'x')
            index_ch=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
            for index1,i in enumerate(index_ch):
                
                sig_x[i]=IN_Info['IN_Trans_para'][0]*sig_x[i]
                sig_y[i]=IN_Info['IN_Trans_para'][0]*sig_y[i]
                sig_z[i]=IN_Info['IN_Trans_para'][0]*sig_z[i]
                self.login_widget.plot1.plot(sig_x[i] ,pen=(255,index1*10,index1*20)) 
                self.login_widget.plot2.plot(sig_y[i], pen=(index1*20,255,index1*20)) 
                self.login_widget.plot3.plot(sig_z[i], pen=(index1*20,index1*20,255)) 
            

            
            
         
        self.login_widget.plot4.clear()   
        self.login_widget.plot5.clear()   
        self.login_widget.plot6.clear()
        
        self.login_widget.plot1.plot(sig_x[index_ch[CH]] ,pen=(255,0,0)) 
        self.login_widget.plot2.plot(sig_y[index_ch[CH]], pen=(0,255,0)) 
        self.login_widget.plot3.plot(sig_z[index_ch[CH]], pen=(0,0,255)) 
        
        
        self.login_widget.plot4.plot(sig_x[index_ch[CH]] ,pen=(255,0,0)) 
        self.login_widget.plot5.plot(sig_y[index_ch[CH]], pen=(0,255,0)) 
        self.login_widget.plot6.plot(sig_z[index_ch[CH]], pen=(0,0,255)) 
        
        tt=float(data['simu_rate'][index])
        k=int(data['X_location'][index])
        j=int(data['S_position'][index])
        
        
        Main_mag_s=IN_Info['Main_mag_s'][0]
        Gap_size=IN_Info['Gap_size'][0]
        HallSensor_h=IN_Info['HallSensor_h'][0]
        
        
        
        temp_pd=Scan_data.loc[(Scan_data['poleid']==poleid) & (Scan_data['stype']==stype) & (Scan_data['s_num']==num) ].reset_index()
        
        # N	정상
        # B	파단
        # U	보류
        # X	측정불가
        # -	없음
        if Scan_data['breakstate'][Scan_data.loc[Scan_data['poleid']==poleid].index].to_list()[0]=='N':
            gg='정상'
        elif Scan_data['breakstate'][Scan_data.loc[Scan_data['poleid']==poleid].index].to_list()[0]=='B':
            gg='파단'
        elif Scan_data['breakstate'][Scan_data.loc[Scan_data['poleid']==poleid].index].to_list()[0]=='U':
            gg='보류'    
        elif Scan_data['breakstate'][Scan_data.loc[Scan_data['poleid']==poleid].index].to_list()[0]=='X':
            gg='측정불가'
        elif Scan_data['breakstate'][Scan_data.loc[Scan_data['poleid']==poleid].index].to_list()[0]=='-':
            gg='없음'    
            
        self.login_widget.label_D_ID.setText('Pole ID: 현장 측정 결과 >> '+gg) 
        
        self.login_widget.lineEdit_limit_x.setText(str(index+1)+"...."+stype+'/'+str(num)+'/'+str(CH+1))
        
        self.login_widget.lineEdit_limit_y.setText(str(int(temp_pd['Scan_len'][0]))+'/'+str(temp_pd['Scan_V'][0]))
        
        if stype=='OUT':
            self.login_widget.lineEdit_limit_z.setText(str(int(temp_pd['PtP_x'][0]))+'/'+str(int(temp_pd['PtP_y'][0]))+'/'+str(int(temp_pd['PtP_z'][0])))
        elif stype=='IN':
            self.login_widget.lineEdit_limit_z.setText(str(int(temp_pd['PtP_x'][0]))) 
        
        
        temp_data=data.loc[(data['poleid']==poleid) & (data['stype']==stype) & (data['s_num']==num)].reset_index(drop=True)
        

        self.login_widget.lineEdit_limit_corrcoef.setText(str(round(temp_data['mac_x_v'][0],2))+'/'+str(round(temp_data['mac_y_v'][0],2))+
                                                          '/'+str(round(temp_data['mac_z_v'][0],2)))
        
        self.login_widget.lineEdit_limit_MAC_x.setText(str(round(temp_data['corrcoef_x'][0],2))+'/'+str(round(temp_data['corrcoef_y'][0],2))+
                                                          '/'+str(round(temp_data['corrcoef_z'][0],2)))
        
        
        self.login_widget.lineEdit_limit_MAC_z.setText(str((temp_data['simu_rate'][0]))+'/'+str((temp_data['S_position'][0]))+
                                                          '/'+str((temp_data['X_location'][0])))
        
        self.login_widget.lineEdit_limit_MAC_y.setText(str((temp_data['total'][0])))
        
        scan_len=temp_pd['Scan_len'][0]
               
        print('scan_len',scan_len)       
        Bs=mag_m.pole_simulation(Main_mag_s,Gap_size,k,len(select_x_data[0]),scan_len*tt,HallSensor_h)
        
        data_x = np.array([[l,w] for l,w in zip(np.gradient(Bs[:,j,0]),select_x_data[CH])])
        data_y = np.array([[l,w] for l,w in zip(np.gradient(Bs[:,j,1]),select_y_data[CH])])
        data_z = np.array([[l,w] for l,w in zip(np.gradient(Bs[:,j,2]),select_z_data[CH])])
                                        
        phi_n_x, mode_scaling=pole_lib.normalize_phi(data_x)
   
                    
        phi_n_y, mode_scaling=pole_lib.normalize_phi(data_y)
   
                                    
        phi_n_z, mode_scaling=pole_lib.normalize_phi(data_z)
        
        
        self.login_widget.plot7.clear()   
        self.login_widget.plot8.clear()   
        self.login_widget.plot9.clear()
        self.login_widget.plot7.plot(phi_n_x[:,0], pen=(255,0,0)) 
        self.login_widget.plot7.plot(phi_n_x[:,1], pen=(255,255,255))
        self.login_widget.plot8.plot(phi_n_y[:,0], pen=(0,0,255))
        self.login_widget.plot8.plot(phi_n_y[:,1], pen=(255,255,255))
        self.login_widget.plot9.plot(phi_n_z[:,0], pen=(0,255,0)) 
        self.login_widget.plot9.plot(phi_n_z[:,1], pen=(255,255,255)) 
        # QtCore.QCoreApplication.processEvents()

    def pushButtonClicked_Analysis_back(self):
        
        global poleid, stype,num,CH
        
        
        index=self.login_widget.ComboBox_D_ID.currentIndex()
        
        if index == 0:
           index=len(data['poleid'])-1
        else:
           index=index-1 
           
        
        self.login_widget.ComboBox_D_ID.setCurrentIndex(index)
        
        
        poleid=self.login_widget.ComboBox_D_ID.currentText()
        
        
        print('index',index)
        stype=data['stype'][index]
        num=int(data['s_num'][index])
        CH=int(data['CH'][index])-1
        
        index_ch=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
            
        self.login_widget.plot1.clear()   
        self.login_widget.plot2.clear()   
        self.login_widget.plot3.clear()
        
        if stype=='OUT':
            select_x_data,select_y_data,select_z_data,select_ch=pole_lib.Signal_Info_select_data_out(poleid,stype,num,0,0,0,IN_Info['Out_Trans_para'][0])
            
            sig_x=PDB.get_meas_data(poleid,num,stype,'x')
            sig_y=PDB.get_meas_data(poleid,num,stype,'y')
            sig_z=PDB.get_meas_data(poleid,num,stype,'z')
            index_ch=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
            
            for index1,i in enumerate(index_ch):
                
                sig_x[i]=IN_Info['Out_Trans_para'][0]*sig_x[i]
                sig_y[i]=IN_Info['Out_Trans_para'][0]*sig_y[i]
                sig_z[i]=IN_Info['Out_Trans_para'][0]*sig_z[i]
                self.login_widget.plot1.plot(sig_x[i] ,pen=(100,index1*20,index1*20)) 
                self.login_widget.plot2.plot(sig_y[i], pen=(index1*20,100,index1*20)) 
                self.login_widget.plot3.plot(sig_z[i], pen=(index1*20,index1*20,100)) 
        elif stype=='IN':
            select_x_data,select_y_data,select_z_data,select_ch=pole_lib.Signal_Info_select_data_in(poleid,stype,num,0,IN_Info['IN_Trans_para'][0])
            sig_x=PDB.get_meas_data(poleid,num,stype,'x')
            sig_y=PDB.get_meas_data(poleid,num,stype,'x')
            sig_z=PDB.get_meas_data(poleid,num,stype,'x')
            index_ch=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
            for index1,i in enumerate(index_ch):
                
                sig_x[i]=IN_Info['IN_Trans_para'][0]*sig_x[i]
                sig_y[i]=IN_Info['IN_Trans_para'][0]*sig_y[i]
                sig_z[i]=IN_Info['IN_Trans_para'][0]*sig_z[i]
                self.login_widget.plot1.plot(sig_x[i] ,pen=(255,index1*10,index1*20)) 
                self.login_widget.plot2.plot(sig_y[i], pen=(index1*20,255,index1*20)) 
                self.login_widget.plot3.plot(sig_z[i], pen=(index1*20,index1*20,255)) 
            

            
            
         
        self.login_widget.plot4.clear()   
        self.login_widget.plot5.clear()   
        self.login_widget.plot6.clear()
        
        self.login_widget.plot1.plot(sig_x[index_ch[CH]] ,pen=(255,0,0)) 
        self.login_widget.plot2.plot(sig_y[index_ch[CH]], pen=(0,255,0)) 
        self.login_widget.plot3.plot(sig_z[index_ch[CH]], pen=(0,0,255)) 
        
        
        self.login_widget.plot4.plot(sig_x[index_ch[CH]] ,pen=(255,0,0)) 
        self.login_widget.plot5.plot(sig_y[index_ch[CH]], pen=(0,255,0)) 
        self.login_widget.plot6.plot(sig_z[index_ch[CH]], pen=(0,0,255)) 
        
        tt=float(data['simu_rate'][index])
        k=int(data['X_location'][index])
        j=int(data['S_position'][index])
        
        
        Main_mag_s=IN_Info['Main_mag_s'][0]
        Gap_size=IN_Info['Gap_size'][0]
        HallSensor_h=IN_Info['HallSensor_h'][0]
        
        
        
        temp_pd=Scan_data.loc[(Scan_data['poleid']==poleid) & (Scan_data['stype']==stype) & (Scan_data['s_num']==num) ].reset_index()
        
        # N	정상
        # B	파단
        # U	보류
        # X	측정불가
        # -	없음
        if Scan_data['breakstate'][Scan_data.loc[Scan_data['poleid']==poleid].index].to_list()[0]=='N':
            gg='정상'
        elif Scan_data['breakstate'][Scan_data.loc[Scan_data['poleid']==poleid].index].to_list()[0]=='B':
            gg='파단'
        elif Scan_data['breakstate'][Scan_data.loc[Scan_data['poleid']==poleid].index].to_list()[0]=='U':
            gg='보류'    
        elif Scan_data['breakstate'][Scan_data.loc[Scan_data['poleid']==poleid].index].to_list()[0]=='X':
            gg='측정불가'
        elif Scan_data['breakstate'][Scan_data.loc[Scan_data['poleid']==poleid].index].to_list()[0]=='-':
            gg='없음'    
            
        self.login_widget.label_D_ID.setText('Pole ID: 현장 측정 결과 >> '+gg) 
        
        self.login_widget.lineEdit_limit_x.setText(str(index+1)+"...."+stype+'/'+str(num)+'/'+str(CH+1))
        
        self.login_widget.lineEdit_limit_y.setText(str(int(temp_pd['Scan_len'][0]))+'/'+str(temp_pd['Scan_V'][0]))
        
        if stype=='OUT':
            self.login_widget.lineEdit_limit_z.setText(str(int(temp_pd['PtP_x'][0]))+'/'+str(int(temp_pd['PtP_y'][0]))+'/'+str(int(temp_pd['PtP_z'][0])))
        elif stype=='IN':
            self.login_widget.lineEdit_limit_z.setText(str(int(temp_pd['PtP_x'][0]))) 
        
        
        temp_data=data.loc[(data['poleid']==poleid) & (data['stype']==stype) & (data['s_num']==num)].reset_index(drop=True)
        

        self.login_widget.lineEdit_limit_corrcoef.setText(str(round(temp_data['mac_x_v'][0],2))+'/'+str(round(temp_data['mac_y_v'][0],2))+
                                                          '/'+str(round(temp_data['mac_z_v'][0],2)))
        
        self.login_widget.lineEdit_limit_MAC_x.setText(str(round(temp_data['corrcoef_x'][0],2))+'/'+str(round(temp_data['corrcoef_y'][0],2))+
                                                          '/'+str(round(temp_data['corrcoef_z'][0],2)))
        
        
        self.login_widget.lineEdit_limit_MAC_z.setText(str((temp_data['simu_rate'][0]))+'/'+str((temp_data['S_position'][0]))+
                                                          '/'+str((temp_data['X_location'][0])))
        
        self.login_widget.lineEdit_limit_MAC_y.setText(str((temp_data['total'][0])))
        
        scan_len=temp_pd['Scan_len'][0]
               
        print('scan_len',scan_len)       
        Bs=mag_m.pole_simulation(Main_mag_s,Gap_size,k,len(select_x_data[0]),scan_len*tt,HallSensor_h)
        
        data_x = np.array([[l,w] for l,w in zip((Bs[:,j,0]),select_x_data[CH])])
        data_y = np.array([[l,w] for l,w in zip((Bs[:,j,1]),select_y_data[CH])])
        data_z = np.array([[l,w] for l,w in zip((Bs[:,j,2]),select_z_data[CH])])
                                        
        phi_n_x, mode_scaling=pole_lib.normalize_phi(data_x)
   
                    
        phi_n_y, mode_scaling=pole_lib.normalize_phi(data_y)
   
                                    
        phi_n_z, mode_scaling=pole_lib.normalize_phi(data_z)
        
        
        self.login_widget.plot7.clear()   
        self.login_widget.plot8.clear()   
        self.login_widget.plot9.clear()
        self.login_widget.plot7.plot(phi_n_x[:,0], pen=(255,0,0)) 
        self.login_widget.plot7.plot(phi_n_x[:,1], pen=(255,255,255))
        self.login_widget.plot8.plot(phi_n_y[:,0], pen=(0,0,255))
        self.login_widget.plot8.plot(phi_n_y[:,1], pen=(255,255,255))
        self.login_widget.plot9.plot(phi_n_z[:,0], pen=(0,255,0)) 
        self.login_widget.plot9.plot(phi_n_z[:,1], pen=(255,255,255)) 
        # QtCore.QCoreApplication.processEvents()   
    
    
    
    def pushButtonClicked_file_open(self):
        
        global DATA,GDATA,data,fname_g, Scan_data,IN_Info,file_dir,Onlyfilename
        
        fname = QtGui.QFileDialog.getOpenFileName(self)
        ffff=fname[0]
        Onlyfilename=ffff.split('/')[-1].split('-')[1]
        kkkk=ffff.split('/')[-1].split('-')[2]
        
        Onlyfilename=Onlyfilename+'-'+kkkk
        projectname=Onlyfilename.split('_')[0]
        
        self.login_widget.label_file_name.setText(ffff.split('/')[-1])
        
        
        fname_g=ffff.split('/')
        fname_g.pop(-1)
        file_dir=''
        for ff in fname_g:
            file_dir=file_dir+ff+'/'
        print(file_dir)
        data= pd.read_csv(fname[0], engine='python')
       
        
        Scan_data=pd.read_csv(file_dir+'Polelist_scan-'+Onlyfilename, engine='python')
        
        IN_fo_dir=file_dir.split('/')
        IN_fo_dir.pop(-1)
        IN_fo_dir[-1]='input'
        IN_fo_file_dir=''
        for ff in IN_fo_dir:
            IN_fo_file_dir=IN_fo_file_dir+ff+'/'
        print(IN_fo_file_dir)
        
        with open(IN_fo_file_dir+projectname+'_pole_auto_conf.json') as f:
          data1 = f.read()
        input_conf = json.loads(data1)
       
        IN_Info = pd.json_normalize(input_conf )
        

        DATA=data
        GDATA=data
        data=data.loc[data['prediction']=='NG'].reset_index()
        
        data=data.sort_values(by=['poleid'], axis=0,ascending=False)
        data=data.reset_index()
        

        
        #for item in data['전산화번호']:
        self.login_widget.ComboBox_D_ID.clear()
        self.login_widget.label_Analysis_pole_total_num.setText(str(len(data['poleid'])))
        self.login_widget.ComboBox_D_ID.addItems(data['poleid']) 
        self.login_widget.ComboBox_D_ID.setCurrentIndex(1)
        
        break_pole=data['poleid']
        break_pole=set(break_pole)
        break_pole=list(break_pole)
        total_pole_num=DATA['poleid']
        total_pole_num=set(total_pole_num)
        total_pole_num=list(total_pole_num)
        
        break_pole_list='# 총 분석 전 주 개수 : '+ str(len(total_pole_num))+'\n'+'# 파단 예측 전주: \n'
        for item in break_pole:
            break_pole_list=break_pole_list+item+'\n'
        break_pole_list=break_pole_list+'....'+str(len(break_pole))+'...'+str(round(len(break_pole)/len(total_pole_num)*100,1)) 
        self.login_widget.label_current_pole.setText(break_pole_list)

class LoginWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(LoginWidget, self).__init__(parent)
        
        self.setWindowTitle('Smart c&S...Vibration serviveability assessment S/W v0.1')

        layoutV1 = QtGui.QVBoxLayout()
        
       
        
        #self.checkbox_file_open = QtGui.QCheckBox('open Meta file')
        self.button_file_open = QtGui.QPushButton('File Open')
        self.label_file_name =  QtGui.QLabel('--file name--')
        
        self.label_current_pole =  QtGui.QLabel('')
        
        
        
        self.label_current_stype =  QtGui.QLabel('후크령  0/1')
        self.label_current_CH =  QtGui.QLabel('CH : 1')
        
        
        self.label_location_xy =  QtGui.QLabel('조사위치: ')
        
        self.label_MAC =  QtGui.QLabel('MAV: 0/0/0')
        self.label_corrcof =  QtGui.QLabel('corrcof: 0')
        
        
        
        self.label_reuslt =  QtGui.QLabel('판단 : ')
        
        self.label_Analysis_pole_total_num =  QtGui.QLabel('')
        
        

        self.label_D_ID =  QtGui.QLabel('pole ID :')
        self.ComboBox_D_ID=QtGui.QComboBox()
        
        self.label_detect_pole =  QtGui.QLabel(' 파단 추정 전주 :')
        self.ComboBox_detect_pole=QtGui.QComboBox()
        
        
        
        self.label_limit_x =  QtGui.QLabel('선택 신호(type/num/CH')
        self.lineEdit_limit_x=  QtGui.QLineEdit('')
        
        self.label_limit_y =  QtGui.QLabel('선택 신호(길이/속도)')
        self.lineEdit_limit_y=  QtGui.QLineEdit('')
        
        self.label_limit_z =  QtGui.QLabel('선택 신호(PtoP)')
        self.lineEdit_limit_z=  QtGui.QLineEdit('')
        
        
        self.label_limit_corrcoef =  QtGui.QLabel('MAC 값(x,y,z))')
        self.lineEdit_limit_corrcoef=  QtGui.QLineEdit('')
        
        
        self.label_limit_MAC_x =  QtGui.QLabel('correlation coef.(x,y,z)')
        self.lineEdit_limit_MAC_x=  QtGui.QLineEdit('')
        
        
        self.label_limit_MAC_y =  QtGui.QLabel('Total')
        self.lineEdit_limit_MAC_y=  QtGui.QLineEdit('')
        
        
        self.label_limit_MAC_z =  QtGui.QLabel('시물레이션 조건(win/S/X)')
        self.lineEdit_limit_MAC_z=  QtGui.QLineEdit('')
        
        
        
        
                                        
        self.button_Analysis = QtGui.QPushButton('>>')
        self.button_Analysis_back = QtGui.QPushButton('<<')
        self.button_Analysis_all = QtGui.QPushButton('해당 신호 삭제')
        
        
        self.button_realtime_stop = QtGui.QPushButton('Analysis(realtime) stop')
        
        self.button_plot_clear = QtGui.QPushButton('최종 결과 파일 생성')
        
        self.button_file_upload = QtGui.QPushButton('결과 파일 탑재')
        self.label_image = QtGui.QLabel()
        
        
        layoutV1.addSpacing(40)
        #layoutV1.addWidget(self.checkbox_file_open)
        layoutV1.addWidget(self.button_file_open)
        layoutV1.addWidget(self.label_file_name)
        layoutV1.addSpacing(30)
        
        layoutV1.addWidget(self.label_Analysis_pole_total_num)
        layoutV1.addWidget(self.label_D_ID)
        layoutV1.addWidget(self.ComboBox_D_ID)
        layoutV1.addSpacing(10)
                
        layoutV1.addWidget(self.label_limit_x)
        layoutV1.addWidget(self.lineEdit_limit_x)
        layoutV1.addSpacing(10)
        
        layoutV1.addWidget(self.label_limit_y)
        layoutV1.addWidget(self.lineEdit_limit_y)
        layoutV1.addSpacing(10)
        
        layoutV1.addWidget(self.label_limit_z)
        layoutV1.addWidget(self.lineEdit_limit_z)
        layoutV1.addSpacing(10)
        
        layoutV1.addWidget(self.label_limit_MAC_z)
        layoutV1.addWidget(self.lineEdit_limit_MAC_z)
        layoutV1.addSpacing(10)
        
        
        layoutV1.addWidget(self.label_limit_corrcoef)
        layoutV1.addWidget(self.lineEdit_limit_corrcoef)
        layoutV1.addSpacing(10)
        
        layoutV1.addWidget(self.label_limit_MAC_x)
        layoutV1.addWidget(self.lineEdit_limit_MAC_x)
        
        layoutV1.addWidget(self.label_limit_MAC_y)
        layoutV1.addWidget(self.lineEdit_limit_MAC_y)
        layoutV1.addSpacing(20)
        
        
        
        
        
        layoutV1.addWidget(self.button_Analysis)
        layoutV1.addSpacing(20)
        layoutV1.addWidget(self.button_Analysis_back)
        layoutV1.addSpacing(30)
        layoutV1.addWidget(self.button_Analysis_all)
        layoutV1.addSpacing(30)   
        layoutV1.addWidget(self.label_current_pole) 
        
    
        layoutV1.addSpacing(30)
        layoutV1.addWidget(self.button_plot_clear)
        layoutV1.addSpacing(30)
        layoutV1.addWidget(self.button_file_upload)
        
                
        '''
        layoutV1.addWidget(self.label_current_pole)
        layoutV1.addWidget(self.label_current_stype)
        layoutV1.addWidget(self.label_current_CH)
        layoutV1.addWidget(self.label_location_xy)
        layoutV1.addWidget(self.label_MAC)
        layoutV1.addWidget(self.label_corrcof)
        
        
        layoutV1.addSpacing(10)
        layoutV1.addWidget(self.label_reuslt)
        
        
        layoutV1.addSpacing(10)
        layoutV1.addWidget(self.label_detect_pole)
        layoutV1.addWidget(self.ComboBox_detect_pole)
        
        
        
        layoutV1.addSpacing(50)
        layoutV1.addWidget(self.label_image)
        layoutV1.addStretch(10)
         '''
        
        
        layoutH2 = QtGui.QHBoxLayout()
        self.plot1 = pg.PlotWidget(title="측정데이터 set : X 축")
        self.plot2 = pg.PlotWidget(title="측정데이터 set : Y 축")
        self.plot3 = pg.PlotWidget(title="측정데이터 set : Z 축 ")
        layoutH2.addWidget(self.plot1)
        layoutH2.addWidget(self.plot2)
        layoutH2.addWidget(self.plot3)
        
        
        layoutH3 = QtGui.QHBoxLayout()
        self.plot4 = pg.PlotWidget(title="선택 신호 : X 축")
        self.plot5 = pg.PlotWidget(title="선택 신호  : Y 축")
        self.plot6 = pg.PlotWidget(title="선택 신호  : Z 축")
        layoutH3.addWidget(self.plot4)
        layoutH3.addWidget(self.plot5)
        layoutH3.addWidget(self.plot6)
        
        layoutH4 = QtGui.QHBoxLayout()
        self.plot7 = pg.PlotWidget(title=" 파단 추정 검출 신호(기울기 변환) : X 축")
        self.plot8 = pg.PlotWidget(title=" 파단 추정 검출 신호(기울기 변환) : Y 축")
        self.plot9 = pg.PlotWidget(title=" 파단 추정 검출 신호(기울기 변환) : Z 축")
        layoutH4.addWidget(self.plot7)
        layoutH4.addWidget(self.plot8)
        layoutH4.addWidget(self.plot9)        
        
        layoutV2 = QtGui.QVBoxLayout()
        layoutV2.addLayout(layoutH2)
        layoutV2.addLayout(layoutH3)
        layoutV2.addLayout(layoutH4)
        
        
        layout = QtGui.QHBoxLayout()
        layout.addLayout(layoutV1)
        layout.addLayout(layoutV2)
        
        layout.setStretchFactor(layoutV1, 0)
        layout.setStretchFactor(layoutV2, 1)
        self.setLayout(layout)
        
        

if __name__ == '__main__':
    app = QtGui.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
