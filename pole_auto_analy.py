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
PDB.poledb_init()


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

def Signal_Info_select_data_out(poleid,stype,num,limit_x,limit_y,limit_z):
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
            print(i,index)
            sig_x_ptp.append(abs(sig_x[i].max()-sig_x[i].min()))
            sig_y_ptp.append(abs(sig_y[i].max()-sig_y[i].min()))
            sig_z_ptp.append(abs(sig_z[i].max()-sig_z[i].min()))
            
        print(sig_x_ptp,sig_y_ptp,sig_z_ptp) 
        
        for index,i in enumerate(index_ch):
            
            sig_x[i]=detrend(sig_x[i])
            sig_x[i]=butter_lowpass_filter(sig_x[i],1, 10)
            sig_x[i]=np.gradient(sig_x[i],1)
            sig_x[i]=rollavg_bottlneck(sig_x[i],10)
            
            sig_y[i]=detrend(sig_y[i])
            sig_y[i]=butter_lowpass_filter(sig_y[i],1, 10)
            sig_y[i]=np.gradient(sig_y[i],1)
            sig_y[i]=rollavg_bottlneck(sig_y[i],10)
            
            sig_z[i]=detrend(sig_z[i])
            sig_z[i]=butter_lowpass_filter(sig_z[i],1, 10)
            sig_z[i]=np.gradient(sig_z[i],1)
            sig_z[i]=rollavg_bottlneck(sig_z[i],10)
            
            
            if sig_x_ptp[index]>=limit_x and sig_y_ptp[index]>=limit_y and sig_z_ptp[index]>=limit_z: 
                select_x_data.append(sig_x[i])
                select_y_data.append(sig_y[i])
                select_z_data.append(sig_z[i])
                select_ch.append(index+1) 
            
    return select_x_data,select_y_data,select_z_data,select_ch


def Signal_Info_select_data_in(poleid,stype,num,limit_x):
    sig_x=PDB.get_meas_data(poleid,num,stype,'x')
    index_ch=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
    
    sig_x_ptp=[]
    select_x_data=[]
    
    select_ch=[]

    
    for index,i in enumerate(index_ch):
        print(i,index)
        sig_x_ptp.append(abs(sig_x[i].max()-sig_x[i].min()))
        
    print(sig_x_ptp) 
    
    for index,i in enumerate(index_ch):
        sig_x[i]=detrend(sig_x[i])
        sig_x[i]=butter_lowpass_filter(sig_x[i],1, 10)
        sig_x[i]=np.gradient(sig_x[i],1)
        sig_x[i]=rollavg_bottlneck(sig_x[i],10)
        
        
        
        if sig_x_ptp[index]>=limit_x : 
            select_x_data.append(sig_x[i])
            select_ch.append(index+1) 
        
    return select_x_data,select_x_data,select_x_data,select_ch










###############################################

Main_mag_s=100
HallSensor_h=2


Gap_size=50
#Gap_location=70   #inverse 75~25




#global Signal_Analy_result 

#select_x_data,select_y_data,select_z_data,select_ch=Signal_Info_select_data(poleid,stype,num,limit_x,limit_y,limit_z)




#Signal_Analy_result=Signal_Analy(select_x_data,select_y_data,select_z_data,select_ch,Main_mag_s,HallSensor_h,scan_len,Gap_size)
    




#print(sig_x_ptp,sig_y_ptp,sig_z_ptp)




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
        
        self.login_widget.button_Analysis_all.clicked.connect(self.pushButtonClicked_Analysis_all)
        
        self.login_widget.button_plot_clear.clicked.connect(self.pushButtonClicked_plot_clear)
        
        self.central_widget.addWidget(self.login_widget)
        
       
        
    def pushButtonClicked_plot_clear(self):
        self.login_widget.plot1.clear() 
        self.login_widget.plot2.clear() 
        self.login_widget.plot3.clear() 
        self.login_widget.plot4.clear() 
        self.login_widget.plot5.clear() 
        self.login_widget.plot6.clear() 
        self.login_widget.plot7.clear() 
        self.login_widget.plot8.clear() 
        self.login_widget.plot9.clear() 
        self.login_widget.ComboBox_detect_pole.clear()
        
    def pushButtonClicked_Analysis_all(self): 
        
        result= pd.DataFrame(columns=['poleid','s_num','stype','X_location','CH','S_position','mac_x_v','mac_y_v','mac_z_v','corrcoef',
                                                "S_x","S_y","S_z"])

        self.login_widget.ComboBox_detect_pole.clear() 
                
        for index,poleid in enumerate(data['전산화번호']):
            
            self.login_widget.label_current_pole.setText('현재 전주: '+ poleid+'..'+str(index+1))
            
            print(poleid,index)
            self.login_widget.ComboBox_D_ID.setCurrentIndex(int(index))
            result.append(self.pushButtonClicked_Analysis())
            
            print('###############################################################')
            print(result)
        result.to_csv('result.csv')    
            #RESULT_RawData_SAVE.to_csv('KICT_RF/result_data/'+fname_g+'_RawData_RF_result.csv')
            
            
            
            
    def pushButtonClicked_Analysis(self):
        
       
        
        global poleid, limit_MAC_X,limit_MAC_Y,limit_MAC_Z,limit_corrcoef,limit_x,limit_y,limit_z
        
        detect_num=0
        
        poleid=self.login_widget.ComboBox_D_ID.currentText()
        
        
   
        limit_MAC_X=float(self.login_widget.lineEdit_limit_MAC_x.text())
        limit_MAC_Y=float(self.login_widget.lineEdit_limit_MAC_y.text())
        limit_MAC_Z=float(self.login_widget.lineEdit_limit_MAC_z.text())
        
        limit_corrcoef=float(self.login_widget.lineEdit_limit_corrcoef.text())
        
        
        limit_x=float(self.login_widget.lineEdit_limit_x.text())
        limit_y=float(self.login_widget.lineEdit_limit_y.text())
        limit_z=float(self.login_widget.lineEdit_limit_z.text())
        
        
        num_sig_out,num_sig_in,re_out,re_in=pole_Info(poleid)
        

        
        Signal_Analy_result= pd.DataFrame(columns=['poleid','s_num','stype','CH','simu_rate','S_position','X_location','mac_x_v','mac_y_v','mac_z_v','corrcoef',
                                                "S_x","S_y","S_z"])



        for kk in range(num_sig_out+num_sig_in):
                if kk <num_sig_out:
                    stype='OUT'
                    num=kk+1
                    select_x_data,select_y_data,select_z_data,select_ch=Signal_Info_select_data_out(poleid,stype,num,limit_x,limit_y,limit_z)
                    scan_len=(re_out['edheight'][num-1]-re_out['stheight'][num-1])*1000 
                    
                    self.login_widget.label_current_stype.setText('후크형: '+str(num)+'/'+str(num_sig_out))
                #else:
                #    stype='IN'
                #    num=(kk-num_sig_out)+1
                #    select_x_data,select_y_data,select_z_data,select_ch=Signal_Info_select_data_in(poleid,stype,num,limit_x)
                #    scan_len=(re_in['edheight'][num-1]-re_in['stheight'][num-1])*1000
            
            
                self.login_widget.plot1.clear()   
                self.login_widget.plot2.clear()   
                self.login_widget.plot3.clear()
                for i,index in enumerate(select_ch):
                    self.login_widget.plot1.plot(select_x_data[i], pen=(100,20*i,20*i)) 
                    self.login_widget.plot2.plot(select_y_data[i], pen=(20*i,20*i,100)) 
                    self.login_widget.plot3.plot(select_z_data[i], pen=(20*i,100,20*i)) 
        
               
                for i ,index in enumerate(select_ch):
                    
                    self.login_widget.label_current_CH.setText('CH : '+str(index))
                    
                    
                    self.login_widget.plot1.plot(select_x_data[i], pen=(255,0,0)) 
                    self.login_widget.plot2.plot(select_y_data[i], pen=(0,0,255))
                    self.login_widget.plot3.plot(select_z_data[i], pen=(0,255,0))
                    
                    for tt in range(20,100,20):  #시물레이션 신호 폭 변경
                        tt=tt/100
                        for k in range(25,75,5):   # 파단 위치
                           Bs=mag_m.pole_simulation(Main_mag_s,Gap_size,k,len(select_x_data[0]),scan_len*tt,HallSensor_h)
                  
                    
                           
                           for j in range(20,80,10):  # 센서 위치
                                 
                                self.login_widget.label_location_xy.setText('손상 : '+str(round(scan_len*((100-k)/100),3))+' 센서 :'+str(j))
    
            
            
                                #data_x = np.array([[l,w] for l,w in zip(np.gradient(Bs[:,j,0],1),select_x_data[i])])
                                #data_y = np.array([[l,w] for l,w in zip(np.gradient(Bs[:,j,1],1),select_y_data[i])])
                                #data_z = np.array([[l,w] for l,w in zip(np.gradient(Bs[:,j,2],1),select_z_data[i])])
                                
                                data_x = np.array([[l,w] for l,w in zip((Bs[:,j,0]),select_x_data[i])])
                                data_y = np.array([[l,w] for l,w in zip((Bs[:,j,1]),select_y_data[i])])
                                data_z = np.array([[l,w] for l,w in zip((Bs[:,j,2]),select_z_data[i])])
                                
                                phi_n_x, mode_scaling=normalize_phi(data_x)
                                mac_x_v=mac(phi_n_x[:,0],phi_n_x[:,1])
                                
                                phi_n_y, mode_scaling=normalize_phi(data_y)
                                mac_y_v=mac(phi_n_y[:,0],phi_n_y[:,1])
                                
                                phi_n_z, mode_scaling=normalize_phi(data_z)
                                mac_z_v=mac(phi_n_z[:,0],phi_n_z[:,1])
                                
                                
                                
                                corr_x = signal.correlate(phi_n_x[:,0],phi_n_x[:,1])
                                corr_x /= np.max(corr_x)
                                corrcoef=np.corrcoef(phi_n_x[:,0],phi_n_x[:,1])[0, 1]
                                
                                
                                new_data={'poleid':poleid,'s_num':num,'stype':stype, 'simu_rate':tt,'X_location':k,'CH':select_ch[i],'S_position':j,'mac_x_v':round(mac_x_v,3),'mac_y_v':round(mac_y_v,3),'mac_z_v':round(mac_z_v,3),'corrcoef':round(corrcoef,3)}
                                          #"S_x":phi_n_x[:,0],"S_y":phi_n_y[:,0],"S_z":phi_n_z[:,0]}
                                
                                                             
                                self.login_widget.label_MAC.setText('MAC : '+str(round(mac_x_v,3))+'/'+str(round(mac_y_v,3))+'/'+str(round(mac_z_v,3)))
                                self.login_widget.label_corrcof.setText('corrcoef :'+str(round(corrcoef,3)))
                                
                               
                                if mac_x_v > limit_MAC_X and mac_y_v > limit_MAC_Y and mac_z_v > limit_MAC_Z and abs(corrcoef)>limit_corrcoef:   
                                   Signal_Analy_result=Signal_Analy_result.append(new_data,ignore_index=True)
                                   self.login_widget.label_reuslt.setText('추정결과 : 파단')
                                   self.login_widget.plot7.plot(phi_n_x[:,1], pen=(255,0,0)) 
                                   self.login_widget.plot8.plot(phi_n_y[:,1], pen=(0,0,255))
                                   self.login_widget.plot9.plot(phi_n_z[:,1], pen=(0,255,0)) 
                                   
                                   detect_num=detect_num+1
                                
                                else:
                                   
                                   self.login_widget.label_reuslt.setText('추정결과 : 정상')
                                #else:
                                #   print('NG ',round(mac_x_v,3),round(mac_y_v,3),round(mac_z_v,3) ,round(corrcoef,3)  )
                                #print(new_data) 
                                self.login_widget.plot4.clear()   
                                self.login_widget.plot5.clear()   
                                self.login_widget.plot6.clear()
                                self.login_widget.plot4.plot(phi_n_x[:,0], pen=(255,0,0)) 
                                self.login_widget.plot4.plot(phi_n_x[:,1], pen=(255,255,255))
                                self.login_widget.plot5.plot(phi_n_y[:,0], pen=(0,0,255))
                                self.login_widget.plot5.plot(phi_n_y[:,1], pen=(255,255,255))
                                self.login_widget.plot6.plot(phi_n_z[:,0], pen=(0,255,0)) 
                                self.login_widget.plot6.plot(phi_n_z[:,1], pen=(255,255,255)) 
                                QtCore.QCoreApplication.processEvents()
        if detect_num>0:
            Signal_Analy_result.to_csv(poleid+'.csv')
            print(Signal_Analy_result)
            self.login_widget.ComboBox_detect_pole.addItems([poleid+'_'+str(detect_num)]) 
            
        return Signal_Analy_result                    




       
    def pushButtonClicked_file_open(self):
        
        global data,fname_g,AE_NMAE,AE_INDEX,Bridge_information
        
        fname = QtGui.QFileDialog.getOpenFileName(self)
        ffff=fname[0]
        self.login_widget.label_file_name.setText(ffff.split('/')[-1])
        fname_g=ffff.split('/')[-1]
        fname_g=fname_g.split('.')[0]
        data= pd.read_csv(fname[0], engine='python')
        
        print(data['전산화번호'])
        #for item in data['전산화번호']:
        self.login_widget.ComboBox_D_ID.clear()
        self.login_widget.label_Analysis_pole_total_num.setText(str(len(data['전산화번호'])))
        self.login_widget.ComboBox_D_ID.addItems(data['전산화번호']) 
        self.login_widget.ComboBox_D_ID.setCurrentIndex(1)
        
        
        
        
       

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
        
        
        
        self.label_limit_x =  QtGui.QLabel('Limit(x)')
        self.lineEdit_limit_x=  QtGui.QLineEdit('1000')
        
        self.label_limit_y =  QtGui.QLabel('Limit(y)')
        self.lineEdit_limit_y=  QtGui.QLineEdit('1000')
        
        self.label_limit_z =  QtGui.QLabel('Limit(z)')
        self.lineEdit_limit_z=  QtGui.QLineEdit('1000')
        
        
        self.label_limit_corrcoef =  QtGui.QLabel('Limit(corrcoef)')
        self.lineEdit_limit_corrcoef=  QtGui.QLineEdit('0.5')
        
        
        self.label_limit_MAC_x =  QtGui.QLabel('MAC(x)')
        self.lineEdit_limit_MAC_x=  QtGui.QLineEdit('0.3')
        
        
        self.label_limit_MAC_y =  QtGui.QLabel('MAC(y)')
        self.lineEdit_limit_MAC_y=  QtGui.QLineEdit('0.3')
        
        
        self.label_limit_MAC_z =  QtGui.QLabel('MAC(z)')
        self.lineEdit_limit_MAC_z=  QtGui.QLineEdit('0.3')
        
        
        
        
                                        
        self.button_Analysis = QtGui.QPushButton('Analysis(file-step)')
        self.button_Analysis_all = QtGui.QPushButton('Analysis(file-ALL)')
        
        self.button_realtime = QtGui.QPushButton('Analysis(realtime) >>')
        self.button_realtime_stop = QtGui.QPushButton('Analysis(realtime) stop')
        
        self.button_plot_clear = QtGui.QPushButton('PLOT CLEAR')
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
        
        layoutV1.addWidget(self.label_limit_MAC_x)
        layoutV1.addWidget(self.lineEdit_limit_MAC_x)
        layoutV1.addSpacing(10)
        
        layoutV1.addWidget(self.label_limit_MAC_y)
        layoutV1.addWidget(self.lineEdit_limit_MAC_y)
        layoutV1.addSpacing(10)
        
        layoutV1.addWidget(self.label_limit_MAC_z)
        layoutV1.addWidget(self.lineEdit_limit_MAC_z)
        layoutV1.addSpacing(10)
        
        layoutV1.addWidget(self.label_limit_corrcoef)
        layoutV1.addWidget(self.lineEdit_limit_corrcoef)
        layoutV1.addSpacing(20)
        
        
        
        layoutV1.addWidget(self.button_Analysis)
        layoutV1.addSpacing(10)
        layoutV1.addWidget(self.button_Analysis_all)
           
    
        layoutV1.addSpacing(10)
        layoutV1.addWidget(self.button_plot_clear)
        layoutV1.addSpacing(10)
        
        
        
        
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

        
        
        layoutH2 = QtGui.QHBoxLayout()
        self.plot1 = pg.PlotWidget(title="측정데이터 : X 축")
        self.plot2 = pg.PlotWidget(title="측정데이터 : Y 축")
        self.plot3 = pg.PlotWidget(title="측정데이터 : Z 축 ")
        layoutH2.addWidget(self.plot1)
        layoutH2.addWidget(self.plot2)
        layoutH2.addWidget(self.plot3)
        
        
        layoutH3 = QtGui.QHBoxLayout()
        self.plot4 = pg.PlotWidget(title="시물레이션 신호 : X 축")
        self.plot5 = pg.PlotWidget(title="시물레이션 신호 : Y 축")
        self.plot6 = pg.PlotWidget(title="시물레이션 신호 : Z 축")
        layoutH3.addWidget(self.plot4)
        layoutH3.addWidget(self.plot5)
        layoutH3.addWidget(self.plot6)
        
        layoutH4 = QtGui.QHBoxLayout()
        self.plot7 = pg.PlotWidget(title=" 파단 추정 검출 신호 : X 축")
        self.plot8 = pg.PlotWidget(title=" 파단 추정 검출 신호 : Y 축")
        self.plot9 = pg.PlotWidget(title=" 파단 추정 검출 신호 : Z 축")
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
        
        
        
        '''
        self.label = pg.LabelItem(justify='right')
        self.label.setText('ssssssfdsjkgjdfksghnkjdf')
        self.plot3.addItem(self.label)
        '''


if __name__ == '__main__':
    app = QtGui.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
