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
        

        
        #self.login_widget.button_Analysis_all.clicked.connect(self.pushButtonClicked_Analysis_delete)
        
        self.login_widget.button_plot_clear.clicked.connect(self.pushButtonClicked_maker_file)
        
        self.central_widget.addWidget(self.login_widget)
        
       
        
    def pushButtonClicked_maker_file(self):
        
        
        # 임시로 저장        
        js = IN_Info.to_json(orient = 'records')
        print(js[0])
        #text_file=open(IN_fo_file_dir+projectname+'_pole_auto_conf.json', 'w')
        #text_file.write(js[0])
        #text_file.close()
        '''
        NF_data["first"]=SSI_Freq_result[0]
                NF_data["second"]=SSI_Freq_result[1]    
                NF_data["third"]=SSI_Freq_result[2]    
            
            NF_dir=Rootdir+S_type+'/'+S_name+'/naturalFrequency/raw_NF/'+Adate+'_raw_NF.json'
            with open(NF_dir, 'w', encoding="utf-8") as make_file:
                  json.dump(NF_data, make_file, ensure_ascii=False, indent="\t")
       ''' 
    
    
    
    
        
        
        
            
    def pushButtonClicked_Analysis(self):
        
        global poleid, stype,num,CH

        result_text=''        
        limit_x=int(self.login_widget.lineEdit_limit_x.text())
        limit_y=int(self.login_widget.lineEdit_limit_y.text())
        limit_z=int(self.login_widget.lineEdit_limit_z.text())
        IN_Info['limit_out_x']=limit_x
        IN_Info['limit_out_y']=limit_y
        IN_Info['limit_out_z']=limit_z
        
        Scan_data_OUT=Scan_data.loc[(Scan_data['stype']=='OUT')].reset_index()
        Scan_data_IN=Scan_data.loc[(Scan_data['stype']=='IN')].reset_index()
        
        
        
        Scan_data_OUT_len=len(Scan_data_OUT)
        Scan_data_IN_len=len(Scan_data_IN)
        
        self.login_widget.plot1.clear()   
        self.login_widget.plot2.clear()   
        self.login_widget.plot3.clear()
        self.login_widget.plot4.clear()   
        self.login_widget.plot5.clear()   
        self.login_widget.plot6.clear()
        self.login_widget.plot7.clear()   
        self.login_widget.plot8.clear()   
        self.login_widget.plot9.clear()
                
        self.login_widget.plot1.plot(Scan_data_OUT['PtP_x'] ,pen=(255,0,0),symbol='t', symbolPen=None, symbolSize=10, symbolBrush=('y')) 
        self.login_widget.plot2.plot(Scan_data_OUT['PtP_y'] ,pen=(0,255,0),symbol='t', symbolPen=None, symbolSize=10, symbolBrush=('y')) 
        self.login_widget.plot3.plot(Scan_data_OUT['PtP_z'] ,pen=(0,0,255),symbol='t', symbolPen=None, symbolSize=10, symbolBrush=('y')) 
        
        self.login_widget.plot1.plot([0,Scan_data_OUT_len],[limit_x,limit_x] ,pen=('w')) 
        self.login_widget.plot2.plot([0,Scan_data_OUT_len],[limit_y,limit_y] ,pen=('w')) 
        self.login_widget.plot3.plot([0,Scan_data_OUT_len],[limit_z,limit_z] ,pen=('w')) 
        
        import numpy as np
        
        #gg=Scan_data_OUT.hist(column="PtP_x",bins=10) 
        
        count_x, division_x = np.histogram(Scan_data_OUT['PtP_x'],bins=20)
        count_y, division_y = np.histogram(Scan_data_OUT['PtP_y'],bins=20)
        count_z, division_z = np.histogram(Scan_data_OUT['PtP_z'],bins=20)
        print(count_z, division_z)
        self.login_widget.plot4.plot(division_x[1:],count_x, pen=(255,0,0),symbol='o', symbolSize=10, symbolBrush=('w'))
        self.login_widget.plot5.plot(division_y[1:],count_y, pen=(0,255,0),symbol='o', symbolSize=10, symbolBrush=('w')) 
        self.login_widget.plot6.plot(division_z[1:],count_z, pen=(0,0,255),symbol='o', symbolSize=10, symbolBrush=('w') )
        
        
        self.login_widget.plot4.plot([limit_x,limit_x],[0,max(count_x)],pen=('w')) 
        self.login_widget.plot5.plot([limit_y,limit_y],[0,max(count_y)],pen=('w')) 
        self.login_widget.plot6.plot([limit_z,limit_z],[0,max(count_z)],pen=('w')) 
        
        
        select_x=Scan_data_OUT['PtP_x']
        select_y=Scan_data_OUT['PtP_y']
        select_z=Scan_data_OUT['PtP_z']
        
        
        
        for index,val in enumerate(select_x):
            if val<limit_x:
                select_x[index]=0
        for index,val in enumerate(select_y):
            if val<limit_y:
                select_y[index]=0
        for index,val in enumerate(select_z):
            if val<limit_z:
                select_z[index]=0        
        
        self.login_widget.plot7.plot(select_x ,pen=(255,0,0),symbol='t', symbolPen=None, symbolSize=10, symbolBrush=('y')) 
        self.login_widget.plot7.plot(select_y ,pen=(0,255,0),symbol='t', symbolPen=None, symbolSize=10, symbolBrush=('y')) 
        self.login_widget.plot7.plot(select_z ,pen=(0,0,255),symbol='t', symbolPen=None, symbolSize=10, symbolBrush=('y')) 
        
        
        select_signal=[]
        select_signal_num=0
        for index in range(len(select_x)):
           if select_x[index]>0 and select_y[index]>0 and select_x[index]>0: 
              select_signal.append((select_x[index]+select_y[index]+select_z[index])/3)         
              select_signal_num+=1
           else:
              select_signal.append(0)
           self.login_widget.plot8.plot([index,index],[0,select_signal[index]],pen=('w'))     
        
        print('#### total signal number :',len(select_x)) 
        print('#### Select signal number :',(select_signal_num)) 
        
        result_text='#### total signal number :'+str(len(select_x))+'\n'
        result_text= result_text+'#### Select signal number :'+str(select_signal_num)+'\n'
        result_text= result_text+'#### Select ratio :'+str(round((select_signal_num/len(select_x))*100,2))+'\n'
        
        
        Scan_data_OUT['sel_val']=select_signal
        
        
        pole_list=Scan_data_OUT['poleid']
        pole_list=list(set(pole_list))
        
        print(pole_list)
        pole_seclect_list=[]
        pole_seclect_val=[]
        
        pole_select_index=0
        for poleid in pole_list:
            temp_pd=Scan_data_OUT.loc[(Scan_data_OUT['poleid']==poleid) & (Scan_data_OUT['sel_val']!=0) ].reset_index()
            
            pole_seclect_val.append(len(temp_pd)) 
            if len(temp_pd)>0:
               pole_seclect_list.append(poleid)
            self.login_widget.plot9.plot([pole_select_index,pole_select_index],[0,pole_seclect_val[pole_select_index]],pen=('w'))     
            pole_select_index+=1
        print('#### total pole number :',len(pole_list)) 
        print('#### Select pole number :',len(pole_seclect_list)) 
        
        
        result_text=result_text+'\n'
        result_text=result_text+'#### total pole number  :'+str(len(pole_list))+'\n'
        result_text= result_text+'#### Select signal number :'+str(len(pole_seclect_list))+'\n'
        result_text= result_text+'#### Select ratio :'+str(round((len(pole_seclect_list)/len(pole_list))*100,2))+'\n'
        
        
        
        
        print(pole_seclect_list)
        
        
        self.login_widget.label_reuslt.setText(result_text)
        
        
        
        
          
    def pushButtonClicked_file_open(self):
        
        global DATA,GDATA,data,fname_g, Scan_data,IN_Info,file_dir,Onlyfilename,IN_fo_file_dir,projectname
        
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
        Scan_data= pd.read_csv(fname[0], engine='python')
       
        
        #Scan_data=pd.read_csv(file_dir+'Polelist_scan-'+Onlyfilename, engine='python')
        
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
        

        
        
        self.login_widget.lineEdit_limit_x.setText(str(IN_Info['limit_out_x'][0]))
        self.login_widget.lineEdit_limit_y.setText(str(IN_Info['limit_out_y'][0]))
        self.login_widget.lineEdit_limit_z.setText(str(IN_Info['limit_out_z'][0]))
        
        print(Scan_data)
        
        DATA=Scan_data.loc[Scan_data['breakstate']=='B'].reset_index().drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
       
        
        #for item in data['전산화번호']:
        self.login_widget.ComboBox_D_ID.clear()
        #self.login_widget.label_Analysis_pole_total_num.setText(str(len(data['poleid'])))
        self.login_widget.ComboBox_D_ID.addItems(DATA['poleid']) 
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
        
        

        self.label_D_ID =  QtGui.QLabel('현장 파단 판정 pole ID :')
        self.ComboBox_D_ID=QtGui.QComboBox()
        
        self.label_detect_pole =  QtGui.QLabel(' 파단 추정 전주 :')
        self.ComboBox_detect_pole=QtGui.QComboBox()
        
        
        
        self.label_limit_x =  QtGui.QLabel('Limit_value_X')
        self.lineEdit_limit_x=  QtGui.QLineEdit('')
        
        self.label_limit_y =  QtGui.QLabel('Limit_value_Y')
        self.lineEdit_limit_y=  QtGui.QLineEdit('')
        
        self.label_limit_z =  QtGui.QLabel('Limit_value_Z')
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
        
        
        layoutV1.addSpacing(20)
        
        
        
        
        
        layoutV1.addWidget(self.button_Analysis)
        layoutV1.addSpacing(20)
      
        layoutV1.addWidget(self.button_Analysis_all)
           
        
        layoutV1.addSpacing(30)
        layoutV1.addWidget(self.button_plot_clear)
        layoutV1.addSpacing(30)
        layoutV1.addWidget(self.label_reuslt)
        
        
        
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
        self.plot1 = pg.PlotWidget(title="P to P: X 축")
        self.plot2 = pg.PlotWidget(title="P to P : Y 축")
        self.plot3 = pg.PlotWidget(title="P to P : Z 축 ")
        layoutH2.addWidget(self.plot1)
        layoutH2.addWidget(self.plot2)
        layoutH2.addWidget(self.plot3)
        
        
        layoutH3 = QtGui.QHBoxLayout()
        self.plot4 = pg.PlotWidget(title="Histogram : X 축")
        self.plot5 = pg.PlotWidget(title="Histogram : Y 축")
        self.plot6 = pg.PlotWidget(title="Histogram : Z 축")
        layoutH3.addWidget(self.plot4)
        layoutH3.addWidget(self.plot5)
        layoutH3.addWidget(self.plot6)
        
        layoutH4 = QtGui.QHBoxLayout()
        self.plot7 = pg.PlotWidget(title=" 선택 P to P : XYZ 축")
        self.plot8 = pg.PlotWidget(title=" 선택된 신호 ")
        self.plot9 = pg.PlotWidget(title=" 선택된 전주")
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
