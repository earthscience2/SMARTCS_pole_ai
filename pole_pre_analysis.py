# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 20:41:02 2021

@author: 스마트제어계측
"""

import matplotlib.pyplot as plt
import pandas as pd

Pole_data= pd.read_csv('analysis/2021/AGSENG/Gangmoung/output/polelist_scan.csv', engine='python')

data=Pole_data.loc[(Pole_data['stype']=='OUT') ].reset_index()

plt.hist(data['PtP_x'])
plt.hist(data['PtP_y'])
plt.hist(data['PtP_z'])
plt.show()