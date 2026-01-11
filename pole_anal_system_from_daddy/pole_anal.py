from scipy.signal import detrend
import pandas as pd
import numpy as np
import logger
from config import poledb
import magpylib as mag3
from magpylib.magnet import Cylinder
from magpylib import Collection

logger = logger.get_logger()

def get_pole_anal_result_in(poleid, measno, in_hall_height, in_insert_len, in_st_angle, thr_val=None):
    logger.info('get_pole_anal_result_in) poleid={} measno={} in_hall_height={} in_insert_len={} in_st_angle={} thr_val={}'.format(poleid, measno, in_hall_height, in_insert_len, in_st_angle, thr_val))

    data = {}

    pole_data_x = poledb.get_meas_data(poleid, measno, 'IN', 'x')
    pole_data_y = poledb.get_meas_data(poleid, measno, 'IN', 'y')
    pole_data_z = poledb.get_meas_data(poleid, measno, 'IN', 'z')

    # Data_type='OUT'
    DATA_NUM_IN = 0
    pole_code=poleid
    Pole_Thread_Value_PtoP_IN=0
    if thr_val is not None:
        Pole_Thread_Value_PtoP_IN = thr_val

    # OUT_Sensor_Interval_angle=in_st_angle
    OUT_Sensor_Num=8
    Start_point_Angle=in_st_angle
    Max_Angle = 360
    Angle_Per_ch = int(Max_Angle / OUT_Sensor_Num)

    St_Heigth=in_hall_height-in_insert_len
    # Ed_Heigth=out_ed_height
    Ed_Heigth=1500
    Pole_IN_Position_list=[]
    # Pole_IN_Position_list=[0,45,90,135,180,225,270,315]
    for i in range(OUT_Sensor_Num):
        Pole_IN_Position_list.append(i*Angle_Per_ch)
    print(Pole_IN_Position_list)

    Pole_IN_DataSet_list=[]
    Pole_IN_DataSet_Name=[]
    data_axis = []
    ch_name_list = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']

    if pole_data_x is None or len(pole_data_x) == 0:
        logger.info('X-axis not found')
    else:
        DATA_NUM_IN = DATA_NUM_IN + 1
        Pole_IN_DataSet_Name.append(pole_code+'_IN_1')
        Pole_IN_DataSet_list.append(pole_data_x[ch_name_list])
        data_axis.append('x')
        logger.info(len(pole_data_x))

    if pole_data_y is None or len(pole_data_y) == 0:
        logger.info('Y-axis not found')
    else:
        DATA_NUM_IN = DATA_NUM_IN + 1
        Pole_IN_DataSet_Name.append(pole_code+'_IN_2')
        Pole_IN_DataSet_list.append(pole_data_y[ch_name_list])
        # logger.info(len(pole_data_y))
        data_axis.append('y')

    if pole_data_z is None or len(pole_data_z) == 0:
        logger.info('Z-axis not found')
    else:
        DATA_NUM_IN = DATA_NUM_IN + 1
        Pole_IN_DataSet_Name.append(pole_code+'_IN_3')
        Pole_IN_DataSet_list.append(pole_data_z[ch_name_list])
        # logger.info(len(pole_data_z))
        data_axis.append('z')

    for i,value in enumerate(Pole_IN_Position_list):
        Pole_IN_Position_list[i]= (value+Start_point_Angle)
        if (value+Start_point_Angle)>= 360:
            Pole_IN_Position_list[i]= Pole_IN_Position_list[i]-360

    # DATA SET column name 변경(코드_IN_01_s1_30)
    for i in range(DATA_NUM_IN):
        kk=Pole_IN_DataSet_list[i].shape
        col_name=pole_code+'_IN'+'_'+str(i+1)
        for j,item in enumerate(Pole_IN_DataSet_list[i].columns):
            Pole_IN_DataSet_list[i].rename(columns = {item :col_name+'_'+item+'_'+str(Pole_IN_Position_list[j])}, inplace = True)
        Pole_IN_DataSet_list[i][col_name+'_Len']=np.linspace((Ed_Heigth-St_Heigth),St_Heigth,kk[0]).tolist()

    #임계치 검토..column index에 추가
    for i in range(DATA_NUM_IN):
        for j,item in enumerate(Pole_IN_DataSet_list[i].columns):
            gg=item.split('_')
            if len(gg)==5:
                ptp=abs(Pole_IN_DataSet_list[i][item].max()-Pole_IN_DataSet_list[i][item].min())   
                if ptp > Pole_Thread_Value_PtoP_IN:
                    Pole_IN_DataSet_list[i].rename(columns = {item :item+'_1'}, inplace = True)
                else:
                    Pole_IN_DataSet_list[i].rename(columns = {item :item+'_0'}, inplace = True)

    # detrend, gradient,GdotD 데이터 추가
    for i in range(DATA_NUM_IN):
        for j,item in enumerate(Pole_IN_DataSet_list[i].columns):
            if item.endswith('_0'):
                continue
            print(item)
            gg=item.split('_')
            if len(gg)==6:
                Pole_IN_DataSet_list[i][item]=Pole_IN_DataSet_list[i][item]-Pole_IN_DataSet_list[i][item][0]
                Pole_IN_DataSet_list[i][item+'_detrend'] = detrend(Pole_IN_DataSet_list[i][item])
                Pole_IN_DataSet_list[i][item+'_detrend'] = (Pole_IN_DataSet_list[i][item+'_detrend'])-(Pole_IN_DataSet_list[i][item+'_detrend'][0])
                Pole_IN_DataSet_list[i][item+'_gradient'] = np.gradient(Pole_IN_DataSet_list[i][item],1)
                Pole_IN_DataSet_list[i][item+'_GdotD'] = Pole_IN_DataSet_list[i][item+'_gradient']*Pole_IN_DataSet_list[i][item+'_detrend']
    
    # print(Pole_IN_DataSet_list[1].columns.tolist())
    # print(Pole_IN_DataSet_list[1])

    for i in range(DATA_NUM_IN):
        axis = data_axis[i]

        anal_result_df = Pole_IN_DataSet_list[i]

        col_list = Pole_IN_DataSet_list[i].columns.tolist()

        height_col_name = ''
        for j in col_list:
            if j.endswith('_Len'):
                height_col_name = j
                break

        height_list = anal_result_df[height_col_name].tolist()
        height_list = [ round(j,2) for j in height_list ]
        angle_list = Pole_IN_Position_list

        # print(height_list)
        # print(angle_list)

        raw = []
        _detrend = []
        gradient = []
        gdotd = []
    
        for j in col_list:
            if j.endswith('_0'):
                continue

            if j.endswith('detrend'):
                _detrend.append(anal_result_df[j].tolist())
            elif j.endswith('gradient'):
                gradient.append(anal_result_df[j].tolist())
            elif j.endswith('GdotD'):
                gdotd.append(anal_result_df[j].tolist())
            elif j.endswith('Len'):
                pass
            else:
                raw.append(anal_result_df[j].tolist())

        d = {}
        d['height'] = height_list
        d['angle'] = angle_list
        d['raw'] = raw
        d['detrend'] = _detrend
        d['gdotd'] = gdotd
        d['gradient'] = gradient

        data[axis] = d

    return data

def get_pole_anal_result_out(poleid, measno, out_st_height, out_ed_height, out_st_angle, thr_val=None):
    logger.info('get_pole_anal_result_out) poleid={} measno={} out_st_height={} out_ed_height={} out_st_angle={} thr_val={}'.format(poleid, measno, out_st_height, out_ed_height, out_st_angle, thr_val))

    data = {}

    pole_data_x = poledb.get_meas_data(poleid, measno, 'OUT', 'x')
    pole_data_y = poledb.get_meas_data(poleid, measno, 'OUT', 'y')
    pole_data_z = poledb.get_meas_data(poleid, measno, 'OUT', 'z')

    # Data_type='OUT'
    DATA_NUM_IN = 0
    pole_code=poleid
    # Pole_Thread_Value_PtoP_IN=500
    # OUT_Sensor_Interval_angle=out_st_angle
    OUT_Sensor_Num=8
    Start_point_Angle=out_st_angle
    Max_Angle = 80
    Angle_Per_ch = int(Max_Angle / OUT_Sensor_Num)
    Pole_Thread_Value_PtoP_IN=0
    if thr_val is not None:
        Pole_Thread_Value_PtoP_IN = thr_val

    St_Heigth=out_st_height
    # Ed_Heigth=out_ed_height
    Ed_Heigth=1500
    Pole_IN_Position_list=[]
    # Pole_IN_Position_list=[0,45,90,135,180,225,270,315]
    for i in range(OUT_Sensor_Num):
        Pole_IN_Position_list.append(i*Angle_Per_ch)
    print(Pole_IN_Position_list)

    Pole_IN_DataSet_list=[]
    Pole_IN_DataSet_Name=[]
    data_axis = []
    ch_name_list = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']

    if pole_data_x is None or len(pole_data_x) == 0:
        logger.info('X-axis not found')
    else:
        DATA_NUM_IN = DATA_NUM_IN + 1
        Pole_IN_DataSet_Name.append(pole_code+'_IN_1')
        Pole_IN_DataSet_list.append(pole_data_x[ch_name_list])
        data_axis.append('x')
        logger.info(len(pole_data_x))

    if pole_data_y is None or len(pole_data_y) == 0:
        logger.info('Y-axis not found')
    else:
        DATA_NUM_IN = DATA_NUM_IN + 1
        Pole_IN_DataSet_Name.append(pole_code+'_IN_2')
        Pole_IN_DataSet_list.append(pole_data_y[ch_name_list])
        # logger.info(len(pole_data_y))
        data_axis.append('y')

    if pole_data_z is None or len(pole_data_z) == 0:
        logger.info('Z-axis not found')
    else:
        DATA_NUM_IN = DATA_NUM_IN + 1
        Pole_IN_DataSet_Name.append(pole_code+'_IN_3')
        Pole_IN_DataSet_list.append(pole_data_z[ch_name_list])
        # logger.info(len(pole_data_z))
        data_axis.append('z')

    for i,value in enumerate(Pole_IN_Position_list):
        Pole_IN_Position_list[i]= (value+Start_point_Angle)
        if (value+Start_point_Angle)>= 360:
            Pole_IN_Position_list[i]= Pole_IN_Position_list[i]-360

    # DATA SET column name 변경(코드_IN_01_s1_30)
    for i in range(DATA_NUM_IN):
        kk=Pole_IN_DataSet_list[i].shape
        col_name=pole_code+'_IN'+'_'+str(i+1)
        for j,item in enumerate(Pole_IN_DataSet_list[i].columns):
            Pole_IN_DataSet_list[i].rename(columns = {item :col_name+'_'+item+'_'+str(Pole_IN_Position_list[j])}, inplace = True)
        Pole_IN_DataSet_list[i][col_name+'_Len']=np.linspace((Ed_Heigth-St_Heigth),St_Heigth,kk[0]).tolist()

    #임계치 검토..column index에 추가
    for i in range(DATA_NUM_IN):
        for j,item in enumerate(Pole_IN_DataSet_list[i].columns):
            gg=item.split('_')
            if len(gg)==5:
                ptp=abs(Pole_IN_DataSet_list[i][item].max()-Pole_IN_DataSet_list[i][item].min())   
                if ptp > Pole_Thread_Value_PtoP_IN:
                    Pole_IN_DataSet_list[i].rename(columns = {item :item+'_1'}, inplace = True)
                else:
                    Pole_IN_DataSet_list[i].rename(columns = {item :item+'_0'}, inplace = True)

    # detrend, gradient,GdotD 데이터 추가
    for i in range(DATA_NUM_IN):
        for j,item in enumerate(Pole_IN_DataSet_list[i].columns):
            if item.endswith('_0'):
                continue

            gg=item.split('_')
            if len(gg)==6:
                Pole_IN_DataSet_list[i][item]=Pole_IN_DataSet_list[i][item]-Pole_IN_DataSet_list[i][item][0]
                print(item)
                Pole_IN_DataSet_list[i][item+'_detrend'] = detrend(Pole_IN_DataSet_list[i][item])
                Pole_IN_DataSet_list[i][item+'_detrend'] = (Pole_IN_DataSet_list[i][item+'_detrend'])-(Pole_IN_DataSet_list[i][item+'_detrend'][0])
                Pole_IN_DataSet_list[i][item+'_gradient'] = np.gradient(Pole_IN_DataSet_list[i][item],1)
                Pole_IN_DataSet_list[i][item+'_GdotD'] = Pole_IN_DataSet_list[i][item+'_gradient']*Pole_IN_DataSet_list[i][item+'_detrend']
    
    # print(Pole_IN_DataSet_list[1].columns.tolist())
    # print(Pole_IN_DataSet_list[1])

    for i in range(DATA_NUM_IN):
        axis = data_axis[i]

        anal_result_df = Pole_IN_DataSet_list[i]

        col_list = Pole_IN_DataSet_list[i].columns.tolist()

        height_col_name = ''
        for j in col_list:
            if j.endswith('_Len'):
                height_col_name = j
                break

        height_list = anal_result_df[height_col_name].tolist()
        height_list = [ round(j,2) for j in height_list ]
        angle_list = Pole_IN_Position_list

        # print(height_list)
        # print(angle_list)

        raw = []
        _detrend = []
        gradient = []
        gdotd = []
    
        for j in col_list:
            if j.endswith('_0'):
                continue

            if j.endswith('detrend'):
                _detrend.append(anal_result_df[j].tolist())
            elif j.endswith('gradient'):
                gradient.append(anal_result_df[j].tolist())
            elif j.endswith('GdotD'):
                gdotd.append(anal_result_df[j].tolist())
            elif j.endswith('Len'):
                pass
            else:
                raw.append(anal_result_df[j].tolist())

        d = {}
        d['height'] = height_list
        d['angle'] = angle_list
        d['raw'] = raw
        d['detrend'] = _detrend
        d['gdotd'] = gdotd
        d['gradient'] = gradient

        data[axis] = d

    return data

def get_pole_anal_data(pole_code, Data_type, pole_data_x, pole_data_y, pole_data_z, Start_point_Angle, End_point_Angle, Start_point, End_point, Pole_Thread_Value_PtoP_IN):
    Pole_IN_DataSet_list=[]
    Pole_IN_DataSet_Name=[]
    DATA_NUM_IN = 3

    [Pole_IN_DataSet_list.append(pole_code+'_IN'+'_'+str(i+1)) for i in range(DATA_NUM_IN)]
    Pole_IN_DataSet_Name=Pole_IN_DataSet_list

    Pole_IN_DataSet_list[0] = pole_data_x
    Pole_IN_DataSet_list[1] = pole_data_y
    Pole_IN_DataSet_list[2] = pole_data_z

    Pole_IN_Position_list=np.linspace(Start_point_Angle,End_point_Angle,8).tolist()

    for i,value in enumerate(Pole_IN_Position_list):
        Pole_IN_Position_list[i]= (value)
        if (value)>= 360:
            Pole_IN_Position_list[i]= Pole_IN_Position_list[i]-360

    # column index에 각도 추가, 거리 column 추가
    for i in range(DATA_NUM_IN):
        kk=Pole_IN_DataSet_list[i].shape
        col_name=pole_code+'_'+Data_type+'_'+str(i+1)
        for j,item in enumerate(Pole_IN_DataSet_list[i].columns):
            Pole_IN_DataSet_list[i].rename(columns = {item :col_name+'_'+item+'_'+str(Pole_IN_Position_list[j])}, inplace = True)
        Pole_IN_DataSet_list[i][col_name+'_Len']=np.linspace(Start_point,End_point,kk[0]).tolist()        

    #임계치 검토..column index에 추가
    for i in range(DATA_NUM_IN):
        for j,item in enumerate(Pole_IN_DataSet_list[i].columns):
            gg=item.split('_')
            if len(gg)==5:
                ptp=abs(Pole_IN_DataSet_list[i][item].max()-Pole_IN_DataSet_list[i][item].min())   
                if ptp > Pole_Thread_Value_PtoP_IN:
                    Pole_IN_DataSet_list[i].rename(columns = {item :item+'_on'}, inplace = True)
                else:
                    Pole_IN_DataSet_list[i].rename(columns = {item :item+'_off'}, inplace = True)        

    for i in range(DATA_NUM_IN):
        for j,item in enumerate(Pole_IN_DataSet_list[i].columns):
            gg=item.split('_')
            if len(gg)==6:
                  Pole_IN_DataSet_list[i][item]=Pole_IN_DataSet_list[i][item]-Pole_IN_DataSet_list[i][item][0]
                  Pole_IN_DataSet_list[i][item+'_detrend']=detrend(Pole_IN_DataSet_list[i][item])
                  Pole_IN_DataSet_list[i][item+'_detrend']=(Pole_IN_DataSet_list[i][item+'_detrend'])-(Pole_IN_DataSet_list[i][item+'_detrend'][0])
                  Pole_IN_DataSet_list[i][item+'_gradient']=np.gradient(Pole_IN_DataSet_list[i][item],1)
                  Pole_IN_DataSet_list[i][item+'_GdotD']=Pole_IN_DataSet_list[i][item+'_gradient']*Pole_IN_DataSet_list[i][item+'_detrend']        

    return  Pole_IN_DataSet_list,Pole_IN_Position_list

def get_pole_3d_data(poleid, dtype, anal_dir, plot_type, thr_over):
    logger.info('get_pole_3d_data) poleid={} dtype={} anal_dir={} plot_type={} thr_over={}'.format(poleid, dtype, anal_dir, plot_type, str(thr_over)))

    Pole_Thread_Value_PtoP_IN = 500
    if thr_over is not None:
        Pole_Thread_Value_PtoP_IN = thr_over

    if anal_dir == 'x':
        anal_dir = 0
    elif anal_dir == 'y':
        anal_dir = 1
    else:
        anal_dir = 2

    pole_meas_list = poledb.get_meas_result(poleid, dtype)
    print(pole_meas_list)

    ch_name_list = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']

    data = {}

    chart_data = []

    for i in range(len(pole_meas_list)):
        pole_meas = pole_meas_list.iloc[i]

        measno = pole_meas['measno']

        pole_data_x = poledb.get_meas_data(poleid, measno, dtype, 'x')
        logger.info('len(x)=' + str(pole_data_x.shape[0]))
        pole_data_y = poledb.get_meas_data(poleid, measno, dtype, 'y')
        logger.info('len(y)=' + str(pole_data_y.shape[0]))
        pole_data_z = poledb.get_meas_data(poleid, measno, dtype, 'z')
        logger.info('len(z)=' + str(pole_data_z.shape[0]))

        if dtype == 'IN':
            pole_meas['eddegree'] = 360
            pole_meas['stheight'] = pole_meas['stheight'] - pole_meas['depth']
            pole_meas['edheight'] = pole_meas['stheight']

        Start_point_Angle = pole_meas['stdegree']
        End_point_Angle = pole_meas['eddegree']
        Start_point = int(pole_meas['stheight'] * 1000)
        End_point = int(pole_meas['edheight'] * 1000)

        meas_info = {
            'meas_no' : int(measno),
            'stdegree': int(pole_meas['stdegree']),
            'eddegree': int(pole_meas['eddegree']),
            'stheight': float(pole_meas['stheight']),
            'edheight': float(pole_meas['edheight'])
        }

        pole_data_x = pole_data_x[ch_name_list]
        pole_data_y = pole_data_y[ch_name_list]
        pole_data_z = pole_data_z[ch_name_list]

        Pole_IN_DataSet_list,Pole_IN_Position_list = get_pole_anal_data(poleid, dtype, pole_data_x, pole_data_y, pole_data_z, Start_point_Angle, End_point_Angle, Start_point, End_point, Pole_Thread_Value_PtoP_IN)

        x = Pole_IN_DataSet_list[anal_dir][poleid+'_'+dtype+'_'+str(anal_dir+1)+'_Len']
        y = Pole_IN_Position_list
        X,Y = np.meshgrid(x,y)
        ZZ = []

        for j,item in enumerate(Pole_IN_DataSet_list[anal_dir].columns):
            print(j, item)
            if plot_type in item:
                if '_off' in item:
                    z=Pole_IN_DataSet_list[anal_dir][item].values.tolist() 
                    z=np.linspace(0,0,len(z)).tolist()
                    z=np.round(z, 3).tolist()
                    ZZ.append(z)
                else:
                    z=Pole_IN_DataSet_list[anal_dir][item].values.tolist()
                    z=np.round(z, 3).tolist()
                    ZZ.append(z)

        tmp = {}
        tmp['meas_info'] = meas_info
        tmp['x'] = np.round(Y, 3).tolist()
        tmp['y'] = np.round(X, 3).tolist()
        tmp['z'] = ZZ
        chart_data.append(tmp)

    data['chart_data'] = chart_data

    # 철근 정보
    total_End_len=2000 # mm, 2m
    total_Start_len=-1000 # mm, -1m
    x = [total_Start_len,total_End_len]
    y = np.linspace(0,360,16).tolist() # 각도, 0~360

    steel_info = []
    for i in range(16):
        tmp = {}
        tmp['x'] = [y[i], y[i]]
        tmp['y'] = x
        tmp['z'] = [0, 0]
        steel_info.append(tmp)

    data['steel_info'] = steel_info

    break_point_x=[1000,200]
    break_point_y=[90,180]
    break_point_z=[0,0]

    data['break_point'] = {
        'x': break_point_y,
        'y': break_point_x,
        'z': break_point_z
    }

    return data

def get_pole_vector_data(poleid, dtype, measno, anal_dir, stdegree, eddegree, stheight, edheight, thr_over=None, scale=None):
    logger.info('get_pole_vector_data) poleid={} dtype={} measno={} anal_dir={} degree={}~{} height={}~{} scale={}'.format(poleid, dtype, str(measno), str(measno), 
                                    anal_dir, str(stdegree), str(eddegree), str(stheight), str(edheight), str(scale)))

    Data_plot_type="detrend"

    Pole_Thread_Value_PtoP_IN = 500
    if thr_over is not None:
        Pole_Thread_Value_PtoP_IN = thr_over

    if scale is None:
        scale = 0.0001

    if anal_dir == 'x':
        anal_dir = 0
    elif anal_dir == 'y':
        anal_dir = 1
    else:
        anal_dir = 2

    pole_data_x = poledb.get_meas_data(poleid, measno, 'x')
    logger.info('len(x)=' + str(pole_data_x.shape[0]))
    pole_data_y = poledb.get_meas_data(poleid, measno, 'y')
    logger.info('len(y)=' + str(pole_data_y.shape[0]))
    pole_data_z = poledb.get_meas_data(poleid, measno, 'z')
    logger.info('len(z)=' + str(pole_data_z.shape[0]))

    Start_point_Angle = stdegree
    End_point_Angle = eddegree
    Start_point = int(stheight * 1000)
    End_point = int(edheight * 1000)

    ch_name_list = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']

    pole_data_x = pole_data_x[ch_name_list]
    pole_data_y = pole_data_y[ch_name_list]
    pole_data_z = pole_data_z[ch_name_list]

    Pole_IN_DataSet_list,Pole_IN_Position_list = get_pole_anal_data(poleid, dtype, pole_data_x, pole_data_y, pole_data_z, Start_point_Angle, End_point_Angle, Start_point, End_point, Pole_Thread_Value_PtoP_IN)

    x = Pole_IN_DataSet_list[anal_dir][poleid+'_'+dtype+'_'+str(anal_dir+1)+'_Len']
    y = Pole_IN_Position_list

    X,Y = np.meshgrid(x,y)

    ZZ = []
    XX = []
    YY = []

    for j,item in enumerate(Pole_IN_DataSet_list[0].columns):
                if Data_plot_type in item:
                        kk=Pole_IN_DataSet_list[0][item].values.tolist() 
                        XX.append(kk)
    VX = np.array(XX)

    for j,item in enumerate(Pole_IN_DataSet_list[2].columns):
                if Data_plot_type in item:
                        z=Pole_IN_DataSet_list[2][item].values.tolist() 
                        ZZ.append(z)
    VZ = np.array(ZZ)


    for j,item in enumerate(Pole_IN_DataSet_list[1].columns):
                if Data_plot_type in item:
                        k=Pole_IN_DataSet_list[1][item].values.tolist() 
                        YY.append(k)
    VY = np.array(YY)

    # vector data

    Vectoe_value=[]
    for i in range(VY.shape[0]):
        for j in range(X.shape[1]):
            a=[VY[i,j]*scale,VX[i,j]*scale,VZ[i,j]*scale]
            Vectoe_value.append(a)

    tvects = Vectoe_value

    orig=[]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            a=[ X[i,j], Y[i,j], 0 ]
            orig.append(a)

    if not hasattr(orig[0],"__iter__"):
        coords = [[orig[i],np.sum([orig[i],v],axis=0)] for i,v in enumerate(tvects)]
    else:
        coords = [[o,np.sum([o,v],axis=0)] for o,v in zip(orig,tvects)]

    data = []
    for i,c in enumerate(coords):
        X1, Y1, Z1 = zip(c[0])
        X2, Y2, Z2 = zip(c[1])

        tmp = {
            'name': 'V'+str(i+1),            
            'x': [round(Y1[0], 3), round(Y2[0], 3)],
            'y': [round(X1[0], 3), round(X2[0], 3)],
            'z': [round(Z1[0], 3), round(Z2[0], 3)]
        }

        data.append(tmp)

    logger.info(len(data))

    data = data[0:1000]

    return data

def pole_simulation(Main_mag_s,Sub_mag_s,Ga_mag_s,HallSensor_h):

    x0=mag3.magnet.Cylinder(magnetization=(0,0,Main_mag_s),dimension=(10,200),position=(0,0,100))
    
    x1=mag3.magnet.Cylinder(magnetization=(0,0,Main_mag_s),dimension=(10,100),position=(50,0,50))
    x2=mag3.magnet.Cylinder(magnetization=(0,0,-Main_mag_s),dimension=(10,100),position=(50,0,155))
    
    x3=mag3.magnet.Cylinder(magnetization=(0,0,Main_mag_s),dimension=(10,100),position=(100,0,50))
    x4=mag3.magnet.Cylinder(magnetization=(0,0,Main_mag_s),dimension=(10,100),position=(100,0,155))
    
    x5=mag3.magnet.Cylinder(magnetization=(0,0,Main_mag_s),dimension=(10,200),position=(150,0,100))
    
    x6=mag3.magnet.Cylinder(magnetization=(0,0,Main_mag_s),dimension=(10,120),position=(205,0,50))
    x7=mag3.magnet.Cylinder(magnetization=(0,0,Main_mag_s),dimension=(10,100),position=(200,0,150))
    
    x8=mag3.magnet.Cylinder(magnetization=(0,0,Main_mag_s),dimension=(10,200),position=(250,0,100))
    
    
    #띠철근
    x9=mag3.magnet.Cylinder(magnetization=(0,0,Sub_mag_s),dimension=(3,175),position=(175,0,50))
    x10=mag3.magnet.Cylinder(magnetization=(0,0,Sub_mag_s),dimension=(3,150),position=(175,0,50))
    
    x11=mag3.magnet.Cylinder(magnetization=(0,0,Sub_mag_s),dimension=(3,300),position=(175,0,100))
    
    x12=mag3.magnet.Cylinder(magnetization=(0,0,Sub_mag_s),dimension=(3,165),position=(175,0,147))
    x13=mag3.magnet.Cylinder(magnetization=(0,0,Sub_mag_s),dimension=(3,170),position=(175,0,150))
    
    
    #조각
    x14=mag3.magnet.Cylinder(magnetization=(0,0,Ga_mag_s),dimension=(10,30),position=(223,0,120))
    
    x15=mag3.magnet.Cylinder(magnetization=(0,0,Ga_mag_s),dimension=(10,30),position=(250,0,75))
    
    
    c=Collection(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15)
    
    x9.rotate_from_angax(90, 'y')
    x9.move((50,-0,0))
    
    x10.rotate_from_angax(90, 'y')
    x10.move((-120,-0,0))
    
    x11.rotate_from_angax(90, 'y')
    x11.move((-50,-0,0))
    
    x12.rotate_from_angax(90, 'y')
    x12.move((50,-0,0))
    
    x13.rotate_from_angax(90, 'y')
    x13.move((-100,-0,0))
    
    x15.rotate_from_angax(90, 'y')
    
    c.rotate_from_angax(90, 'x',(50,0,100))
    c.move((0,100,-100))
    
    xs=np.linspace(0, 250,250)
    ys=np.linspace(25, 175,150)
    
    POS=np.array([(x,y,HallSensor_h) for y in ys for x in xs ])
    
    Bs=c.getB(POS).reshape(150,250,3)
    
    X,Y=np.meshgrid(xs,ys)

    data = {}

    chart_data = {}

    chart_data['Bs'] = np.round(Bs, 2).tolist()
    chart_data['X'] = np.round(X, 2).tolist()
    chart_data['Y'] = np.round(Y, 2).tolist()

    data['chart_data'] = chart_data

    step_data = []

    for step in range(10,240):
        tmp = {}
        tmp['step_no'] = step-10
        tmp['name'] = 'Time = ' + str(step-10)
        tmp['3d_chart_line'] = { 'x': [step-10,step-10], 'y':[0,200], 'z': [1,1] }
        tmp['3d_chart_line1'] = { 'x': [step,step], 'y':[0,200], 'z': [1,1] }
        tmp['3d_chart_line2'] = { 'x': [step+10,step+10], 'y':[0,200], 'z': [1,1] }

        tmp['x'] = np.round(Y[:,1], 2).tolist()
        tmp['chart_x'] = np.round(Bs[:,step-10,0], 2).tolist()
        tmp['chart_x1'] = np.round(Bs[:,step,0], 2).tolist()
        tmp['chart_x2'] = np.round(Bs[:,step+10,0], 2).tolist()

        tmp['chart_y'] = np.round(Bs[:,step-10,1], 2).tolist()
        tmp['chart_y1'] = np.round(Bs[:,step,1], 2).tolist()
        tmp['chart_y2'] = np.round(Bs[:,step+10,1], 2).tolist()        

        tmp['chart_z'] = np.round(Bs[:,step-10,2], 2).tolist()
        tmp['chart_z1'] = np.round(Bs[:,step,2], 2).tolist()
        tmp['chart_z2'] = np.round(Bs[:,step+10,2], 2).tolist()
        step_data.append(tmp)

    data['step_data'] = step_data

    return data

if __name__ == '__main__':
    poledb.poledb_init()

    poleid = '1001A011'
    dtype = 'OUT'
    measno = 1
    anal_dir = 'z'
    stdegree = 0
    eddegree = 90
    stheight = 0.0
    edheight = 1.0
    scale = 0.001

    v = get_pole_vector_data(poleid, dtype, measno, anal_dir, stdegree, eddegree, stheight, edheight, scale)
