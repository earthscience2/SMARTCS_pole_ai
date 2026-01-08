import poledb as PDB
import os
from tqdm import tqdm  
import shutil

global in_num
global out_num

def Signal_Info_out(poleid,stype,num):
    global out_x
    global out_y
    global out_z
    print('mum',poleid,stype,num)
    sig_x=PDB.get_meas_data(poleid,num,stype,'x')
    sig_y=PDB.get_meas_data(poleid,num,stype,'y')
    sig_z=PDB.get_meas_data(poleid,num,stype,'z')
    out_x = sig_x
    out_y = sig_y
    out_z = sig_z
    
def Signal_Info_in(poleid,stype,num):
    
    global sig_x
    sig_x=PDB.get_meas_data(poleid,num,stype,'x')
    index_ch=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']
    sig_x_ptp=[]
    sig_Scan_points=0
    x=0
    
def pole_Info(poleid):
    global in_num
    global out_num
    global re_out
    global re_in
    pole_sgn_count=PDB.get_meas_result_count(poleid)
    re_out=PDB.get_meas_result(poleid, 'OUT')
    num_sig_out=re_out.shape[0]
    re_in=PDB.get_meas_result(poleid, 'IN')
    num_sig_in=re_in.shape[0]
    in_num = num_sig_in
    out_num = num_sig_out
    
    return num_sig_out,num_sig_in,re_out,re_in

def conf_file_open(dir, poleid_name):
    check = "no"
    Pole_list = PDB.get_pole_list(dir)
    if Pole_list.empty:
        print(dir ,  " 데이터가 없습니다.")
    else:
        main_path = 'raw_data/' + dir + " " + poleid_name + " 원본데이터"
        dir_path_csv = main_path + "/" + dir + " " + poleid_name + ' csv'
        dir_path_xlsx = main_path + "/" + dir + " " + poleid_name + ' xlsx'
        
        if not os.path.exists(main_path):
            os.mkdir(main_path)
        if not os.path.exists(dir_path_csv):
            os.mkdir(dir_path_csv)
        if not os.path.exists(dir_path_xlsx):
            os.mkdir(dir_path_xlsx)
            
        print(dir, " 원본 데이터 수집")
        for poleid in tqdm(Pole_list['poleid'], desc="Processing poles"):
            pole_info = pole_Info(poleid)
            if  poleid == poleid_name:
                check = "yes"
                for kk in range(in_num):
                    stype = 'IN'
                    num = int(re_in['measno'][kk])
                    time = str(re_in['sttime'][kk])
                    time = (time.split(" "))[0]
                    in_x = PDB.get_meas_data(poleid, num, stype, 'x')
                    in_x.to_csv(os.path.join(dir_path_csv, dir + '_' + f"{poleid}_{kk+1}_{time}_T.csv"), index=False)
                    in_x.to_excel(os.path.join(dir_path_xlsx, dir + '_' + f"{poleid}_{kk+1}_{time}_T.xlsx"), index=False)

                for kk in range(out_num):
                    stype = 'OUT'
                    num = int(re_out['measno'][kk])
                    time = str(re_out['sttime'][kk])
                    time = (time.split(" "))[0]
                    out_z = PDB.get_meas_data(poleid, num, stype, 'z')
                    out_z.to_csv(os.path.join(dir_path_csv, dir + '_' + f"{poleid}_{kk+1}_{time}_H.csv"), index=False)
                    out_z.to_excel(os.path.join(dir_path_xlsx, dir + '_' + f"{poleid}_{kk+1}_{time}_H.xlsx"), index=False)
                    
        if check == "no":
            print(poleid_name + " 전주가 존재하지 않습니다")
            if os.path.exists(main_path):
                shutil.rmtree(main_path)
        else:
            print(poleid_name + " 전주 데이터 수집완료")
                
            
        
#====================================================================
# 원본데이터 수집
#====================================================================
name_list = "노원도봉-2503"
poleid_name_list = ["0432R025", "0429P091"]
# main(메인서버), is(이수서버), kh(건화서버), jt(제이티엔지니어링)
server = "is" 

PDB.poledb_init(server)
for poleid_name in poleid_name_list:
    conf_file_open(name_list, poleid_name)
