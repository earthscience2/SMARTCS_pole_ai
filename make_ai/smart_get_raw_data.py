import poledb as PDB
import os
from tqdm import tqdm  

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

def conf_file_open(dir):
    Pole_list = PDB.get_pole_list(dir)
    if Pole_list is None:
        print(dir, " 데이터베이스 연결 실패 또는 데이터를 가져올 수 없습니다.")
        return
    if Pole_list.empty:
        print(dir ,  " 데이터가 없습니다.")
    else:
        main_path = 'raw_data/' + dir + " 원본데이터"
        dir_path_csv = main_path + '/' + dir + ' csv'
        dir_path_xlsx = main_path + '/' + dir + ' xlsx'
        
        if not os.path.exists(main_path):
            os.mkdir(main_path)
        if not os.path.exists(dir_path_csv):
            os.mkdir(dir_path_csv)
        if not os.path.exists(dir_path_xlsx):
            os.mkdir(dir_path_xlsx)
            
        print(dir, " 원본 데이터 수집")
        for poleid in tqdm(Pole_list['poleid'], desc="Processing poles"):
            pole_info = pole_Info(poleid)

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
        
#====================================================================
# 원본데이터 수집
#====================================================================
name_list = ["천안지사2-2511",
             "아산지사2-2511"]
# main(메인서버), is(이수서버), kh(건화서버), jt(제이티엔지니어링)
server = "is" 

PDB.poledb_init(server)
for i in name_list:
    conf_file_open(i)

print("데이터 추출 완료")