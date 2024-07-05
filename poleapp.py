
from flask import Flask, jsonify, request
import sys, getopt
import pole_anal
import poledb
import threading
import time

# Error Message
HTTP_ERR_MANDATORY_PARAMETER_MISSING = 'Mandatory Parameter Missing'
HTTP_ERR_WRONG_PARAMETER_VALUE = 'Wrong Parameter Value'

app = Flask(__name__)

def get_ok_resp(data):
    return jsonify(success=True, data=data, error_msg='')

def get_nok_resp(error_msg):
    return jsonify(success=False, data=None, error_msg=error_msg)

@app.route('/polediag/getPoleDataAnalResult', methods=['GET'])
def get_anal_result():
    data = {}

    poleid = request.args.get('poleid', default='', type=str)
    dtype = request.args.get('dtype', default='', type=str) # 'IN','OUT'
    measno = request.args.get('measno', default=-1, type=int)
    thr_over = request.args.get('thr_over', default=-1, type=int)
    param_thr_val = request.args.get('thr_val', default=-1, type=int)

    in_hall_height = request.args.get('in_hall_height', default=-1, type=float)
    in_insert_len = request.args.get('in_insert_len', default=-1, type=float)
    in_st_angle = request.args.get('in_st_angle', default=-1, type=int)

    out_st_height = request.args.get('out_st_height', default=-1, type=float)
    out_ed_height = request.args.get('out_ed_height', default=-1, type=float)
    out_st_angle = request.args.get('out_st_angle', default=-1, type=int)

    if poleid == '' or dtype == '' or measno == -1:
        return get_nok_resp(HTTP_ERR_MANDATORY_PARAMETER_MISSING)

    if dtype.lower() == 'in':
        if in_hall_height == -1 or in_insert_len == -1 or in_st_angle == -1:
            return get_nok_resp(HTTP_ERR_MANDATORY_PARAMETER_MISSING)
    elif dtype.lower() == 'out':
        if out_st_height == -1 or out_ed_height == -1 or out_st_angle == -1:
            return get_nok_resp(HTTP_ERR_MANDATORY_PARAMETER_MISSING)
    else:
        return get_nok_resp(HTTP_ERR_WRONG_PARAMETER_VALUE)

    thr_val = None
    if thr_over == 1 and param_thr_val != -1:
        thr_val = param_thr_val

    if dtype.lower() == 'in':
        data = pole_anal.get_pole_anal_result_in(poleid, measno, in_hall_height, in_insert_len, in_st_angle, param_thr_val)
    else:
        data = pole_anal.get_pole_anal_result_out(poleid, measno, out_st_height, out_ed_height, out_st_angle, param_thr_val)

    return get_ok_resp(data)

@app.route('/polediag/getPole3DData', methods=['GET'])
def get_pole_3d_data():
    data = {}

    poleid = request.args.get('poleid', default='', type=str)
    dtype = request.args.get('dtype', default='', type=str) # 'IN','OUT'
    anal_dir = request.args.get('anal_dir', default='', type=str) # 'x','y','z'
    plot_type = request.args.get('plot_type', default='', type=str) # 'x','y','z'
    thr_over = request.args.get('thr_over', default=None, type=int)

    if poleid == '' or dtype == '' or anal_dir =='' or plot_type == '':
        return get_nok_resp(HTTP_ERR_MANDATORY_PARAMETER_MISSING)

    data = pole_anal.get_pole_3d_data(poleid, dtype, anal_dir, plot_type, thr_over)

    return get_ok_resp(data)

@app.route('/polediag/getPoleVectorData', methods=['GET'])
def get_pole_vector_data():
    data = {}

    poleid = request.args.get('poleid', default='', type=str)
    dtype = request.args.get('dtype', default='', type=str) # 'IN','OUT'
    measno = request.args.get('measno', default=-1, type=int)
    anal_dir = request.args.get('anal_dir', default=None, type=str) # 'x','y','z'
    stdegree = request.args.get('stdegree', default=None, type=int)
    eddegree = request.args.get('eddegree', default=None, type=int)
    stheight = request.args.get('stheight', default=None, type=float)
    edheight = request.args.get('edheight', default=None, type=float)
    thr_over = request.args.get('thr_over', default=None, type=int)
    scale = request.args.get('scale', default=None, type=float)

    if poleid == '' or dtype == '' or measno == -1:
        return get_nok_resp(HTTP_ERR_MANDATORY_PARAMETER_MISSING)

    if anal_dir is None or stdegree is None or eddegree is None or stheight is None or edheight is None:
        return get_nok_resp(HTTP_ERR_MANDATORY_PARAMETER_MISSING)

    data = pole_anal.get_pole_vector_data(poleid, dtype, measno, anal_dir, stdegree, eddegree, stheight, edheight, thr_over, scale)

    return get_ok_resp(data)

@app.route('/polediag/getPoleSimulData', methods=['GET'])
def get_pole_simul_data():
    data = {}

    Main_mag_s = request.args.get('main_mag_s', default=None, type=int)
    Sub_mag_s = request.args.get('sub_mag_s', default=None, type=int)
    Ga_mag_s = request.args.get('ga_mag_s', default=None, type=int)
    HallSensor_h = request.args.get('hallsensor_h', default=None, type=int)

    if Main_mag_s is None or Sub_mag_s is None or Ga_mag_s is None or HallSensor_h is None:
        return get_nok_resp(HTTP_ERR_MANDATORY_PARAMETER_MISSING)

    data = pole_anal.pole_simulation(Main_mag_s, Sub_mag_s, Ga_mag_s, HallSensor_h)

    return get_ok_resp(data)

def check_db():
    while True:
        poledb.ping()
        time.sleep(600)

if __name__ == '__main__':
    PORT = 9001
    argv = sys.argv

    poledb.poledb_init()

    t = threading.Thread(target=check_db)
    t.daemon = True
    t.start()

    try:
        opts, etc_args = getopt.getopt(argv[1:], 'hp:', ['help', 'port='])
    except getopt.GetoptError:
        print(argv[0], '-p <port>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(argv[0], '-p <port> -m <mode>')
            sys.exit()
        elif opt in ('-p', '--port'):
            PORT = int(arg)

    print('PORT=%d' %(PORT))

    app.run(host='0.0.0.0', port=PORT)
