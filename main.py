import datetime
import os
import time
import random
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, session, Response
import traceback
import pymysql
from camera_opencv import Camera
from werkzeug.utils import secure_filename
from detect import *

app = Flask(__name__)
app.secret_key = 'any random string'
# 车牌图片存储的路径
app.config['UPLOAD_FOLDER'] = 'static/img/upload/'


# 首页--车牌识别
@app.route('/')
def index():
    db = pymysql.connect(host="localhost", user="root", password="qy020320", database="car_flask")
    cursor = db.cursor()
    sql1 = "select driveway,in_time,plate from plate;"
    try:
        # 执行sql语句
        cursor.execute(sql1)
        results = cursor.fetchall()
        # 历遍results，使对应车位时间存入对应session
        for i in range(80):
            session[str(i + 1)] = 0
        for i in results:
            for j in range(80):
                if i[0] == j + 1:
                    session[str(j + 1)] = i[1] + '\n' + '车牌号：' + i[2]
    except:
        traceback.print_exc()
        db.rollback()
        return '失败'
    db.close()
    return render_template("index.html")


# 车牌识别界面
@app.route('/upload')
def upload_file():
    return render_template('upload.html')


# 获取上传的车牌图片并识别返回数据
@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        # 获取格式为年-月-日 时：分：秒的当前时间
        clock = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        # 获取格式为时：分：秒的当前时间
        employee_clock = time.strftime('%H:%M:%S', time.localtime())
        # 停车时间
        park_time = "--"
        # 获取post的file
        f = request.files['file']
        # 在html中显示图片的路径名--图片的存储路径
        file_path = "static/img/upload/" + str(f.filename)
        char_all, color = detect(file_path)
        if len(char_all) == 0:
            print('未能正确识别车牌')
            plate = '未能正确识别车牌'
            plate_type = ''
            location = ''
            warn=''
            return render_template("upload.html",time=clock, parking=park_time, fee='',
                                           file_path=file_path, warn=warn, plate=plate,location=location,plate_type=plate_type)
        else:
            plate = ''.join(char_all)
            location_label = char_all[0]
            LOCATION_DICT = {"川": '四川省', "鄂": '湖北省', "赣": '江西省', "甘": '甘肃省', "贵": '贵州省', "桂": '广西壮族自治区',
                             '黑': '黑龙江省', "沪": '上海市', "冀": '河北省', "津": '天津市',
                             "京": '北京市', "吉": '吉林省', "辽": '辽宁省', "鲁": '山东省', "蒙": '内蒙古自治区', "闽": '福建省', "宁": '宁夏回族自治区',
                             "青": '青海省', "琼": '海南省', "陕": '山西省',
                             "苏": '江苏省', "晋": '山西省', "皖": '安徽省', "湘": '湖南省', "新": '新疆维吾尔自治区', "豫": '河南省', "渝": '重庆市',
                             "粤": '广东省', "云": '云南省', "藏": '西藏自治区', "浙": '浙江省'}
            COLOR_DICT = {'蓝色': '燃油汽车', '绿色': '新能源汽车', '黄色': '客车'}
            plate_type = COLOR_DICT[color]
            location = LOCATION_DICT[location_label]
        # 把用户名和密码注册到数据库中
        # 连接数据库,此前在数据库中创建数据库
        db = pymysql.connect(host="localhost", user="root", password="qy020320", database="car_flask")
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # s记录随机选择车位
        s = random.randint(1, 80)
        # sql语句
        sql1 = "select in_time from plate where plate='" + str(plate) + "';"
        sql2 = "insert into plate values ('" + str(plate) + "','" + str(clock) + "',null," + str(s) + ");"
        sql3 = "delete from plate where plate='" + str(plate) + "';"
        sql4 = "select * from plate where driveway=" + str(s) + ";"
        sql5 = "select * from plate;"
        sql6 = "select * from question where plate='" + str(plate) + "';"
        sql7 = "select type from fast_pass where plate='" + str(plate) + "';"
        sql8 = "select start_time ,end_time from employee where plate='" + str(plate) + "';"
        try:
            # 执行sql语句
            cursor.execute(sql1)
            results = cursor.fetchall()
            cursor.execute(sql8)
            employee_date = cursor.fetchall()
            # 车辆出库
            if len(results) == 1:
                # 把时间从string类型转为datetime.datetime类型
                in_time = datetime.datetime.strptime(results[0][0], '%Y-%m-%d %H:%M:%S')
                out_time = datetime.datetime.strptime(clock, '%Y-%m-%d %H:%M:%S')
                # 获取时间差--为datetime.timedelta类型
                park_time = out_time - in_time
                fifteenmin = datetime.timedelta(minutes=15)
                onehour = datetime.timedelta(hours=1)
                # 是否是救护车或者消防车
                cursor.execute(sql7)
                is_fast_pass = cursor.fetchall()
                if len(is_fast_pass) == 1:
                    # 判断type
                    if is_fast_pass[0][0] == "救护车":
                        warn = "紧急情况，请救护车快速通过"
                    else:
                        warn = "紧急情况，请消防车快速通过"
                    return render_template("upload.html", time=clock, parking=park_time, fee="0元",
                                           file_path=file_path, warn=warn, plate=plate,location=location,plate_type=plate_type)
                else:
                    # 车辆入库15分钟内免费
                    if fifteenmin >= park_time:
                        fee = 0
                    # 入库没满一个小时按一个小时计算车费
                    elif fifteenmin < park_time <= onehour:
                        fee = 3
                    else:
                        # i为车费按几小时计算
                        i = 2
                        while park_time > datetime.timedelta(hours=i):
                            i = i + 1
                        fee = 3 * i
                    # 出库的是员工车辆
                    if len(employee_date) != 0:
                        # end_time下班时间、now_time当前时间
                        end_time = datetime.datetime.strptime(employee_date[0][1], '%H:%M:%S')
                        now_time = datetime.datetime.strptime(employee_clock, '%H:%M:%S')
                        # 判断是否下班，未下班离开发出警告
                        # 员工车费为0
                        if (end_time - now_time) > datetime.timedelta(minutes=0):
                            warn = "您离下班时间还有" + str(end_time - now_time)
                            fee = 0
                    else:
                        warn = ""
                    # 删除数据库里车辆信息
                    cursor.execute(sql3)
                    db.commit()
                    # 车费封顶20元
                    if fee > 20:
                        fee = 20
                    fee = str(fee) + '元'
                    # 返回time当前时间、parking停车时间、file_path图片路径、fee车费、warn警告到车牌识别界面
                    return render_template("upload.html", time=clock, parking=park_time, file_path=file_path,
                                           fee=fee, warn=warn, plate=plate,location=location,plate_type=plate_type)
            # 车辆入库
            else:
                # 是否是问题车辆
                cursor.execute(sql6)
                is_question = cursor.fetchall()
                # 是否是救护车或者消防车
                cursor.execute(sql7)
                is_fast_pass = cursor.fetchall()
                # 是问题车辆
                if len(is_question) == 1:
                    warn = "警告：发现问题车辆！！！"
                    return render_template("upload.html", time=clock, file_path=file_path, warn=warn, plate=plate,location=location,
                                           plate_type=plate_type)
                # 是救护车或者消防车
                elif len(is_fast_pass) == 1:
                    # 判断type
                    if is_fast_pass[0][0] == "救护车":
                        warn = "紧急情况，请救护车快速通过"
                    else:
                        warn = "紧急情况，请消防车快速通过"
                    return render_template("upload.html", time=clock, file_path=file_path, warn=warn, plate=plate,location=location,
                                           plate_type=plate_type)
                # 正常车
                else:
                    # 入库的是员工车辆
                    if len(employee_date) != 0:
                        start_time = datetime.datetime.strptime(employee_date[0][0], '%H:%M:%S')
                        now_time = datetime.datetime.strptime(employee_clock, '%H:%M:%S')
                        if (now_time - start_time) > datetime.timedelta(minutes=0):
                            warn = "您已迟到" + str(now_time - start_time)
                    else:
                        warn = ""
                    cursor.execute(sql5)
                    number = cursor.fetchall()
                    # 车库里车辆总数是否超过80
                    if len(number) < 80:
                        # 若随机的车位序号已有则循环直至序号为未出现的值
                        while session.get(str(s))!=0 and session.get(str(s))!=None:
                            s = s+1
                            if s>80:
                                s=1
                            print(s)
                            print(session.get(str(s)))
                        # 把车辆信息传入数据库
                        print(s)
                        cursor.execute(sql2)
                        db.commit()
                        # 把车位序号转换为车位号
                        site = ''
                        if (s - 1) // 20 == 0:
                            site += 'A'
                        elif (s - 1) // 20 == 1:
                            site += 'B'
                        elif (s - 1) // 20 == 2:
                            site += 'C'
                        elif (s - 1) // 20 == 3:
                            site += 'D'
                        elif (s - 1) // 20 == 4:
                            site += 'E'
                        else:
                            site += 'F'
                        j = (s - 1) % 20
                        if (j + 1) // 10 == 0:
                            site += '00' + str(j + 1)
                        else:
                            site += '0' + str(j + 1)
                    # 超过80显示车位已满
                    else:
                        site = '车位已满'
                    return render_template("upload.html", time=clock, parking=park_time,
                                           file_path=file_path, site=site, warn=warn, plate=plate,location=location,plate_type=plate_type)
        except:
            # 抛出错误信息
            traceback.print_exc()
            # 如果发生错误则回滚
            db.rollback()
            return '上传失败'
        db.close


# 登陆界面
@app.route('/login')
def loginpage():
    return render_template("login.html")


# 登录过程
@app.route('/loginProcess', methods=['POST', 'GET'])
def loginProcesspage():
    if request.method == 'POST':
        db = pymysql.connect(host="localhost", user="root", password="qy020320", database="car_flask")
        cursor = db.cursor()
        sql1 = "select * from user where email='" + str(request.form['exampleInputEmail2']) + \
               "' and username='" + str(request.form['exampleInputName2']) + "' and password='" + \
               str(request.form['exampleInputPassword3']) + "';"
        sql2 = "update user set login = True where email='" + str(request.form['exampleInputEmail2']) + \
               "' and username='" + str(request.form['exampleInputName2']) + "' and password='" + \
               str(request.form['exampleInputPassword3']) + "';"
        try:
            cursor.execute(sql1)
            results = cursor.fetchall()
            print(len(results))
            # 已经注册
            if len(results) == 1:
                # 获取登陆的用户名并存到session中
                session['username'] = request.form['exampleInputName2']
                # 更新数据库中数据为已登陆
                cursor.execute(sql2)
                db.commit()
                return render_template('index.html')
            else:
                return '用户名或密码不正确'
        except:
            traceback.print_exc()
            db.rollback()
            return '登陆失败'
        db.close()


# 注册界面
@app.route('/enroll')
def enrollpage():
    return render_template("enroll.html")


# 注册过程
@app.route('/enrollProcess', methods=['POST', 'GET'])
def enrollProcesspage():
    if request.method == 'POST':
        db = pymysql.connect(host="localhost", user="root", password="qy020320", database="car_flask")
        cursor = db.cursor()
        # 输入不为空
        if request.form['exampleInputPassword1'] != '' and request.form['exampleInputEmail1'] != '' \
                and request.form['exampleInputName1'] != '':
            # 两次密码是否相同
            if request.form['exampleInputPassword1'] == request.form['exampleInputPassword2']:
                # 注册信息传入数据库
                sql = "insert into user values ('" + str(request.form['exampleInputEmail1']) + "', '" + \
                      str(request.form['exampleInputName1']) + "', '" + \
                      str(request.form['exampleInputPassword1']) + "', False);"
                try:
                    cursor.execute(sql)
                    db.commit()
                    return render_template('login.html')
                except:
                    traceback.print_exc()
                    db.rollback()
                    return '注册失败'
            else:
                return '密码输入错误，请重新注册'
        else:
            return '请补充完整表格'
        db.close()


# 注销界面
@app.route('/logout')
def logoutpage():
    return render_template("logout.html")


# 注销过程
@app.route('/logoutProcess', methods=['POST', 'GET'])
def logoutProcesspage():
    if request.method == 'POST':
        db = pymysql.connect(host="localhost", user="root", password="qy020320", database="car_flask")
        cursor = db.cursor()
        sql1 = "select * from user where email='" + str(request.form['exampleInputEmail3']) + \
               "' and username='" + str(request.form['exampleInputName3']) + "' and password='" + \
               str(request.form['exampleInputPassword4']) + "'and login=True;"
        sql2 = "DELETE FROM user WHERE email = '" + str(request.form['exampleInputEmail3']) + "';"
        try:
            # 确认注销用户信息
            cursor.execute(sql1)
            results = cursor.fetchall()
            print(len(results))
            if len(results) == 1:
                # 清除session中‘username’
                session.pop('username')
                # 删除数据库中用户信息
                cursor.execute(sql2)
                db.commit()
                return render_template('index.html')
            else:
                return '用户名或密码不正确'
        except:
            traceback.print_exc()
            db.rollback()
            return '注销失败'
        db.close()


# 登出界面
@app.route('/logoff')
def logoffpage():
    db = pymysql.connect(host="localhost", user="root", password="qy020320", database="car_flask")
    cursor = db.cursor()
    # 改变数据库中登陆状态
    sql1 = "update user set login = False where username='" + str(session['username']) + "';"
    try:
        cursor.execute(sql1)
        db.commit()
    except:
        traceback.print_exc()
        db.rollback()
        return '登出失败'
    db.close()
    # 清除session中‘username’数据
    session.pop('username')
    return render_template("index.html")


# 个人界面
@app.route('/users')
def users():
    usersdata = "账户"
    return render_template("users.html", data=usersdata)


# 停车场地图界面
@app.route('/park')
def parkypage():
    db = pymysql.connect(host="localhost", user="root", password="qy020320", database="car_flask")
    cursor = db.cursor()
    sql1 = "select driveway,in_time,plate from plate;"
    try:
        # 执行sql语句
        cursor.execute(sql1)
        results = cursor.fetchall()
        # number为车库中剩余车位数量
        number = 80 - len(results)
        # 历遍results，使对应车位时间存入对应session
        for i in range(80):
            session[str(i+1)]=0
        for i in results:
            for j in range(80):
                if i[0]==j+1:
                    session[str(j+1)]=i[1]+'\n'+'车牌号：'+i[2]
        return render_template("park.html", number=number)
    except:
        traceback.print_exc()
        db.rollback()
        return '失败'
    db.close()


# 收费细则界面
@app.route('/fee_rules')
def fee_rulespage():
    return render_template("fee_rules.html")


# 实时识别车牌
@app.route('/video')
def video():
    return render_template("video.html")


@app.route('/video_upload', methods=['GET', 'POST'])
def video_upload():
    if request.method == 'POST':
        clock = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        employee_clock = time.strftime('%H:%M:%S', time.localtime())
        park_time = "--"
        char_all,color = detect('static/img/video/catch.jpg')
        if len(char_all)==0:
            print('未能正确识别车牌')
            plate='未能正确识别车牌'
            plate_type = ''
            location = ''
            warn = ''
            return render_template("video.html", time=clock, parking=park_time, fee='', warn=warn, plate=plate, location=location,
                                   plate_type=plate_type)
        else:
            plate = ''.join(char_all)
            location_label = char_all[0]
            LOCATION_DICT = {"川": '四川省', "鄂": '湖北省', "赣": '江西省', "甘": '甘肃省', "贵": '贵州省', "桂": '广西壮族自治区',
                             '黑': '黑龙江省', "沪": '上海市', "冀": '河北省', "津": '天津市',
                             "京": '北京市', "吉": '吉林省', "辽": '辽宁省', "鲁": '山东省', "蒙": '内蒙古自治区', "闽": '福建省', "宁": '宁夏回族自治区',
                             "青": '青海省', "琼": '海南省', "陕": '山西省',
                             "苏": '江苏省', "晋": '山西省', "皖": '安徽省', "湘": '湖南省', "新": '新疆维吾尔自治区', "豫": '河南省', "渝": '重庆市',
                             "粤": '广东省', "云": '云南省', "藏": '西藏自治区', "浙": '浙江省'}
            COLOR_DICT = {'蓝色': '燃油汽车', '绿色': '新能源汽车'}
            plate_type = COLOR_DICT[color]
            location = LOCATION_DICT[location_label]
        db = pymysql.connect(host="localhost", user="root", password="qy020320", database="car_flask")
        cursor = db.cursor()
        s = random.randint(1, 80)
        sql1 = "select in_time from plate where plate='" + str(plate) + "';"
        sql2 = "insert into plate values ('" + str(plate) + "','" + str(clock) + "',null," + str(s) + ");"
        sql3 = "delete from plate where plate='" + str(plate) + "';"
        sql4 = "select * from plate where driveway=" + str(s) + ";"
        sql5 = "select * from plate;"
        sql6 = "select * from question where plate='" + str(plate) + "';"
        sql7 = "select type from fast_pass where plate='" + str(plate) + "';"
        sql8 = "select start_time ,end_time from employee where plate='" + str(plate) + "';"
        try:
            # 执行sql语句
            cursor.execute(sql1)
            results = cursor.fetchall()
            cursor.execute(sql8)
            employee_date = cursor.fetchall()
            if len(results) == 1:
                # 把时间从string类型转为datetime.datetime类型
                in_time = datetime.datetime.strptime(results[0][0], '%Y-%m-%d %H:%M:%S')
                out_time = datetime.datetime.strptime(clock, '%Y-%m-%d %H:%M:%S')
                # 获取时间差--为datetime.timedelta类型
                park_time = out_time - in_time
                fifteenmin = datetime.timedelta(minutes=15)
                onehour = datetime.timedelta(hours=1)
                # 是否是救护车或者消防车
                cursor.execute(sql7)
                is_fast_pass = cursor.fetchall()
                if len(is_fast_pass) == 1:
                    # 判断type
                    if is_fast_pass[0][0] == "救护车":
                        warn = "紧急情况，请救护车快速通过"
                    else:
                        warn = "紧急情况，请消防车快速通过"
                    return render_template("video.html", time=clock, parking=park_time, fee="0元", warn=warn,
                                           plate=plate,location=location,plate_type=plate_type)
                else:
                    # 车辆入库15分钟内免费
                    if fifteenmin >= park_time:
                        fee = 0
                    # 入库没满一个小时按一个小时计算车费
                    elif fifteenmin < park_time <= onehour:
                        fee = 3
                    else:
                        # i为车费按几小时计算
                        i = 2
                        while park_time > datetime.timedelta(hours=i):
                            i = i + 1
                        fee = 3 * i
                    # 出库的是员工车辆
                    if len(employee_date) != 0:
                        # end_time下班时间、now_time当前时间
                        end_time = datetime.datetime.strptime(employee_date[0][1], '%H:%M:%S')
                        now_time = datetime.datetime.strptime(employee_clock, '%H:%M:%S')
                        # 判断是否下班，未下班离开发出警告
                        # 员工车费为0
                        if (end_time - now_time) > datetime.timedelta(minutes=0):
                            warn = "您离下班时间还有" + str(end_time - now_time)
                            fee = 0
                    else:
                        warn = ""
                    # 删除数据库里车辆信息
                    cursor.execute(sql3)
                    db.commit()
                    # 车费封顶20元
                    if fee > 20:
                        fee = 20
                    fee = str(fee) + '元'
                    # 返回time当前时间、parking停车时间、file_path图片路径、fee车费、warn警告到车牌识别界面
                    return render_template("video.html", time=clock, parking=park_time, fee=fee, warn=warn, plate=plate
                                           ,location=location,plate_type=plate_type)
            # 车辆入库
            else:
                # 是否是问题车辆
                cursor.execute(sql6)
                is_question = cursor.fetchall()
                # 是否是救护车或者消防车
                cursor.execute(sql7)
                is_fast_pass = cursor.fetchall()
                # 是问题车辆
                if len(is_question) == 1:
                    warn = "警告：发现问题车辆！！！"
                    return render_template("video.html", time=clock, warn=warn, plate=plate,location=location,plate_type=plate_type)
                # 是救护车或者消防车
                elif len(is_fast_pass) == 1:
                    # 判断type
                    if is_fast_pass[0][0] == "救护车":
                        warn = "紧急情况，请救护车快速通过"
                    else:
                        warn = "紧急情况，请消防车快速通过"
                    return render_template("video.html", time=clock, warn=warn, plate=plate,location=location,plate_type=plate_type)
                # 正常车
                else:
                    # 入库的是员工车辆
                    if len(employee_date) != 0:
                        start_time = datetime.datetime.strptime(employee_date[0][0], '%H:%M:%S')
                        now_time = datetime.datetime.strptime(employee_clock, '%H:%M:%S')
                        if (now_time - start_time) > datetime.timedelta(minutes=0):
                            warn = "您已迟到" + str(now_time - start_time)
                    else:
                        warn = ""
                    cursor.execute(sql5)
                    number = cursor.fetchall()
                    # 车库里车辆总数是否超过80
                    if len(number) < 80:
                        # 若随机的车位序号已有则循环直至序号为未出现的值
                        cursor.execute(sql4)
                        results_num = cursor.fetchall()
                        while len(results_num) == 1:
                            s = random.randint(1, 80)
                            cursor.execute(sql4)
                            results_num = cursor.fetchall()
                        # 把车辆信息传入数据库
                        cursor.execute(sql2)
                        db.commit()
                        # 把车位序号转换为车位号
                        site = ''
                        if (s - 1) // 20 == 0:
                            site += 'A'
                        elif (s - 1) // 20 == 1:
                            site += 'B'
                        elif (s - 1) // 20 == 2:
                            site += 'C'
                        elif (s - 1) // 20 == 3:
                            site += 'D'
                        elif (s - 1) // 20 == 4:
                            site += 'E'
                        else:
                            site += 'F'
                        j = (s - 1) % 20
                        if (j + 1) // 10 == 0:
                            site += '00' + str(j + 1)
                        else:
                            site += '0' + str(j + 1)
                    # 超过80显示车位已满
                    else:
                        site = '车位已满'
                    return render_template("video.html", time=clock, parking=park_time, site=site, warn=warn,
                                           plate=plate,location=location,plate_type=plate_type)
        except:
            # 抛出错误信息
            traceback.print_exc()
            # 如果发生错误则回滚
            db.rollback()
            return '上传失败'
        db.close


def gen(camera):
    # 视频流生成
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    # 视频流的路线。将其放在img标记的src属性中。
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
