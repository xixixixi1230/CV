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
    msg = "识别中国大陆各类机动车车牌信息，支持蓝牌、绿牌。"
    return render_template("index.html", data=msg)


# 车牌识别界面
@app.route('/upload')
def upload_file():
    return render_template('upload.html')


# 获取上传的车牌图片并识别返回数据
@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        # 获取post的file
        f = request.files['file']
        # 在html中显示图片的路径名--图片的存储路径
        file_path = "static/img/upload/" + str(f.filename)
        char_all=detect(file_path)
        plate=''.join(char_all)
        # 获取格式为年-月-日 时：分：秒的当前时间
        clock = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        # 获取格式为时：分：秒的当前时间
        employee_clock = time.strftime('%H:%M:%S', time.localtime())
        # 停车时间
        park_time = "--"
        # 把用户名和密码注册到数据库中
        # 连接数据库,此前在数据库中创建数据库
        db = pymysql.connect(host="localhost", user="root", password="qy020320", database="car_flask")
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # s记录随机选择车位
        s = random.randint(1, 80)
        # sql语句
        sql1 = "select in_time from plate where plate='" + str(f.filename) + "';"
        sql2 = "insert into plate values ('" + str(f.filename) + "','" + str(clock) + "',null," + str(s) + ");"
        sql3 = "delete from plate where plate='" + str(f.filename) + "';"
        sql4 = "select * from plate where driveway=" + str(s) + ";"
        sql5 = "select * from plate;"
        sql6 = "select * from question where plate='" + str(f.filename) + "';"
        sql7 = "select type from fast_pass where plate='" + str(f.filename) + "';"
        sql8 = "select start_time ,end_time from employee where plate='" + str(f.filename) + "';"
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
                                           file_path=file_path, warn=warn,plate=plate)
                else:
                    #车辆入库15分钟内免费
                    if fifteenmin>=park_time:
                        fee=0
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
                                           fee=fee, warn=warn,plate=plate)
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
                    return render_template("upload.html", time=clock, file_path=file_path, warn=warn,plate=plate)
                # 是救护车或者消防车
                elif len(is_fast_pass) == 1:
                    # 判断type
                    if is_fast_pass[0][0] == "救护车":
                        warn = "紧急情况，请救护车快速通过"
                    else:
                        warn = "紧急情况，请消防车快速通过"
                    return render_template("upload.html", time=clock, file_path=file_path, warn=warn,plate=plate)
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
                    return render_template("upload.html", time=clock, parking=park_time,
                                           file_path=file_path, site=site, warn=warn,plate=plate)
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
    sql1 = "select driveway,in_time from plate;"
    try:
        # 执行sql语句
        cursor.execute(sql1)
        results = cursor.fetchall()
        # number为车库中剩余车位数量
        number = 80 - len(results)
        session['1'] = 0
        session['2'] = 0
        session['3'] = 0
        session['4'] = 0
        session['5'] = 0
        session['6'] = 0
        session['7'] = 0
        session['8'] = 0
        session['9'] = 0
        session['10'] = 0
        session['11'] = 0
        session['12'] = 0
        session['13'] = 0
        session['14'] = 0
        session['15'] = 0
        session['16'] = 0
        session['17'] = 0
        session['18'] = 0
        session['19'] = 0
        session['20'] = 0
        session['21'] = 0
        session['22'] = 0
        session['23'] = 0
        session['24'] = 0
        session['25'] = 0
        session['26'] = 0
        session['27'] = 0
        session['28'] = 0
        session['29'] = 0
        session['30'] = 0
        session['31'] = 0
        session['32'] = 0
        session['33'] = 0
        session['34'] = 0
        session['35'] = 0
        session['36'] = 0
        session['37'] = 0
        session['38'] = 0
        session['39'] = 0
        session['40'] = 0
        session['41'] = 0
        session['42'] = 0
        session['43'] = 0
        session['44'] = 0
        session['45'] = 0
        session['46'] = 0
        session['47'] = 0
        session['48'] = 0
        session['49'] = 0
        session['50'] = 0
        session['51'] = 0
        session['52'] = 0
        session['53'] = 0
        session['54'] = 0
        session['55'] = 0
        session['56'] = 0
        session['57'] = 0
        session['58'] = 0
        session['59'] = 0
        session['60'] = 0
        session['61'] = 0
        session['62'] = 0
        session['63'] = 0
        session['64'] = 0
        session['65'] = 0
        session['66'] = 0
        session['67'] = 0
        session['68'] = 0
        session['69'] = 0
        session['70'] = 0
        session['71'] = 0
        session['72'] = 0
        session['73'] = 0
        session['74'] = 0
        session['75'] = 0
        session['76'] = 0
        session['77'] = 0
        session['78'] = 0
        session['79'] = 0
        session['80'] = 0
        # 历遍results，使对应车位时间存入对应session
        for i in results:
            if i[0] == 1:
                session['1'] = i[1]
            elif i[0] == 2:
                session['2'] = i[1]
            elif i[0] == 3:
                session['3'] = i[1]
            elif i[0] == 4:
                session['4'] = i[1]
            elif i[0] == 5:
                session['5'] = i[1]
            elif i[0] == 6:
                session['6'] = i[1]
            elif i[0] == 7:
                session['7'] = i[1]
            elif i[0] == 8:
                session['8'] = i[1]
            elif i[0] == 9:
                session['9'] = i[1]
            elif i[0] == 10:
                session['10'] = i[1]
            elif i[0] == 11:
                session['11'] = i[1]
            elif i[0] == 12:
                session['12'] = i[1]
            elif i[0] == 13:
                session['13'] = i[1]
            elif i[0] == 14:
                session['14'] = i[1]
            elif i[0] == 15:
                session['15'] = i[1]
            elif i[0] == 16:
                session['16'] = i[1]
            elif i[0] == 17:
                session['17'] = i[1]
            elif i[0] == 18:
                session['18'] = i[1]
            elif i[0] == 19:
                session['19'] = i[1]
            elif i[0] == 20:
                session['20'] = i[1]
            elif i[0] == 21:
                session['21'] = i[1]
            elif i[0] == 22:
                session['22'] = i[1]
            elif i[0] == 23:
                session['23'] = i[1]
            elif i[0] == 24:
                session['24'] = i[1]
            elif i[0] == 25:
                session['25'] = i[1]
            elif i[0] == 26:
                session['26'] = i[1]
            elif i[0] == 27:
                session['27'] = i[1]
            elif i[0] == 28:
                session['28'] = i[1]
            elif i[0] == 29:
                session['29'] = i[1]
            elif i[0] == 30:
                session['30'] = i[1]
            elif i[0] == 31:
                session['31'] = i[1]
            elif i[0] == 32:
                session['32'] = i[1]
            elif i[0] == 33:
                session['33'] = i[1]
            elif i[0] == 34:
                session['34'] = i[1]
            elif i[0] == 35:
                session['35'] = i[1]
            elif i[0] == 36:
                session['36'] = i[1]
            elif i[0] == 37:
                session['37'] = i[1]
            elif i[0] == 38:
                session['38'] = i[1]
            elif i[0] == 39:
                session['39'] = i[1]
            elif i[0] == 40:
                session['40'] = i[1]
            elif i[0] == 41:
                session['41'] = i[1]
            elif i[0] == 42:
                session['42'] = i[1]
            elif i[0] == 43:
                session['43'] = i[1]
            elif i[0] == 44:
                session['44'] = i[1]
            elif i[0] == 45:
                session['45'] = i[1]
            elif i[0] == 46:
                session['46'] = i[1]
            elif i[0] == 47:
                session['47'] = i[1]
            elif i[0] == 48:
                session['48'] = i[1]
            elif i[0] == 49:
                session['49'] = i[1]
            elif i[0] == 50:
                session['50'] = i[1]
            elif i[0] == 51:
                session['51'] = i[1]
            elif i[0] == 52:
                session['52'] = i[1]
            elif i[0] == 53:
                session['53'] = i[1]
            elif i[0] == 54:
                session['54'] = i[1]
            elif i[0] == 55:
                session['55'] = i[1]
            elif i[0] == 56:
                session['56'] = i[1]
            elif i[0] == 57:
                session['57'] = i[1]
            elif i[0] == 58:
                session['58'] = i[1]
            elif i[0] == 59:
                session['59'] = i[1]
            elif i[0] == 60:
                session['60'] = i[1]
            elif i[0] == 61:
                session['61'] = i[1]
            elif i[0] == 62:
                session['62'] = i[1]
            elif i[0] == 63:
                session['63'] = i[1]
            elif i[0] == 64:
                session['64'] = i[1]
            elif i[0] == 65:
                session['65'] = i[1]
            elif i[0] == 66:
                session['66'] = i[1]
            elif i[0] == 67:
                session['67'] = i[1]
            elif i[0] == 68:
                session['68'] = i[1]
            elif i[0] == 69:
                session['69'] = i[1]
            elif i[0] == 70:
                session['70'] = i[1]
            elif i[0] == 71:
                session['71'] = i[1]
            elif i[0] == 72:
                session['72'] = i[1]
            elif i[0] == 73:
                session['73'] = i[1]
            elif i[0] == 74:
                session['74'] = i[1]
            elif i[0] == 75:
                session['75'] = i[1]
            elif i[0] == 76:
                session['76'] = i[1]
            elif i[0] == 77:
                session['77'] = i[1]
            elif i[0] == 78:
                session['78'] = i[1]
            elif i[0] == 79:
                session['79'] = i[1]
            elif i[0] == 80:
                session['80'] = i[1]
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


@app.route('/video_upload',methods=['GET','POST'])
def video_upload():
    if request.method=='POST':
        char_all = detect('static/img/video/catch.jpg')
        plate = ''.join(char_all)
        clock = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        employee_clock = time.strftime('%H:%M:%S', time.localtime())
        park_time = "--"
        db = pymysql.connect(host="localhost", user="root", password="qy020320", database="car_flask")
        cursor = db.cursor()
        s = random.randint(1, 80)
        sql1 = "select in_time from plate where plate='catch.jpg';"
        sql2 = "insert into plate values ('catch.jpg','" + str(clock) + "',null," + str(s) + ");"
        sql3 = "delete from plate where plate='catch.jpg';"
        sql4 = "select * from plate where driveway=" + str(s) + ";"
        sql5 = "select * from plate;"
        sql6 = "select * from question where plate='catch.jpg';"
        sql7 = "select type from fast_pass where plate='catch.jpg';"
        sql8 = "select start_time ,end_time from employee where plate='catch.jpg';"
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
                                           plate=plate)
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
                    return render_template("video.html", time=clock, parking=park_time, fee=fee, warn=warn, plate=plate)
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
                    return render_template("video.html", time=clock, warn=warn, plate=plate)
                # 是救护车或者消防车
                elif len(is_fast_pass) == 1:
                    # 判断type
                    if is_fast_pass[0][0] == "救护车":
                        warn = "紧急情况，请救护车快速通过"
                    else:
                        warn = "紧急情况，请消防车快速通过"
                    return render_template("video.html", time=clock, warn=warn, plate=plate)
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
                                           plate=plate)
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
