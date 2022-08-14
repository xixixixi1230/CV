import os

from flask import Flask, render_template, request, redirect, url_for, session
import traceback
import pymysql
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'any random string'
app.config['UPLOAD_FOLDER'] = 'static/img/upload/'

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader',methods=['GET','POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        return render_template('upload.html',isload=True)
    else:
        return render_template('upload.html',isload=False)

@app.route('/')
def index():
    msg = "识别中国大陆各类机动车车牌信息，支持蓝牌、绿牌。"
    return render_template("index.html", data=msg)


@app.route('/login')
def loginpage():
    return render_template("login.html")


@app.route('/loginProcess', methods=['POST', 'GET'])
def loginProcesspage():
    if request.method == 'POST':
        # 把用户名和密码注册到数据库中
        # 连接数据库,此前在数据库中创建数据库TESTDB
        db = pymysql.connect(host="localhost", user="root", password="qwertyjwt11", database="car_flask")
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # SQL 插入语句
        sql1 = "select * from user where email='" + str(request.form['exampleInputEmail2']) + \
               "' and username='" + str(request.form['exampleInputName2']) + "' and password='" + \
               str(request.form['exampleInputPassword3']) + "';"
        sql2 = "update user set login = True where email='" + str(request.form['exampleInputEmail2']) + \
               "' and username='" + str(request.form['exampleInputName2']) + "' and password='" + \
               str(request.form['exampleInputPassword3']) + "';"
        try:
            # 执行sql语句
            cursor.execute(sql1)
            results = cursor.fetchall()
            print(len(results))
            if len(results) == 1:
                session['username'] = request.form['exampleInputName2']
                cursor.execute(sql2)
                # 提交到数据库执行
                db.commit()
                return render_template('index.html')
            else:
                return '用户名或密码不正确'
        except:
            # 如果发生错误则回滚
            traceback.print_exc()
            db.rollback()
            return '登陆失败'
        # 关闭数据库连接
        db.close()


@app.route('/enroll')
def enrollpage():
    return render_template("enroll.html")


@app.route('/enrollProcess', methods=['POST', 'GET'])
def enrollProcesspage():
    if request.method == 'POST':
        # 把用户名和密码注册到数据库中
        # 连接数据库,此前在数据库中创建数据库TESTDB
        db = pymysql.connect(host="localhost", user="root", password="qwertyjwt11", database="car_flask")
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # SQL 插入语句
        if request.form['exampleInputPassword1'] != '' and request.form['exampleInputEmail1'] != '' \
                and request.form['exampleInputName1'] != '':
            if request.form['exampleInputPassword1'] == request.form['exampleInputPassword2']:
                sql = "insert into user values ('" + str(request.form['exampleInputEmail1']) + "', '" + \
                      str(request.form['exampleInputName1']) + "', '" + \
                      str(request.form['exampleInputPassword1']) + "', False);"
                try:
                    # 执行sql语句
                    cursor.execute(sql)
                    # 提交到数据库执行
                    db.commit()
                    # 注册成功之后跳转到登录页面
                    return render_template('login.html')
                except:
                    # 抛出错误信息
                    traceback.print_exc()
                    # 如果发生错误则回滚
                    db.rollback()
                    return '注册失败'
            else:
                return '密码输入错误，请重新注册'
        else:
            return '请补充完整表格'
        # 关闭数据库连接
        db.close()


@app.route('/logout')
def logoutpage():
    return render_template("logout.html")


@app.route('/logoutProcess', methods=['POST', 'GET'])
def logoutProcesspage():
    if request.method == 'POST':
        # 把用户名和密码注册到数据库中
        # 连接数据库,此前在数据库中创建数据库
        db = pymysql.connect(host="localhost", user="root", password="qwertyjwt11", database="car_flask")
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # SQL 插入语句
        sql1 = "select * from user where email='" + str(request.form['exampleInputEmail3']) + \
               "' and username='" + str(request.form['exampleInputName3']) + "' and password='" + \
               str(request.form['exampleInputPassword4']) + "'and login=True;"
        sql2 = "DELETE FROM user WHERE email = '" + str(request.form['exampleInputEmail3']) + "';"
        try:
            # 执行sql语句
            cursor.execute(sql1)
            results = cursor.fetchall()
            print(len(results))
            if len(results) == 1:
                session.clear()
                cursor.execute(sql2)
                db.commit()
                return render_template('index.html')
            else:
                return '用户名或密码不正确'
        except:
            # 如果发生错误则回滚
            traceback.print_exc()
            db.rollback()
            return '注销失败'
        # 关闭数据库连接
        db.close()


@app.route('/logoff')
def logoffpage():
    session.clear()
    return render_template("index.html")


@app.route('/users')
def users():
    usersdata = "账户"
    return render_template("users.html", data=usersdata)


@app.route('/park')  # 增加一个product页面
def communitypage():
    return render_template("park.html")


@app.route('/fee_rules')
def fee_rulespage():
    return render_template("fee_rules.html")


if __name__ == "__main__":
    print(app.url_map)
    app.run(debug=True)
