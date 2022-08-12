from flask import Flask, render_template,request,session,redirect,url_for
import traceback
import pymysql

app = Flask(__name__)
app.secret_key = 'any random string'


@app.route('/')
def index():
    msg = "识别中国大陆各类机动车车牌信息，支持蓝牌、黄牌（单双行）、绿牌、大型新能源（黄绿）、领使馆车牌、警牌、" \
          "武警牌（单双行）、军牌（单双行）、港澳出入境车牌、农用车牌、民航车牌，并能同时识别图像中的多张车牌"
    return render_template("index.html", data=msg)

@app.route('/login')
def loginpage():
    return render_template("login.html")

@app.route('/loginProcess',methods=['POST','GET'])
def loginProcesspage():
    if request.method== 'POST':
        # 把用户名和密码注册到数据库中
        # 连接数据库,此前在数据库中创建数据库TESTDB
        db = pymysql.connect(host="localhost", user="root", password="qwertyjwt11", database="car_flask")
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # SQL 插入语句
        sql = "select * from user where email='" + str(request.form['exampleInputEmail2']) +\
              "' and username='" + str(request.form['exampleInputName2']) + "' and password='"+\
              str(request.form['exampleInputPassword3'])+"';"
        try:
            # 执行sql语句
            cursor.execute(sql)
            results = cursor.fetchall()
            print(len(results))
            if len(results) == 1:
                session['username']=request.form['exampleInputName2']
                return render_template('index.html')
            else:
                return '用户名或密码不正确'
            # 提交到数据库执行
            db.commit()
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

@app.route('/enrollProcess',methods=['POST','GET'])
def enrollProcesspage():
    if request.method=='POST':
        # 把用户名和密码注册到数据库中
        # 连接数据库,此前在数据库中创建数据库TESTDB
        db = pymysql.connect(host="localhost", user="root", password="qwertyjwt11", database="car_flask")
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # SQL 插入语句
        if request.form['exampleInputPassword1']!='' and request.form['exampleInputEmail1']!='' \
            and request.form['exampleInputName1']!='':
            if request.form['exampleInputPassword1'] == request.form['exampleInputPassword2']:
                sql = "insert into user values ('" + str(request.form['exampleInputEmail1']) + "', '" + \
                      str(request.form['exampleInputName1']) + "', '" + \
                      str(request.form['exampleInputPassword1']) + "');"
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

@app.route('/logoutProcess',methods=['POST','GET'])
def logoutProcesspage():
    if request.method== 'POST':
        # 把用户名和密码注册到数据库中
        # 连接数据库,此前在数据库中创建数据库
        db = pymysql.connect(host="localhost", user="root", password="qwertyjwt11", database="car_flask")
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # SQL 插入语句
        sql1 = "select * from user where email='" + str(request.form['exampleInputEmail3']) +\
              "' and username='" + str(request.form['exampleInputName3']) + "' and password='"+\
              str(request.form['exampleInputPassword4'])+"';"
        sql2 = "DELETE FROM user WHERE email = '"+str(request.form['exampleInputEmail3'])+"';"
        try:
            # 执行sql语句
            cursor.execute(sql1)
            results = cursor.fetchall()
            print(len(results))
            if len(results) == 1:
                session['username']=''
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

@app.route('/users')
def users():
    usersdata = "账户"
    return render_template("users.html", data=usersdata)


@app.route('/community')  # 增加一个product页面
def community():
    return render_template("community.html")


if __name__ == "__main__":
    print(app.url_map)
    app.run(host='0.0.0.0', debug=True)