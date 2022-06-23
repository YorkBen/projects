from flask import Flask, render_template, redirect, url_for, request, session, flash
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect
import requests
import json

from Lib.Login import *
# from Lib.PatientList import get_table as get_patient_table

app = Flask(__name__)
bootstrap = Bootstrap(app)
csrf = CSRFProtect(app)

app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['BOOTSTRAP_BOOTSWATCH_THEME'] = 'lumen'
app.config["SECRET_KEY"] = '79537d00f4834892986f09a100aa1edf'

@app.route('/')
def patient_list():
    if 'userid' not in session:
        return redirect(url_for('login'))
    else:
        if '门诊' in session['depart']:
            header = ['序号', '登记号', '患者姓名', '患者性别', '患者年龄', '就诊日期', '详情']
        else:
            header = ['床号', '住院号', '患者姓名', '患者性别', '患者年龄', '就诊日期', '详情']

        return render_template('patient_list.html', header=header, depart=session['depart'])

# 病患列表数据接口
@app.route('/patient_list_data')
def patient_list_data():
    if 'userid' not in session:
        return {
            'status_code': 500,
            'reason': 'Authorization Failed!',
            'text': ''
        }
    else:
        userid = session['userid']
        depart = session['depart']
        res = requests.get(url='http://127.0.0.1:5001/getPatientList', params={'userid': userid, 'depart': depart})
        return {
            'status_code': res.status_code,
            'reason': res.reason,
            'text': json.loads(res.text)
        }

@app.route('/patient/<int:pid>')
def patient_detail(pid):
    if 'userid' not in session:
        return redirect(url_for('login'))
    else:
        return render_template('patient_detail.html', pid=pid)

# 病患页面数据请求接口
@app.route('/patient_info/<int:pid>')
def patient_info(pid):
    if 'userid' not in session:
        return {
            'status_code': 500,
            'reason': 'Authorization Failed!',
            'text': ''
        }
    else:
        res = requests.get(url='http://127.0.0.1:5001/patient', params={'pid': pid})
        return {
            'status_code': res.status_code,
            'reason': res.reason,
            'text': json.loads(res.text)
        }

@app.route('/diease/<pidd>')
def diease_detail(pidd):
    if 'userid' not in session:
        return redirect(url_for('login'))
    else:
        arr = pidd.split('_')
        pid, diease = arr[0], arr[1]
        return render_template('disease_detail.html', depart_type=1, pid=pid, diease=diease)


@app.route('/login', methods = ['GET', 'POST'])
def login():
    session.pop('_flashes', None)
    if request.method == 'POST':
        # username在前端请求信息科回传
        username = request.form['username']
        userid = request.form['userid']
        password = request.form['password']
        depart = request.form['depart']

        # 登录验证
        if login_validate(userid, password, depart):
            flash('登录成功！', 'success')
            session['userid'] = userid
            session['username'] = username
            session['depart'] = depart
            return redirect(url_for('patient_list'))
        else:
            flash('账号或密码错误！', 'warning')
            form = LoginForm()
            # form.userid = userid
            # form.depart = depart
            return render_template('login.html', form=form)
    else:
        form = LoginForm()
        return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    session.pop('userid', None)
    session.pop('username', None)
    session.pop('depart', None)
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
