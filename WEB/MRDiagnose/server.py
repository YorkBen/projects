from flask import Flask, render_template, redirect, url_for, request, session, flash
from flask_wtf import CSRFProtect
from flask_cors import cross_origin
import json

app = Flask(__name__)
csrf = CSRFProtect(app)

app.config["SECRET_KEY"] = '79537d00f4834892986f09a100aa1edf'

def assemble_keys(mrdata):
    if isinstance(mrdata, dict):
        keys = list(mrdata.keys())
        mrdata['keys'] = keys
        for key in keys:
            mrdata[key] = assemble_keys(mrdata[key])
    elif isinstance(mrdata, list):
        for mrrecord in mrdata:
            mrrecord = assemble_keys(mrrecord)

    return mrdata


mrdata = {}
with open('mrdata_sample.json') as f:
    mrdata = json.load(f)
    mrdata = assemble_keys(mrdata)

@app.route('/getPatientList')
@cross_origin()
def patient_list():
    depart = request.args.get('depart')
    print(depart)
    data = []
    for i in range(1, 21):
        data.append([i, '1321321213', '蒋欣%s' % i, '女', '32', '2022-06-01', 'link'])

    return json.dumps(data, ensure_ascii=False)


@app.route('/patient')
def patient_info():
    pid = request.args.get('pid')
    # return json.dumps(data, ensure_ascii=False)
    return json.dumps(mrdata, ensure_ascii=False)

@app.route('/diease/<int:did>')
def diease_detail(did):
    if 'userid' not in session:
        return redirect(url_for('login'))
    else:
        if '门诊' in session['depart']:
            return render_template('diease_detail.html', depart_type=0)
        else:
            return render_template('diease_detail.html', depart_type=1)


@app.route('/login', methods = ['GET', 'POST'])
def login():
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
            form.userid = userid
            form.depart = depart
            return render_template('login.html', form=form)
    else:
        form = LoginForm()
        return render_template('login.html', form=form)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5001", debug=True)
