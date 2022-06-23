from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField, PasswordField, HiddenField
from wtforms.validators import DataRequired

#定义登录表单
class LoginForm(FlaskForm):
    username = HiddenField('姓名')
    userid = StringField('工号', validators=[DataRequired()])
    password = PasswordField('密码', validators=[DataRequired()])
    depart = SelectField('科室', choices=[])
    submit = SubmitField('提交')


def login_validate(userid, password, depart):
    """
    验证用户账户、密码、部门
    """
    if userid == 'test' and password == '123' and '消化内科' in depart:
        return True
