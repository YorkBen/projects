# from flask_table import Table, Col, DateCol, LinkCol

# class PatientList(Table):
#     name = Col('name', '患者姓名')
#     sex = Col('sex', '患者性别')
#     age = Col('age', '患者年龄')
#     date = DateCol('date', '就诊日期')
#     detail = LinkCol('detail', '详情', url_kwargs=dict(mrno='mrno'), anchor_attrs={'class': 'myclass'})
#
#
# class HospitalPatientList(Table):
#     no = Col('no', '床号')
#     mrno = Col('mrno', '住院号')
#     name = Col('name', '患者姓名')
#     sex = Col('sex', '患者性别')
#     age = Col('age', '患者年龄')
#     date = DateCol('date', '就诊日期')
#     detail = LinkCol('detail', '详情', url_kwargs=dict(mrno='mrno'), anchor_attrs={'class': 'myclass'})
#
#
# class ClinicPatientList(Table):
#     no = Col('no', '序号')
#     mrno = Col('mrno', '登记号')
#     name = Col('name', '患者姓名')
#     sex = Col('sex', '患者性别')
#     age = Col('age', '患者年龄')
#     date = DateCol('date', '就诊日期')
#     detail = LinkCol('detail', '详情', url_kwargs=dict(mrno='mrno'), anchor_attrs={'class': 'myclass'})
#
#
# class Patient(object):
#     def __init__(self, no, mrno, name, sex, age, date, detail):
#         self.no = no
#         self.mrno = mrno
#         self.name = name
#         self.sex = sex
#         self.age = age
#         self.date = date
#         self.detail = detail


# patients = [
#     Patient('1', '1321321213', '蒋欣', '女', '32', '2022-06-01', 'link'),
#     Patient('2', '1321321213', '蒋欣', '女', '32', '2022-06-01', 'link'),
#     Patient('3', '1321321213', '蒋欣', '女', '32', '2022-06-01', 'link'),
#     Patient('4', '1321321213', '蒋欣', '女', '32', '2022-06-01', 'link'),
# ]
# # Or, more likely, load items from your database with something like
# # items = ItemModel.query.all()
# # 从数据库加载
# tables = ClinicPatientList(patients)
# print(tables.items)
# print(tables.no)
# print(tables._cols)
# for item in dir(tables):
#     print(item)
#
# def get_table(depart_type):
#     print(ClinicPatientList(patients))
#     return None
    # if depart_type == 0:
    #     return ClinicPatientList(patients)
    # else:
    #     return HospitalPatientList(patients)
