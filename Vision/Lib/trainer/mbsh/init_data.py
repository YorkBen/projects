import os
from flask_script import Command
import shutil
from flask_migrate import upgrade, migrate, init


class InitData(Command):
    def run(self):
        from mbsh import db

        from mbsh.core.models import Role
        from mbsh.core.models import User, SmallModel, to_json

        self.migrate_db(db)
        session = db.session

        u = User.query.filter(User.username == 'admin').first()
        print(u)
        if not u:
            Role.insert_roles()
            u = User(username='admin', password='whu', email='154900333@qq.com')
            session.add(u)
            session.commit()

        print(User.query.all())
        db_models = SmallModel.query.all()
        for model in db_models:
            session.delete(model)

        models = [
            SmallModel('清洁度评分', "../data/imgs_clean",
                       ['0分', '1分', '2分', '3分'], number=3, case_type=2)

            , SmallModel('内镜分类', "../data/imgs_parts",
                         ['不合格', '食管', '正镜贲门', '胃窦', '十二指肠球部', '十二指肠降部', '正镜胃体下部', '正镜胃体中上部', u'倒镜胃底', '倒镜胃体中上部',
                          '倒镜胃角', 'NBI放大', '口咽部', '无法判断', '图像模糊', '肛门口', '回盲瓣', '肛管', '肠粘膜', '其它肠', '伪回盲瓣', '阑尾口',
                          '伪阑尾口', '手术中']
                         , number=3, case_type=0)

            , SmallModel('胃部分类27', "../data/imgs_wei_27",
                         ['无法判断', '食管', '贲门', '胃窦#大弯', '胃窦#后壁', '胃窦#前壁', '胃窦#小弯', '十二指肠球部', '十二指肠降部',
                          '正镜胃体下部#大弯', '正镜胃体下部#后壁', '正镜胃体下部#前壁', '正镜胃体下部#小弯',
                          '正镜胃体中上部#大弯', '正镜胃体中上部#后壁', '正镜胃体中上部#前壁', '正镜胃体中上部#小弯',
                          '倒镜胃底#大弯', '倒镜胃底#后壁', '倒镜胃底#前壁', '倒镜胃底#小弯',
                          '倒镜胃体中上部#后壁', '倒镜胃体中上部#前壁', '倒镜胃体中上部#小弯',
                          '倒镜胃角#后壁', '倒镜胃角#前壁', '倒镜胃角#小弯'],
                         number=3, case_type=1)

            , SmallModel('胃部分类', "../data/imgs_wei",
                         ['无法判断', '食管', '正镜贲门', '胃窦', '十二指肠球部', '十二指肠降部', '正镜胃体下部', '正镜胃体中上部', u'倒镜胃底', '倒镜胃体中上部',
                          '倒镜胃角'], number=3, case_type=1)

            , SmallModel('肠部位分类', "../data/imgs_chang",
                         ['吸水', '贴壁',  '清晰'], number=3, case_type=2)

            , SmallModel('溃疡分类', "../data/imgs_u·   lcer",
                         ['正常', 'S型', 'H型', 'A型', '癌变'], number=3, case_type=1)

            , SmallModel('体内外分类', "../data/imgs_intestine",
                         ['in vitro', 'in vivo', 'unqualified'], number=3, case_type=1)
            # , SmallModel('溃疡分类', "../data/imgs_ulcer",
            #              ['正常', 'A1型', 'A2型', 'H1型', 'H2型', 'S1型', 'S2型'], number=3, case_type=1)

            , SmallModel('早癌识别', "../data/imgs_ai", ['正常', '可见癌变'], number=4, case_type=1)
            , SmallModel('早癌识别NBI', "../data/imgs_nbi_ca", ['正常', '可见癌变'], number=4, case_type=1)
            , SmallModel('息肉识别', "../data/imgs_xi", ['正常', '可见息肉'], number=3, case_type=0)

            , SmallModel('息肉oct', "../data/imgs_xi_oct", ['正常', '腺瘤'], number=1)

            , SmallModel('肠部分类4', "../data/imgs_chang_4",
                         ['无法判断', '右半结肠', '横结肠', '左半结肠', '恶性病变', '食管', '正镜贲门', '胃体', '十二指肠球部'],
                         number=5, case_type=2)

            , SmallModel('肠癌识别', "../data/imgs_intestine_ca", ['正常', '可见癌变'], number=5, case_type=2)

            , SmallModel('回盲瓣识别', "../data/imgs_mang", ['否', '是'], number=3, case_type=2)

            , SmallModel('息肉分型', "../data/imgs_nice", ['Ⅰ型', 'Ⅱ型', 'Ⅲ型'], number=3, case_type=2)
            # '胆汁反流性', '糜烂性',
            , SmallModel('胃炎分型', "../data/imgs_gastritis", ['浅表性', '萎缩性', '正常'], number=3, case_type=2)

        ]
        for sm in models:
            session.merge(sm)
        session.commit()

        models = SmallModel.query.all()
        print(to_json(models))

    def migrate_db(self, db):
        from mbsh import logger, g_data_dir
        db_migrate = os.path.join(g_data_dir, "migrations")
        try:
            db.create_all()
        except:
            pass
        try:
            init(directory=db_migrate)

        except:
            logger.exception('init db  error')
            pass
        try:
            migrate(directory=db_migrate)
            pass

        except:
            logger.exception('migrate db error')
        try:
            upgrade(directory=db_migrate)
            pass
        except:
            logger.exception('upgrade db  error')
            pass

        shutil.rmtree(db_migrate, ignore_errors=True)

    print('success')
