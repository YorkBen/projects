__author__ = 'Hill Hu'
import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'this is a secret string'
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    MODEL_API_HOST = '127.0.0.1'
    FLASKY_MAIL_SUBJECT_PREFIX = '[Flasky]'
    FLASKY_MAIL_SENDER = '3122896970@qq.com'
    FLASKY_ADMIN = os.environ.get('FLASKY_ADMIN')
    TEMPLATES_AUTO_RELOAD = True
    MBSH_VIDEO_DIR = r'F:\medical\video'

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = False
    MODEL_API_HOST = '127.0.0.1'

    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
                              'postgresql://znyx:hzk@192.168.0.110:5432/znyx_db'


class TestingConfig(Config):
    TESTING = True
    MODEL_API_HOST = '127.0.0.1'
    MBSH_VIDEO_DIR = r'e:\projects\video'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
                              'postgresql://znyx:hzk@192.168.56.153:5432/znyx_db'


class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
                              'sqlite:///' + os.path.join(basedir,'../data' ,'production.db')


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': ProductionConfig
}
