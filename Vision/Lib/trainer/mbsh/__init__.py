import datetime
import logging
from logging.handlers import RotatingFileHandler
import os
import sys

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from config import config, basedir

g_utc_8 = datetime.timezone(datetime.timedelta(hours=8))
g_data_dir = os.path.join(basedir, '..', 'data')
g_temp_dir = r'd:\znyx\temp'
if os.path.exists("d:"):
    g_temp_dir = r'd:\znyx\temp'
else:
    g_temp_dir = r'e:\znyx\temp'
log_dir = os.path.join(basedir, '..', 'logs')

for k_dir in [g_data_dir, log_dir, g_temp_dir]:
    if not os.path.exists(k_dir):
        os.makedirs(k_dir, exist_ok=True)

print('use temp dir:' + g_temp_dir)


def init_logger():
    file_name = 'mbsh'

    for f in sys.argv:
        if 'py' in f:
            f = f.split("\\")[-1]
            f = f.split("/")[-1]
            file_name = f.split('.')[0]
    try:
        log_file_path = '%s/%s.log' % (log_dir, file_name)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
    except:
        pid = os.getpid()
        log_file_path = '%s/%s_%s.log' % (log_dir, file_name, pid)

    handler = RotatingFileHandler(log_file_path, maxBytes=1024 * 1024 * 100, backupCount=1, encoding='UTF-8')

    handler.setLevel(logging.DEBUG)
    logging_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
    handler.setFormatter(logging_format)
    logger = logging.getLogger()
    logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(logging_format)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('ini  logger file %s' % (os.path.abspath(log_file_path)))
    return logger


db = SQLAlchemy()
logger = init_logger()


def create_app(config_name):
    app = Flask(__name__)
    migrate = Migrate(app, db)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    logger.info("db url =%s" % app.config['SQLALCHEMY_DATABASE_URI'])
    db.init_app(app)

    return app
