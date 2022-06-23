import shutil

from mbsh.core.models import *
from trainer.trainer import Trainer, create_link

__author__ = 'gpu'

import unittest
import os


from mbsh import create_app, g_temp_dir


class TrainTestCase(unittest.TestCase):
    def setUp(self):
        app = create_app(os.getenv('FLASK_CONFIG') or 'default')
        app.app_context().push()
        self.app = app

    def tearDown(self):
        print('tear down')
        # os.unlink(mbsh.app.config['DATABASE'])

    def test_train(self):
        sm = SmallModel.query.get('内镜分类')
        trainer = Trainer(sm)


    def test_refresh_status(self):
        Trainer.refresh_status()

    def test_update_result(self):
        sm = SmallModel.query.get('是否合格')
        trainer = Trainer(sm)
        trainer.target_fold = 'E:/projects/znyx/data/job/38'
        trainer.fit_models(1)

    def test_crate_link(self):
        create_link(["E:/projects/znyx/data/imgs_valid/job/31/train/0/_1004_9814.jpeg"],
                    "E:/projects/znyx/data/imgs_valid/job/31/k_folds_5/1/train")

    def test_export_valid(self):
        sm = SmallModel.query.get('是否合格')

        trainer = Trainer(sm)
        trainer.target_fold = sm.target_fold
        trainer.export_predict(r'E:\projects\znyx\znyx-model\tests\res\crop\1', g_temp_dir + '/classes/test',
                               min_level=0.7)

    def test_export_ai(self):
        sm = SmallModel.query.get('早癌识别')

        trainer = Trainer(sm)
        trainer.target_fold = sm.target_fold
        trainer.export_predict(r'E:\projects\znyx\data\imgs_valid\res\1', g_temp_dir + '/classes/ai',
                               min_level=0.5)

    def test_export_26(self):
        sm = SmallModel.query.get('胃部分类26')

        trainer = Trainer(sm)
        trainer.target_fold = 'E:/projects/znyx/data/job/43'
        trainer.export_predict(r'E:\projects\znyx\data\temp\classes\valid\valid', g_temp_dir + '/classes/wei_26',
                               min_level=0.5)

    def test_create_out(self):
        sm = SmallModel.query.get('胃部分类26')

        trainer = Trainer(sm)
        trainer.target_fold = 'E:/projects/znyx/data/job/43'
        trainer.create_classes_fold(r'E:\projects\znyx\data\temp\classes\temp')


if __name__ == '__main__':
    unittest.main()
