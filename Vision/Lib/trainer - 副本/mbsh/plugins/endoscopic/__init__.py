import datetime
import os

from mbsh import g_temp_dir, g_utc_8
from mbsh.core.models import Merchant

__author__ = 'hill.hu'


def get_device_dir(device_id, tag, timestamp=None):
    merchant = Merchant.query.first()
    if timestamp:
        created_at = datetime.datetime.fromtimestamp(timestamp, g_utc_8)
    else:
        created_at = datetime.datetime.now()
    year_tag = created_at.strftime('%Y')
    month_tag = created_at.strftime('%m')
    day_tag = created_at.strftime('%d')

    path_join = os.path.join(g_temp_dir, merchant.merchant_id, tag, year_tag, month_tag, day_tag, device_id)
    if not os.path.exists(path_join):
        os.makedirs(path_join, exist_ok=True)
    return path_join


class CaseTarget:
    """
        检查部位，胃部或者肠部

    """

    def cal_score(self, report_params=None):
        """
        计算报告分数
        :return: paper
        """
        pass

    def cal_color(self, label, nbi_mode=False):
        """
        计算颜色
        :return: color
        """
        confidence = label.confidence
        if label.label_name == '息肉识别':
            return 255, 0, 0
        if confidence >= 90:
            if nbi_mode:
                return 0, 0, 255
            else:
                return 0, 0, 255
        return 0, 255, 0
