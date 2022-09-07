__author__ = 'hill'


class BasePlugin:
    """
    不同的业务类型实现
    """

    def start_work(self):
        pass

    def upload_datapoint(self, datapoint):
        pass

    def latest_report(self, device_id):
        pass

    def fetch_reports(self, device_id):
        pass

    def fetch_top_reports(self):
        pass
