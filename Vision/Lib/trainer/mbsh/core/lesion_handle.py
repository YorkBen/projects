import base64
from scipy import signal
from io import BytesIO
import cv2
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math

"""处理病灶区域"""


def get_base64(val):
    """转换base64"""
    base64_data = base64.b64encode(val)
    s = base64_data.decode()
    res = 'data:image/jpeg;base64,%s' % s
    return res


def smooth_curve(x, y, n=12):
    """拟合曲线(默认用12次多项式拟合)"""
    f1 = np.polyfit(x, y, n)
    p1 = np.poly1d(f1)
    yvals = p1(x)
    return x, yvals


def format_num(x, pos):
    """plt坐标格式"""
    return '$%.0f$' % x


formatter = FuncFormatter(format_num)
COLORS = {'blue': '#000079', 'lightblue': '#2894FF', 'red': '#FF0000', 'black': '#000000', 'gray': '#7B7B7B'}


def init_lesion_res():
    """初始化类变量"""
    SmRect.status = False
    BigRect.status = False
    BigRect.lock_rect = None
    BigRect.is_ini = False
    BigRect.shape = None
    ImageStable.img_stable = None
    ImageStable.is_ini = False


class BigRect:
    lock_rect = None
    shape = None
    status = False
    is_ini = False

    def __init__(self, ctrs):
        self.ctrs = ctrs  # 病灶轮廓
        self.big_rect = None  # 病灶外切矩形框
        self.sm_rects = None  # 病灶分割矩形框
        self.cmp_rect = None  # 正常组织矩形框
        self.fit_rect = None
        self.fin_res = {}

    def init_rect(self):
        """初始化对象"""
        self.big_rect = self.get_big_rect()
        self.sm_rects = self.get_sm_rects()
        self.cmp_rect = self.get_cmp_rect()
        self.fit_rect = self.get_fit_rect()

        # 计算有效性
        for sm_rect in self.sm_rects:
            sm_rect.cal_vid(self.ctrs)
        # 添加标志位
        BigRect.is_ini = True

    def deal_bbox(self, ai_ctrs):
        """病灶轮廓外切矩形"""
        return cv2.boundingRect(ai_ctrs[0])

    def deal_ctrs(self, ai_ctrs):
        """转换病灶轮廓"""
        result_list = []
        if not ai_ctrs:
            result_list = []
        else:
            ai_ctrs = ai_ctrs[0].tolist()
            if len(ai_ctrs) > 0:
                for items in ai_ctrs:
                    result_list.append({"x": items[0][0], "y": items[0][1]})
        return result_list

    def deal_rects(self, big_rect, shape=(4, 4)):
        """病灶轮廓外切矩形分割成小矩形，默认4*4"""
        x, y, w, h = big_rect
        if not BigRect.shape:
            BigRect.shape = shape
        else:
            r, c = BigRect.shape
        x_offset = math.ceil(w / c)
        y_offset = math.ceil(h / r)
        rects = []
        for i in range(int(r)):
            for j in range(int(c)):
                pos = (x + x_offset * j, y + y_offset * i, x_offset, y_offset)
                number = i * c + j + 1
                sm_rect = SmRect(number, pos)
                rects.append(sm_rect)
        return rects

    def get_big_rect(self):
        """获取轮廓外切矩形"""
        big_rect = cv2.boundingRect(self.ctrs[0])
        return big_rect

    def get_sm_rects(self):
        """获取轮廓外切矩形切分后的小矩形"""
        big_rect = self.get_big_rect()
        sm_rects = self.deal_rects(big_rect)
        return sm_rects

    def get_cmp_rect(self, size=15):
        """添加正常区域对比像素块（20*20）"""
        x, y, w, h = self.big_rect
        pos = [x - size, y - size, size, size]
        # pos = [x+w-80, y-5, size, size]
        cmp_rect = SmRect(0, pos)
        flag = True if min(pos) > 0 else False
        return cmp_rect

    def get_fit_rect(self):
        x, y, w, h = self.big_rect
        r, c = BigRect.shape
        x_offset = math.ceil(w / c)
        y_offset = math.ceil(h / r)
        w1, h1 = x_offset * c, y_offset * r
        fit_rect = [x, y, w1, h1]
        return fit_rect

    def get_vid_sm_rects(self):
        """获取所有有效的重叠的小矩形"""
        vid_sm_rects = []
        for sm_rect in self.sm_rects:
            if sm_rect.is_vid:
                vid_sm_rects.append(sm_rect)
        return vid_sm_rects

    def get_rect_means(self):
        """获取病灶区平均灰度值"""
        vid_sm_rects = self.get_vid_sm_rects()
        means = np.zeros(len(vid_sm_rects[0].means))
        for sm_rect in vid_sm_rects:
            means += np.array(sm_rect.means)
        means = means / len(vid_sm_rects)
        means = means.tolist()
        return means

    def cal_sm_rects_mean(self, img):
        """计算发送过来帧的所有分割块的平均灰度值（病灶区域和正常对比区域）"""
        img_gray = None
        if len(img.shape) == 3 or len(img.shape) == 4:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        for sm_rect in self.sm_rects:
            sm_rect.cal_sm_rect_mean(img_gray)
        self.cmp_rect.cal_sm_rect_mean(img_gray)

    def find_peak(self, y):
        """求20-30s极大值,如果没有的话就取30-40s极大值,如果还没有就取40-50s极大值
           否则取25-30s的最大值"""
        y = np.array(y)
        peak = [None, None]
        peak_x = signal.argrelextrema(y, np.greater)
        peak_y = y[peak_x]

        if len(peak_x[0]) > 0:
            index = np.where((peak_x[0] >= 20) & (peak_x[0] <= 30))
            if len(index[0]) == 0:
                index = np.where((peak_x[0] >= 30) & (peak_x[0] <= 40))
            if len(index[0]) == 0:
                index = np.where((peak_x[0] >= 40) & (peak_x[0] <= 50))
            if len(index[0]) > 0:
                peak_x = int(np.mean(index))
                peak_y = np.mean(peak_y[index])
                peak = [peak_x, peak_y]

        return peak

    def choose_color(self, c_peak, t_peak):
        """
        根据正常区域峰值和病灶区域峰值对比计算颜色和标签
        :param c_peak: 正常区域峰值
        :param t_peak: 病灶区域峰值
        :return:
        """
        color = None
        label = None
        if c_peak[1]:
            if t_peak[1]:
                if t_peak[1] - c_peak[1] > c_peak[1] * 0.1:
                    color = 'blue'
                elif c_peak[1] * 0.1 >= t_peak[1] - c_peak[1] > -c_peak[1] * 0.1:
                    color = 'lightblue'
                elif -c_peak[1] * 0.85 < t_peak[1] - c_peak[1] < -c_peak[1] * 0.1:
                    color = 'red'
                elif t_peak[1] - c_peak[1] < -c_peak[1] * 0.85:
                    color = 'black'
                else:
                    pass
                # 显示百分比标签
                percent = ((t_peak[1] - c_peak[1]) / abs(c_peak[1])) * 100
                if percent > 0:
                    label = '+{:.2f}%'.format(percent)
                else:
                    label = '{:.2f}%'.format(percent)
            else:
                color = 'black'
        else:
            color = 'gray'
        color = COLORS.get(color, None)
        return color, label

    def pre_means(self, means, k=10):
        """将所有means减去背景值,默认为前10张均值"""
        bg_mean = sum(means[:k]) / len(means[:k])
        means = [rect_mean - bg_mean for rect_mean in means]
        return means

    def cal_rects(self):
        """计算所有矩阵块的曲线图以及指标"""

        # 每次采图的病灶区域平均灰度值
        rect_means = self.get_rect_means()
        rect_means = self.pre_means(rect_means)
        # 每次采图的正常区域平均灰度值
        cmp_rect_means = self.cmp_rect.means
        cmp_rect_means = self.pre_means(cmp_rect_means)

        # 绘制病灶与非病灶区域的平均灰度变化对比曲线(整个病灶区)
        lengths = [i for i in range(len(rect_means))]
        _X, c_Y, t_Y, = lengths, cmp_rect_means, rect_means
        y_lim = max(c_Y + t_Y)
        plt.figure(figsize=(10, 5))
        plt.style.use('dark_background')
        f, ax = plt.subplots(1, 1)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_yticks(np.linspace(-y_lim * 3, y_lim * 3, 8))
        ax.tick_params(axis='both', which='major', labelsize=20)
        # 正常区域
        _x, c_y = smooth_curve(_X, c_Y)
        ax.plot(_x, c_y, label="Normal", color='green')
        c_peak = self.find_peak(np.array(c_y))
        # 病灶区域
        _x, t_y = smooth_curve(_X, t_Y)
        ax.plot(_x, t_y, label="Abnormal", color='red')
        t_peak = self.find_peak(np.array(t_y))
        # 添加标签
        color, label = self.choose_color(c_peak, t_peak)
        if label:
            if label.startswith('+'):
                ax.text(int(len(_x) / 4), y_lim * 2, label, fontsize=20)
            else:
                ax.text(int(len(_x) / 4), -y_lim * 2, label, fontsize=20)

        plt.legend(loc='lower right')
        curve = BytesIO()
        plt.savefig(curve, dpi=500, bbox_inches='tight', format='png')
        curve_img = get_base64(curve.getvalue())

        # 绘制病灶与非病灶区域的平均灰度变化对比曲线(分割病灶区)
        shape = BigRect.shape
        plt.figure(figsize=(10, 10))
        plt.style.use('dark_background')
        sm_rects = self.get_vid_sm_rects()
        for sm_rect in sm_rects:
            sm_bg_mean = sum(sm_rect.means[:10]) / len(sm_rect.means[:10])
            sm_rect_means = [mean - sm_bg_mean for mean in sm_rect.means]
            X = [i for i in range(len(sm_rect_means))]
            Y = sm_rect_means
            new_x, new_y = smooth_curve(X, Y)
            if sm_rect.is_vid:
                sub_plt = plt.subplot(shape[0], shape[1], sm_rect.number)
                sub_plt.yaxis.set_major_formatter(formatter)
                sub_plt.set_ylim(-y_lim, y_lim)
                sub_plt.set_yticks(np.linspace(-y_lim * 3, y_lim * 3, 8))
                sub_plt.plot(_x, c_y, label="Normal", color='green')
                sub_plt.plot(new_x, new_y, label="Abnormal", color='red')
                sub_plt.tick_params(axis='both', which='major', labelsize=10)
                t_peak = self.find_peak(np.array(new_y))
                # 添加宫格颜色属性以及标签
                color, label = self.choose_color(c_peak, t_peak)
                if color:
                    sm_rect.color = color
                if label:
                    if label.startswith('+'):
                        sub_plt.text(int(len(_x) / 4), y_lim * 2, label, fontsize=15)
                    else:
                        sub_plt.text(int(len(_x) / 4), -y_lim * 2, label, fontsize=15)

        sm_curve = BytesIO()
        plt.savefig(sm_curve, dpi=500, bbox_inches='tight', format='png')
        sm_curve_img = get_base64(sm_curve.getvalue())

        BigRect.status = True
        self.fin_res.update({'curve': curve_img, 'sm_curve': sm_curve_img})

    def model_2_json(self):
        return {"sm_rects": self.sm_rects, 'fin_res': self.fin_res,
                "ctrs": self.deal_ctrs(self.ctrs),
                "big_rect": self.big_rect, "fit_rect": self.fit_rect, 'cmp_rect': self.cmp_rect,
                'is_ini': BigRect.is_ini}


class SmRect:
    status = False

    def __init__(self, number, pos):
        self.number = number
        self.pos = pos
        self.means = []
        self.avg_mean = None
        self.is_vid = False
        self.color = None

    def get_trans_pos(self):
        """将x, y, w, h数值转换为四个顶角坐标"""
        x, y, w, h = self.pos
        p1 = (x, y)
        p2 = (x + w, y)
        p3 = (x, y + h)
        p4 = (x + w, y + h)
        trans_pos = [p1, p2, p3, p4]
        return trans_pos

    def pre_trans_pos(self, k=3):
        """判断矩形与轮廓相交时先预处理"""
        trans_pos = self.get_trans_pos()
        p1 = (trans_pos[0][0] + k, trans_pos[0][1] + k)
        p2 = (trans_pos[1][0] - k, trans_pos[1][1] + k)
        p3 = (trans_pos[2][0] + k, trans_pos[2][1] - k)
        p4 = (trans_pos[3][0] - k, trans_pos[3][1] - k)
        pre_tran_pos = [p1, p2, p3, p4]
        return pre_tran_pos

    def cal_vid(self, ctrs):
        """计算分割小矩形是否与病灶轮廓右交集"""
        pre_trans_pos = self.pre_trans_pos()
        for p in pre_trans_pos:
            dist = cv2.pointPolygonTest(ctrs[0], p, False)
            if dist > 0:
                self.is_vid = True
                break
        else:
            self.is_vid = False

    def cal_sm_rect_mean(self, img_gray):
        """依次计算每个小矩阵块的平均灰度值"""
        x, y, w, h = self.pos
        offset = img_gray.shape[1] // 2
        roi = img_gray[y:y + h, x + offset:x + w + offset]
        mean = np.mean(roi)
        self.means.append(mean)
        self.avg_mean = sum(self.means) / len(self.means)

    def model_2_json(self):
        return {"number": self.number, "pos": self.pos, "is_vid": self.is_vid, 'color': self.color}


class ImageStable:
    SMOOTHING_RADIUS = 50
    img_stable = None
    is_ini = False

    def __init__(self, img, rect):
        self.first_pts = self.ctrs_2_corners(rect)
        self.pre_gray = self.image_2_gray(img)
        self.transforms = []
        ImageStable.is_ini = True

    def movingAverage(self, curve, radius):
        window_size = 2 * radius + 1
        # Define the filter
        f = np.ones(window_size) / window_size
        # Add padding to the boundaries
        curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
        # Apply convolution
        curve_smoothed = np.convolve(curve_pad, f, mode='same')
        # Remove padding
        curve_smoothed = curve_smoothed[radius:-radius]
        # return smoothed curve
        return curve_smoothed

    def smooth(self, trajectory):
        smoothed_trajectory = np.copy(trajectory)
        # Filter the x, y and angle curves
        for i in range(3):
            smoothed_trajectory[:, i] = self.movingAverage(
                trajectory[:, i], radius=ImageStable.SMOOTHING_RADIUS)
        return smoothed_trajectory

    def fixBorder(self, frame):
        s = frame.shape
        # Scale the image 4% without moving the center
        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame

    def ctrs_2_corners(self, rect):
        # 将病灶区域矩形加入角点
        ctrs_pts = rect.ctrs[0]
        # 将正常区域矩形加入角点
        cmp_rect = rect.get_cmp_rect()
        trans_pos = cmp_rect.get_trans_pos()
        rect_pts = np.expand_dims(np.array(trans_pos), 1)
        pts = np.concatenate((ctrs_pts, rect_pts), 0).astype(np.float32)
        return pts

    def image_2_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def get_transform(self, curr):
        curr_gray = self.image_2_gray(curr)
        prev_gray = self.pre_gray
        prev_pts = self.first_pts
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None)
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)
        # Extract traslation
        dx = m[0, 2]
        dy = m[1, 2]
        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])
        transform = [dx, dy, da]
        self.transforms.append(transform)
        self.pre_gray = curr_gray
        return transform

    def get_trajectory(self, curr):
        self.get_transform(curr)
        transforms = np.array(self.transforms, dtype=np.float32)
        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = self.smooth(trajectory)
        difference = smoothed_trajectory - trajectory
        transforms_smooth = transforms + difference
        return transforms_smooth

    def start_stable(self, curr):
        # Extract transformations from the new transformation array
        left_curr = curr[:, 0:int(curr.shape[1] / 2)]
        right_curr = curr[:, int(curr.shape[1] / 2):]
        transforms_smooth = self.get_trajectory(left_curr.copy())
        dx = transforms_smooth[-1, 0]
        dy = transforms_smooth[-1, 1]
        da = transforms_smooth[-1, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        h, w = self.pre_gray.shape
        right_frame_stabilized = cv2.warpAffine(right_curr, m, (w, h))

        # Fix border artifacts  最后稳定的图像结果
        right_frame_stabilized = self.fixBorder(right_frame_stabilized)

        frame_stabilized = np.hstack([left_curr, right_frame_stabilized])
        return frame_stabilized
