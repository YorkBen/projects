import base64
import os
import random
import time

__author__ = 'hill'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc, roc_curve
import itertools
from mbsh import logger, basedir
import cv2

transparent_img = cv2.imread(basedir + r'/tests\res\transparent.png', -1)


class Plot:
    @staticmethod
    def transparent_overlay(src, overlay):

        h, w, _ = overlay.shape  # Size of foreground

        rows, cols, _ = src.shape  # Size of background Image
        pos = (random.randint(rows // 8, rows // 3), random.randint(cols // 8, cols // 3))
        y, x = pos[0], pos[1]  # Position of foreground/overlay image

        alpha_rate = random.randint(60, 100) * 0.01
        # loop over all pixels and apply the blending equation
        for i in range(h):
            for j in range(w):
                if x + i >= rows or y + j >= cols:
                    continue
                alpha = float(overlay[i][j][3] / 255.0) * alpha_rate  # read the alpha channel
                src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]

        return src

    @staticmethod
    def plot_cam(img, cam, out_put_dir, size=224, **img_param):
        cam = cv2.resize(cam, (size, size))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam <= 0.6)] = 0
        opacity = img_param.get("opacity", 0.4)
        weight_opacity = 0.2
        out = cv2.addWeighted(img, opacity, heatmap, weight_opacity, 0)

        # blank_img = np.zeros((224, 224, 4))
        heatmap_a = np.concatenate((heatmap, np.zeros((size, size, 1))), axis=2)
        bbox_out = heatmap_a.copy()
        bbox_out[np.where(cam > 0.6)] = 255

        bbox_out = cv2.addWeighted(bbox_out, 0.2, heatmap_a, 0.8, 0)
        retval, buffer = cv2.imencode('.png', bbox_out)
        cam_img = str(base64.b64encode(buffer)).replace("b'", 'data:image/png;base64,').replace("'", "")
        file_path = Plot.save_img(out, out_put_dir)
        return cam_img, file_path

    @staticmethod
    def plot_cam_bbox(src_img, rects, label, out_put_dir, color=(0, 255, 0), size=224, line_weight=1):
        """
        color 为bgr的模式
        """
        img = src_img.copy()
        bbox_img = cv2.resize(img, (size, size))
        scale = size / 224
        for rect in rects:
            x, y, x2, y2 = int(rect[0] * scale), int(rect[1] * scale), int(rect[2] * scale), int(rect[3] * scale)
            cv2.rectangle(bbox_img, (x, y), (x2, y2), color, line_weight)
            # 显示置信度
            if label:
                cv2.putText(bbox_img, str(label), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
            img[np.where(bbox_img > 0)] = 0
        cam_file_path = Plot.save_img(bbox_img, out_put_dir)

        return cam_file_path

    @staticmethod
    def cal_cam_bbox(cam):
        cam = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam <= 0.6)] = 0
        heatmap[np.where(heatmap > 0)] = 255
        gray_image = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        x, y, w, h = cv2.boundingRect(gray_image)
        max_width = int(224 * 0.5)
        if w > max_width:
            x += (w - max_width) // 2
            w = max_width

        if h > max_width:
            y += (h - max_width) // 2
            h = max_width
        if y < 4:
            y = 4
        return [x, y, x + w, y + h]

    @staticmethod
    def save_img(bbox_img, out_put_dir):
        rand = random.randint(1, 1000)
        file_name = time.strftime("%Y-%m-%d-%H-%M_", time.localtime()) + str(rand) + ".png"
        file_path = os.path.abspath(out_put_dir + '/' + file_name)
        cv2.imwrite(file_path, bbox_img)
        logger.info("export img %s " % file_path)
        return file_path

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        # 自定义混淆矩阵颜色
        # cmap： color map，https://matplotlib.org/examples/color/colormaps_reference.html
        # https://scipy.github.io/old-wiki/pages/Cookbook/Matplotlib/Show_colormaps
        # custom_cm = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#7598AC'], 256) # 灰蓝
        # custom_cm = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#705286'], 100) # 紫色
        # custom_cm = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#304193'], 100) # 深蓝

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        # 去掉该语句，否则保存的混淆矩阵图左侧文字显示不全
        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    @staticmethod
    def show_matrix(y_pred, y_true, classes_count, out_put_dir, fig_size=4, dpi=110, classes=None):

        cnf_matrix = confusion_matrix(y_true, y_pred)
        # Plot non-normalized confusion matrix

        plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
        if classes is None:
            classes = [str(x) for x in range(classes_count)]

        Plot.plot_confusion_matrix(cnf_matrix, classes=classes,
                                   title='Confusion matrix')

        rand = random.randint(1, 1000)
        file_name = time.strftime("%Y-%m-%d-%H-%M_", time.localtime()) + str(rand) + ".png"
        file_path = os.path.abspath(out_put_dir + '/' + file_name)
        plt.tight_layout()
        plt.savefig(file_path)
        logger.info("save plot image to %s " % file_path)

        err = 0

        txt_path = os.path.abspath(out_put_dir + '/cm.txt')
        txt_str = ''

        for i in range(0, len(y_pred)):
            if y_pred[i] != y_true[i]:
                err += 1

        overall_acc = 1 - err * 1.0 / len(y_pred)
        print(cnf_matrix)
        txt_str += cnf_matrix.__str__() + '\n'
        acc_list = []
        for i in range(cnf_matrix.shape[0]):
            acc = 100 * cnf_matrix[i, i] / np.sum(cnf_matrix[i, :])
            print('%02d acc: %.2f%%' % (i, acc))
            txt_str += ('%02d acc: %.2f%%' % (i, acc)).__str__() + '\n'
            acc_list.append(acc)
        print('overall acc: %.2f%%, avg acc: %.2f%%' % (100 * overall_acc, np.mean(acc_list)))
        txt_str += ('overall acc: %.2f%%, avg acc: %.2f%%' % (100 * overall_acc, np.mean(acc_list))).__str__()
        with open(txt_path, 'w') as f:
            f.write(txt_str)

        logger.info("accurracy:{}".format(overall_acc))
        return file_path

    @staticmethod
    def get_roc(y_true, predicts, to_check_path_result, threshold_num=20, to_print=True):
        pred_list = []
        for itm in predicts:
            pred_list.extend(list(itm))

        data = list(zip(pred_list, y_true))

        fpr, tpr, thresholds = roc_curve(y_true, pred_list, pos_label=1)
        tp = []
        fp = []
        fn = []
        tn = []

        for thrd in thresholds:
            thrd_tp, thrd_fp, thrd_fn, thrd_tn = [0] * 4
            for item in data:
                if item[1] == 1:
                    if item[0] >= thrd:
                        thrd_tp += 1
                    else:
                        thrd_fn += 1
                elif item[1] == 0:
                    if item[0] >= thrd:
                        thrd_fp += 1
                    else:
                        thrd_tn += 1

            tp.append(thrd_tp)
            fp.append(thrd_fp)
            fn.append(thrd_fn)
            tn.append(thrd_tn)

        diff = [round(tpr[i] - fpr[i], 5) for i in range(len(tpr))]
        optimal_idx = np.argmax(diff)
        optimal_threshold = thresholds[optimal_idx]

        optimal_acc0 = round((tn[optimal_idx] / (tn[optimal_idx] + fp[optimal_idx])) * 100, 3)
        optimal_acc1 = round((tp[optimal_idx] / (tp[optimal_idx] + fn[optimal_idx])) * 100, 3)
        optimal_avg_acc = np.mean([optimal_acc0, optimal_acc1])
        optimal_overall_acc = round((tn[optimal_idx] + tp[optimal_idx]) / len(y_true) * 100, 3)

        print('optimal_threshold: ', optimal_threshold,
              ' overall acc:  %.3f%%, avg acc: %.3f%%' % (optimal_overall_acc, optimal_avg_acc))

        if to_print:
            print("{}\t{}\t{}\t{}".format('thred', 'tpr', 'fpr', 'diff'))
            for i, thrd in enumerate(thresholds):
                print('{}\t{}\t{}\t{}'.format(thresholds[i], tpr[i], fpr[i], diff[i]))

        acc0 = [round((item / (item + fp[i])) * 100, 3) for i, item in enumerate(tn)]
        acc1 = [round((item / (item + fn[i])) * 100, 3) for i, item in enumerate(tp)]
        auc_list = [auc(fpr, tpr)]
        for i in range(len(thresholds) - 1):
            auc_list.append(np.nan)
        df = pd.DataFrame({'thresholds': thresholds, 'tpr': tpr, 'fpr': fpr, 'tpr-fpr': diff,
                           'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                           'acc0': acc0, 'acc1': acc1, 'AUC': auc_list})
        df = df.loc[:, ['thresholds', 'tpr', 'fpr', 'tpr-fpr', 'tn', 'fp', 'fn', 'tp', 'acc0', 'acc1', 'AUC']]
        # df = df.ix[:, ['thresholds', 'tpr', 'fpr', 'tpr-fpr', 'tn', 'fp', 'fn', 'tp', 'acc0', 'acc1']]
        df.to_csv(to_check_path_result + r'\roc_%s_%s.csv' % (optimal_threshold, optimal_avg_acc), encoding='utf-8')

        fontsize = 14
        ax = plt.figure(figsize=(10, 8))  # 为了让图像大一点
        plt.plot(fpr, tpr, lw=2)  # 画出这些点连成的线
        # for a, b in zip(fpr, tpr):
        #     plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)
        plt.title('AUC:{:.2f}'.format(auc(fpr, tpr)))
        plt.ylabel('sensitivity', fontdict={'family': 'Times New Roman', 'size': fontsize})
        plt.xlabel('1-specificity', fontdict={'family': 'Times New Roman', 'size': fontsize})

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        new_ticks = np.linspace(0, 1, 11)
        plt.xticks(new_ticks, fontproperties='Times New Roman', fontsize=fontsize)
        plt.yticks(new_ticks, fontproperties='Times New Roman', fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)

        plt.annotate(r'threshold={:.3f}'.format(optimal_threshold), xy=(fpr[optimal_idx], tpr[optimal_idx]),
                     xycoords='data', xytext=(+30, -30),
                     textcoords='offset points', fontsize=fontsize, color='blue',
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1", color='red'))

        plt.savefig(to_check_path_result + r'\roc_%s_%s.png' % (optimal_threshold, optimal_avg_acc))

    @staticmethod
    def get_roc_bak(y_true, predicts, to_check_path_result, threshold_num=20, to_print=True):
        pred_list = []
        for itm in predicts:
            pred_list.extend(list(itm))

        data = list(zip(pred_list, y_true))
        thresholds = [i / threshold_num for i in range(0, threshold_num, 1)]
        thresholds.append(1)

        tp = []
        fp = []
        fn = []
        tn = []
        tpr = []
        fpr = []
        for thrd in thresholds:
            thrd_tp, thrd_fp, thrd_fn, thrd_tn = [0] * 4
            for item in data:
                if item[1] == 1:
                    if item[0] >= thrd:
                        thrd_tp += 1
                    else:
                        thrd_fn += 1
                elif item[1] == 0:
                    if item[0] >= thrd:
                        thrd_fp += 1
                    else:
                        thrd_tn += 1

            thrd_tpr = round(float(thrd_tp) / (thrd_tp + thrd_fn), 5)
            thrd_fpr = round(float(thrd_fp) / (thrd_tn + thrd_fp), 5)

            tp.append(thrd_tp)
            fp.append(thrd_fp)
            fn.append(thrd_fn)
            tn.append(thrd_tn)
            tpr.append(thrd_tpr)
            fpr.append(thrd_fpr)

        diff = [round(tpr[i] - fpr[i], 5) for i in range(len(tpr))]
        optimal_idx = np.argmax(diff)
        optimal_threshold = thresholds[optimal_idx]

        optimal_acc0 = round(tn[optimal_idx] / (tn[optimal_idx] + fp[optimal_idx]), 5) * 100
        optimal_acc1 = round(tp[optimal_idx] / (tp[optimal_idx] + fn[optimal_idx]), 5) * 100
        optimal_avg_acc = np.mean([optimal_acc0, optimal_acc1])
        optimal_overall_acc = round((tn[optimal_idx] + tp[optimal_idx]) / len(y_true), 5) * 100

        print('optimal_threshold: ', optimal_threshold,
              ' overall acc:  %.3f%%, avg acc: %.3f%%' % (optimal_overall_acc, optimal_avg_acc))

        if to_print:
            print("{}\t{}\t{}\t{}".format('thred', 'tpr', 'fpr', 'diff'))
            for i, thrd in enumerate(thresholds):
                print('{}\t{}\t{}\t{}'.format(thresholds[i], tpr[i], fpr[i], diff[i]))

        acc0 = [round(item / (item + fp[i]), 5) * 100 for i, item in enumerate(tn)]
        acc1 = [round(item / (item + fn[i]), 5) * 100 for i, item in enumerate(tp)]
        auc_list = [auc(fpr, tpr)]
        for i in range(len(thresholds) - 1):
            auc_list.append(np.nan)
        df = pd.DataFrame({'thresholds': thresholds, 'tpr': tpr, 'fpr': fpr, 'tpr-fpr': diff,
                           'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                           'acc0': acc0, 'acc1': acc1, 'AUC': auc_list})
        df = df.loc[:, ['thresholds', 'tpr', 'fpr', 'tpr-fpr', 'tn', 'fp', 'fn', 'tp', 'acc0', 'acc1', 'AUC']]
        # df = df.ix[:, ['thresholds', 'tpr', 'fpr', 'tpr-fpr', 'tn', 'fp', 'fn', 'tp', 'acc0', 'acc1']]
        df.to_csv(to_check_path_result + r'\roc_%s_%s.csv' % (optimal_threshold, optimal_avg_acc), encoding='utf-8')

        fontsize = 14
        ax = plt.figure(figsize=(10, 8))  # 为了让图像大一点
        plt.plot(fpr, tpr, lw=2)  # 画出这些点连成的线
        # for a, b in zip(fpr, tpr):
        #     plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)
        plt.title('AUC:{:.2f}'.format(auc(fpr, tpr)))
        plt.ylabel('sensitivity', fontdict={'family': 'Times New Roman', 'size': fontsize})
        plt.xlabel('1-specificity', fontdict={'family': 'Times New Roman', 'size': fontsize})

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        new_ticks = np.linspace(0, 1, 11)
        plt.xticks(new_ticks, fontproperties='Times New Roman', fontsize=fontsize)
        plt.yticks(new_ticks, fontproperties='Times New Roman', fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)

        plt.annotate(r'threshold={:.3f}'.format(optimal_threshold), xy=(fpr[optimal_idx], tpr[optimal_idx]),
                     xycoords='data', xytext=(+30, -30),
                     textcoords='offset points', fontsize=fontsize, color='blue',
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1", color='red'))

        plt.savefig(to_check_path_result + r'\roc_%s_%s.png' % (optimal_threshold, optimal_avg_acc))
