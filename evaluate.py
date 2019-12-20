from evaluate.velocity import VeloEval
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
import os


def visualization(idx, pred, gt):
    file = "/data/songzb/velocity/benchmark_velocity_test/clips/%d/imgs/040.jpg" % idx
    idx = idx - 1
    image = Image.open(file).convert('RGB')
    draw = ImageDraw.Draw(image)
    for i in range(0, len(pred[idx])):
        pred_label = pred[idx][i]
        gt_label = VeloEval.find_nearest_gt(pred_label, gt[idx])
        rect = [pred_label['bbox'][0][1], pred_label['bbox'][0][0],
                pred_label['bbox'][0][3], pred_label['bbox'][0][2]]
        draw.rectangle([(rect[0], rect[1]), (rect[2], rect[3])], outline='yellow')
        draw.text((rect[2], rect[3]), "%d" % i, fill=(255, 0, 0))
        draw.text((20, 20 + 20*i), "%d- z:%.1f tz:%.1f v:%.1f tv:%.lf"
                  % (i, pred_label['position'][0], gt_label['position'][0],
                     pred_label['velocity'][0], gt_label['velocity'][0]), fill=(255, 0, 0))
    # image.show()
    image.save("visualization/%d.png" % (idx+1))


def eval(b_plot=False, pred=None, gt=None):
    VeloEval.accuracy_depth(pred, gt)
    print('-----------------------------------')
    VeloEval.accuracy_position(pred, gt)
    print('-----------------------------------')
    VeloEval.accuracy_velocity(pred, gt)
    print('-----------------------------------')
    VeloEval.benchmark(pred, gt)

    error_v, error_z, z = VeloEval.distribution(pred, gt)

    if b_plot:
        fig = plt.figure(dpi=360)
        ax = fig.add_subplot(111)
        ax.set_xlabel('depth')
        ax.set_ylabel('RMSE velocity')
        ax.bar(z[0]+z[1]+z[2], error_v[0]+error_v[1]+error_v[2], width=0.3, edgecolor='none')
        # ax.set_xticks(z[0])
        # ax.scatter(z[0], error_v[0], c='g', marker='.', s=10, linewidth=0, alpha=1)
        # ax.scatter(z[1], error_v[1], c='g', marker='.', s=10, linewidth=0, alpha=1)
        # ax.scatter(z[2], error_v[2], c='g', marker='.', s=10, linewidth=0, alpha=1)
        # plt.title('error distribution velocity')
        plt.savefig("tusimple_velocity_distance.jpg")
        plt.show()

        fig1 = plt.figure(dpi=360)
        ax = fig1.add_subplot(111)
        ax.set_xlabel('depth')
        ax.set_ylabel('absRel depth')
        ax.bar(z[0] + z[1] + z[2], error_z[0] + error_z[1] + error_z[2], width=0.3, edgecolor='none')
        # ax.scatter(z[0], error_z[0], c='g', marker='.', s=10, linewidth=0, alpha=1)
        # ax.scatter(z[1], error_z[1], c='g', marker='.', s=10, linewidth=0, alpha=1)
        # ax.scatter(z[2], error_z[2], c='g', marker='.', s=10, linewidth=0, alpha=1)
        # plt.title('error distribution distance')
        plt.savefig("tusimple_absRel_distance.jpg")
        plt.show()

        # fig2 = plt.figure(dpi=360)
        # ax = fig2.add_subplot(131)
        # ax.scatter(error_z[0], error_v[0], c='g', marker='.', s=10, linewidth=0, alpha=1)
        # ax = fig2.add_subplot(132)
        # ax.scatter(error_z[1], error_v[1], c='g', marker='.', s=10, linewidth=0, alpha=1)
        # ax = fig2.add_subplot(133)
        # ax.scatter(error_z[2], error_v[2], c='g', marker='.', s=10, linewidth=0, alpha=1)
        # plt.title('distance/velocity')
        # # plt.ylim(0.0, 20)
        # plt.show()


if __name__ == '__main__':
    # dicts = []
    # f = open('data/original/kitti_motion_test.txt', "r")
    # for line in f:
    #     line = line.strip("\n")
    #     fields = line.split(" ")
    #     dicts.append('/data/songzb' + '/' + fields[2])
    # f.close()
    # kitti_gt = VeloEval.load_annotation(dicts)

    # kitti_pred = VeloEval.load_annotation(['results/kitti_depth/kitti_dorndepth_min.json'])
    #
    # gt_list = []
    # pred_list = []
    # for i in range(len(kitti_gt)):
    #     for j in range(len(kitti_gt[i])):
    #         gt_list.append(kitti_gt[i][j]['position'][0])
    #         pred_list.append(kitti_pred[i][j]['position'][0])
    # import numpy as np
    # gt_np = np.array(gt_list)
    # pred_np = np.array(pred_list)
    # scalor = np.median(gt_np-2) / np.median(pred_np)
    # for i in range(len(kitti_gt)):
    #     for j in range(len(kitti_gt[i])):
    #         kitti_pred[i][j]['position'][0] = kitti_pred[i][j]['position'][0]*scalor
    #
    # eval(False, kitti_pred, kitti_gt)

    # for i in range(1, len(ts_gt)):
    #     visualization(i, ts_pred, ts_gt)

    ts_gt = VeloEval.load_annotation(['data/gt.json'])
    ts_pred = VeloEval.load_annotation(['results/crop_velocity.json'])
    eval(False, ts_pred, ts_gt)

