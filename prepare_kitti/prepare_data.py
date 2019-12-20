import pykitti
import numpy as np
from prepare_kitti import parseTrackletXML as xmlParser

import matplotlib.pyplot as plt

import os
import shutil
import random

import json

basedir = '/data/songzb'
date = '2011_09_26'


def takeSecond(elem):
    return elem[1]


class GroundPlane:
    def __init__(self, basedir, date, drive):
        self.dataset = pykitti.raw(basedir, date, drive)
        self.frames = len(self.dataset.timestamps)
        tracklet_path = '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir, date, date, drive)
        self.tracklets, self.tracklet_rects, self.tracklet_types, self.track_id, self.track_center \
            = self.load_tracklets_for_frames(tracklet_path)
        self.track_velocity, self.flowlets = self.load_car_velocity()
        self.track_bb = self.load_car_bb()

    def load_tracklets_for_frames(self, tracklet_path):
        """
        Loads dataset labels also referred to as tracklets, saving them individually for each frame.
        Returns
        -------
        Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
        contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
        types as strings.
        """
        tracklets = xmlParser.parseXML(tracklet_path)

        frame_tracklets = {}
        frame_tracklets_types = {}
        frame_trackid = {}
        frame_track_centers = {}
        for i in range(self.frames):
            frame_tracklets[i] = []
            frame_tracklets_types[i] = []
            frame_trackid[i] = []
            frame_track_centers[i] = []

        # loop over tracklets
        for i, tracklet in enumerate(tracklets):
            # this part is inspired by kitti object development kit matlab code: computeBox3D
            h, w, l = tracklet.size
            # in velodyne coordinates around zero point and without orientation yet
            trackletBox = np.array([
                [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                [0.0, 0.0, 0.0, 0.0, h, h, h, h]
            ])
            # loop over all data in tracklet
            for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
                # determine if object is in the image; otherwise continue
                if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                    continue
                # re-create 3D bounding box in velodyne coordinate system
                yaw = rotation[2]  # other rotations are supposedly 0
                assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
                rotMat = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0]
                ])
                cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
                cam2_velo = self.dataset.calib.T_cam2_velo
                cornerPosInCam2 = np.dot(cam2_velo[0:3, 0:3], cornerPosInVelo) + np.tile(cam2_velo[0:3, 3], (8, 1)).T
                frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInCam2]

                frame_trackid[absoluteFrameNumber] = frame_trackid[absoluteFrameNumber] + [i]

                locationInCam2 = np.dot(cam2_velo[0:3, 0:3], translation) + cam2_velo[0:3, 3]
                frame_track_centers[absoluteFrameNumber] = frame_track_centers[absoluteFrameNumber] + [locationInCam2]

                frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [tracklet.objectType]

        return (tracklets, frame_tracklets, frame_tracklets_types, frame_trackid, frame_track_centers)

    def get_velocity(self, location1, location2, T21):
        l21 = np.dot(T21[0:3, 0:3], location1.reshape(-1)) + T21[0:3, 3]
        velocity = location2.reshape(-1) - l21
        return velocity

    # 获得在相机2图像中的bounding box
    def load_car_bb(self):
        frame_bb_incam2 = {}
        for i in range(0, self.frames):
            frame_bb_incam2[i] = []

        cam2_in = self.dataset.calib.K_cam2
        for i in range(0, self.frames):
            frame = self.tracklet_rects[i]
            if len(frame) == 0:
                continue
            for rect in frame:
                rect_homo = rect / rect[2, :]
                pixel = np.dot(cam2_in, rect_homo)
                bb = [min(pixel[0, :]), min(pixel[1, :]), max(pixel[0, :]), max(pixel[1, :])]
                frame_bb_incam2[i] = frame_bb_incam2[i] + [bb]
        return frame_bb_incam2

    # 在相机2坐标中的坐标来做
    def load_car_velocity(self):
        frame_velocity = {}
        frame_flowlets = {}
        for i in range(self.frames):
            frame_velocity[i] = []
            frame_flowlets[i] = []

        for i in range(1,self.frames):
            obj_num = len(self.track_id[i])
            if obj_num == 0:
                continue;
            else:
                for j in range(0, obj_num):
                    obj_idx = self.track_id[i][j]
                    if obj_idx not in self.track_id[i-1]:
                        continue
                    else:
                        pre_j = self.track_id[i-1].index(obj_idx)
                        if self.tracklet_types[i-1][pre_j] != "Bus"and self.tracklet_types[i-1][pre_j] != "Car" \
                                and self.tracklet_types[i-1][pre_j] != "Van":
                            continue
                        T21 = self.get_cam2_motion(i-1, i)
                        if np.abs(T21[1, 3]) > 0.1:
                            T21[1, 3] = 0
                        X1 = self.track_center[i-1][pre_j]
                        X2 = self.track_center[i][j]
                        # TODO: velocity definition
                        velocity = self.get_velocity(X1, X2, T21)
                        thresh = np.linalg.norm(velocity)
                        if thresh < 0.275:
                            velocity = [0, 0, 0]
                        frame_velocity[i] = frame_velocity[i] + [[obj_idx, X2, velocity]]
                        frame_flowlets[i] = frame_flowlets[i] + [[obj_idx, X2, X2-X1]]
        return frame_velocity, frame_flowlets

    def get_cam2_motion(self, target_idx, source_idx):
        T_imu0_cam0 = np.linalg.inv(self.dataset.calib.T_cam2_imu)
        T_cam1_imu1 = self.dataset.calib.T_cam2_imu
        T_w_imu0 = self.dataset.oxts[target_idx].T_w_imu
        T_imu1_w = np.linalg.inv(self.dataset.oxts[source_idx].T_w_imu)
        T_cam0_cam1 = np.dot(T_cam1_imu1, np.dot(T_imu1_w, np.dot(T_w_imu0, T_imu0_cam0)))
        return T_cam0_cam1

    def gt_visualization(self, idx, b_save=False):
        cam2_velo = self.dataset.calib.T_cam2_velo
        pc_velo = self.dataset.get_velo(idx)
        pc_velo[:, 3] = 1
        pc_cam2 = np.dot(pc_velo, cam2_velo.T)
        pc_cam2[pc_cam2[:, 2] < 1, :] = [0, 0, 0, 0]
        pc_cam2[pc_cam2[:, 2] > 81, :] = [0, 0, 0, 0]
        pc_cam2[pc_cam2[:, 0] < -40, :] = [0, 0, 0, 0]
        pc_cam2[pc_cam2[:, 0] > 40, :] = [0, 0, 0, 0]

        x = pc_cam2[:, 0]
        z = pc_cam2[:, 2]

        fig = plt.figure(dpi=360)
        ax = fig.add_subplot(111)
        plt.title('labeled image')
        ax.scatter(x, z, c='g', marker='.', s=0.3, linewidth=0, alpha=1)  # , cmap='spectral')

        center = self.track_center[idx]
        id = self.track_id[idx]
        velocity = self.track_velocity[idx]

        for i in range(0, len(velocity)):
            id_i = velocity[i][0]
            length = np.linalg.norm(velocity[i][1])
            arrow_x = 4.0 * velocity[i][1][0] / length
            arrow_z = 4.0 * velocity[i][1][2] / length

            j = id.index(id_i)
            obj_o = center[j]

            plt.arrow(obj_o[0], obj_o[2], arrow_x, arrow_z,
                      length_includes_head=True,
                      head_width=0.25, head_length=0.5, fc='r', ec='b')
            plt.text(obj_o[0], obj_o[2], "%.1f m/0.1s" % length)

        ax.set_xlim(-40, 40)
        ax.set_ylim(1, 81)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        if b_save:
            plt.savefig("result.png" % i)
        plt.show()
        plt.close("all")

    def get_label_strings(self, idx):
        velocity = self.track_velocity[idx]
        flow = self.flowlets[idx]
        bb = self.track_bb[idx]
        id = self.track_id[idx]

        if len(velocity) == 0:
            return []

        image = self.dataset.get_cam2(idx)
        labels = []
        json_labels = []
        T = self.get_cam2_motion(idx, idx-1)
        cam2_in = self.dataset.calib.K_cam2
        for i in range(0, len(velocity)):
            obj_idx = velocity[i][0]
            j = id.index(obj_idx)

            b_left = bb[j][0]
            b_top = bb[j][1]
            b_right = bb[j][2]
            b_bottom = bb[j][3]

            if b_left < 0:
                b_left = 0
            if b_top < 0:
                b_top = 0
            if b_right > image.width - 1:
                b_right = image.width - 1
            if b_bottom > image.height - 1:
                b_bottom = image.height - 1

            bi_left = np.round(b_left).astype(np.int)
            bi_top = np.round(b_top).astype(np.int)
            bi_right = np.round(b_right).astype(np.int)
            bi_bottom = np.round(b_bottom).astype(np.int)

            if bi_left == bi_right or bi_top == bi_bottom:
                continue

            x = velocity[i][1][0]
            y = velocity[i][1][1]
            z = velocity[i][1][2]
            if z > 80 or z < 6.0:
                continue

            if x / z > 0.8765 or x / z < -0.8448:
                continue

            if y / z > 0.2802:
                continue

            v = np.linalg.norm(velocity[i][2])
            if v > 5.0:
                continue

            # if v == 0:
            #     continue

            v_x = velocity[i][2][0]
            v_y = velocity[i][2][1]
            v_z = velocity[i][2][2]
            rot = np.arctan2(v_z, v_x) * 180 / np.pi

            homo_c = np.array([x/z, y/z, 1])
            pixel_cam2 = np.dot(cam2_in, homo_c)
            u = pixel_cam2[0]
            v = pixel_cam2[1]

            h, w, l = self.tracklets[obj_idx].size
            a = [u, v, z, h, w, l, v_x, v_y, v_z, b_left, b_top, b_right, b_bottom,
                 T[0, 0], T[0, 1], T[0, 2], T[0, 3],
                 T[1, 0], T[1, 1], T[1, 2], T[1, 3],
                 T[2, 0], T[2, 1], T[2, 2], T[2, 3]
                 ]
            labels.append(a)

            f_x = flow[i][2][0]
            f_y = flow[i][2][1]
            f_z = flow[i][2][2]
            json_labels.append({"velocity":[f_z, f_x],"bbox":{"top":b_top,"right":b_right,"left":b_left,"bottom":b_bottom},
                                "position":[z, x]})

        return np.array(labels), json_labels

    def label_image_show_and_save(self, save_dir, visual_dir, idx):
        out_labels = self.get_label_strings(idx)
        if len(out_labels) == 0 or len(out_labels[0]) == 0:
            print("no cars in image %d" % idx)
            return 0
        labels, json_labels = out_labels
        with open("%s/%010d.json" % (save_dir, idx), "w") as f:
            json.dump(json_labels, f)
        # np.save("%s/%010d.npy" % (save_dir, idx), labels)
        # image = self.dataset.get_cam2(idx)
        # draw = ImageDraw.Draw(image)
        # for i in range(0, len(labels)):
        #     draw.rectangle(labels[i][3], outline='yellow')
        #     draw.text((labels[i][3][2], labels[i][3][3]), "%d" % i, fill=(255, 0, 0))
        #     draw.text((20, 20 + 20*i), "%d- z:%.1f v:%.1f r:%.1f" % (i, labels[i][0], labels[i][1], labels[i][2]), fill=(255, 0, 0))
        # image.show()
        # image.save("%s/%010d.png" % (visual_dir, idx))
        return 1


def generate_raw():
    drive_id = ["0001", "0002", "0005", "0009", "0011", "0013", "0014",
                "0015", "0017", "0018", "0019", "0020", "0022", "0023",
                "0027", "0028", "0029", "0032", "0035", "0036", "0039",
                "0046", "0048", "0051", "0052", "0056", "0057", "0059",
                "0060", "0061", "0064", "0070", "0079", "0084", "0086",
                "0087", "0091"]

    # drive_id = ["0001", "0002", "0011", "0013", "0014",
    #             "0015", "0017", "0018", "0019", "0020", "0022", "0023",
    #             "0027", "0028", "0029", "0035", "0036", "0039",
    #             "0046", "0052", "0056", "0057", "0059",
    #             "0060", "0061", "0064", "0070", "0079", "0084", "0086", "0091", "0093"]

    # drive_id = ["0005"]

    if os.path.exists("../data/flowlet_pair.txt"):
        os.remove("../data/flowlet_pair.txt")
    f = open("../data/flowlet_pair", "a")

    # test = GroundPlane(basedir, date, drive_id[0])
    # b = test.get_label_strings(5)

    for drive in drive_id:
        dataset_dir = basedir + "/" + date + "/" + date + "_drive_" + drive + "_sync"
        relative_dir = date + "/" + date + "_drive_" + drive + "_sync"

        label_dir = dataset_dir + "/flowlet_02"
        visual_dir = dataset_dir + "/visual_02"

        image_relative_dir = relative_dir + "/image_02/data"
        label_relative_dir = relative_dir + "/flowlet_02"

        if os.path.exists(label_dir):
            shutil.rmtree(label_dir)
        os.makedirs(label_dir)

        # if os.path.exists(visual_dir):
        #     shutil.rmtree(visual_dir)
        # os.makedirs(visual_dir)

        test = GroundPlane(basedir, date, drive)
        for i in range(0, test.frames):
            b = test.label_image_show_and_save(label_dir, visual_dir, i)
            if b:
                f.write("%s/%010d.png %s/%010d.png %s/%010d.json\n" %
                        (image_relative_dir, i-1, image_relative_dir, i, label_relative_dir, i))
    f.close()


def write_id_data(id, file_dir):
    dicts = []
    f = open("data/%s.txt" % id, "r")
    ff = open(file_dir, "a")
    for line in f:
        line = line.strip("\n")
        fields = line.split(" ")
        dicts.append({"first_image": fields[0], "second_image": fields[1],
                      "velocity": fields[2]})
    f.close()

    for x in dicts:
        ff.write("%s %s %s\n" % (x["first_image"], x["second_image"], x["velocity"]))
    ff.close()


def separate_by_id():
    drive_id = ["0001", "0002", "0005", "0009", "0011", "0013", "0014",
                "0015", "0017", "0018", "0019", "0020", "0022", "0023",
                "0027", "0028", "0029", "0032", "0035", "0036", "0039",
                "0046", "0048", "0051", "0052", "0056", "0057", "0059",
                "0060", "0061", "0064", "0070", "0079", "0084", "0086",
                "0087", "0091"]

    city_id = set(list(["0001", "0002", "0005", "0009", "0011", "0013", "0014",
                        "0017", "0018", "0048", "0051", "0056", "0057", "0059",
                        "0060", "0084", "0091"]))

    residential_id = set(list(["0019", "0020", "0022", "0023", "0035", "0036", "0039",
                               "0046", "0061", "0064", "0079", "0086", "0087"]))

    road_id = set(list(["0015", "0027", "0028", "0029", "0032", "0052", "0070"]))

    test_id = set(list(["0005", "0048", "0013", "0017", "0057", "0023", "0086", "0028"]))

    city = "data/v2_city_all.txt"
    residential = "data/v2_residential_all.txt"
    road = "data/v2_road_all.txt"
    train_file = "data/v2_train_seq.txt"
    test_file = "data/v2_test_seq.txt"

    if os.path.exists(city):
        os.remove(city)
    if os.path.exists(residential):
        os.remove(residential)
    if os.path.exists(road):
        os.remove(road)
    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(test_file):
        os.remove(test_file)

    for id in drive_id:
        if id in city_id:
            write_id_data(id, city)
        if id in residential_id:
            write_id_data(id, residential)
        if id in road_id:
            write_id_data(id, road)
        if id not in test_id:
            write_id_data(id, train_file)
        if id in test_id:
            write_id_data(id, test_file)


def separate_train():
    dicts = []
    f = open("data/v2_only_motion.txt", "r")
    for line in f:
        line = line.strip("\n")
        fields = line.split(" ")
        dicts.append({"first_image": fields[0], "second_image": fields[1],
                      "velocity": fields[2]})
    f.close()

    # for x in dicts:
    #     f = open("data/%s.txt" % x['first_image'][-38:-34], "a")
    #     f.write("%s %s %s\n" % (x["first_image"], x["second_image"], x["velocity"]))
    #     f.close()

    total_num = len(dicts)
    valid_num = 321
    train_num = int(1.0 * float(total_num)) - valid_num
    test_num = total_num - train_num - valid_num

    train_index = random.sample(range(0, total_num), train_num)
    train_index_set = set(train_index)

    train_dicts = []
    test_and_valid_dicts = []
    for i in range(0, total_num):
        if i in train_index_set:
            train_dicts.append(dicts[i])
        else:
            test_and_valid_dicts.append(dicts[i])

    valid_index = random.sample(range(0, test_num + valid_num), valid_num)
    valid_index_set = set(valid_index)

    valid_dicts = []
    test_dicts = []
    for i in range(0, test_num + valid_num):
        if i in valid_index_set:
            valid_dicts.append(test_and_valid_dicts[i])
        else:
            test_dicts.append(test_and_valid_dicts[i])

    if os.path.exists("velocity_train.txt"):
        os.remove("velocity_train.txt")
    f_train = open("velocity_train.txt", "a")
    for data in train_dicts:
        f_train.write("%s %s %s\n" % (data["first_image"], data["second_image"], data["velocity"]))
    f_train.close()

    # if os.path.exists("velocity_test.txt"):
    #     os.remove("velocity_test.txt")
    # f_test = open("velocity_test.txt", "a")
    # for data in test_dicts:
    #     f_test.write("%s %s %s\n" % (data["first_image"], data["second_image"], data["velocity"]))
    # f_test.close()

    if os.path.exists("velocity_valid.txt"):
        os.remove("velocity_valid.txt")
    f_test = open("velocity_valid.txt", "a")
    for data in valid_dicts:
        f_test.write("%s %s %s\n" % (data["first_image"], data["second_image"], data["velocity"]))
    f_test.close()


def separate_motion():
    basedir = '/data/songzb'
    dicts = []
    f = open("../data/original/v2_pair.txt", "r")
    for line in f:
        line = line.strip("\n")
        fields = line.split(" ")
        dicts.append({"first_image": fields[0], "second_image": fields[1],
                      "velocity": fields[2]})
    f.close()
    for i in range(len(dicts)):
        vall = np.load(basedir + "/" + dicts[i]["velocity"])
        for j in range(len(vall)):
            v1 = vall[j][6]
            v2 = vall[j][7]
            v3 = vall[j][8]
            if v1 != 0 or v2 != 0 or v3 != 0:
                f_test = open("../data/original/v2_only_motion.txt", "a")
                f_test.write("%s %s %s\n" % (dicts[i]["first_image"], dicts[i]["second_image"], dicts[i]["velocity"]))
                f_test.close()
                break


def check_file():
    drive_id = ["0001", "0002", "0005", "0009", "0011", "0013", "0014",
                "0015", "0017", "0018", "0019", "0020", "0022", "0023",
                "0027", "0028", "0029", "0032", "0035", "0036", "0039",
                "0046", "0048", "0051", "0052", "0056", "0057", "0059",
                "0060", "0061", "0064", "0070", "0079", "0084", "0086",
                "0087", "0091"]

    for drive in drive_id:
        dataset_dir = basedir + "/" + date + "/" + date + "_drive_" + drive + "_sync"
        label_dir = dataset_dir + "/motion_02"
        files = os.listdir(label_dir)
        if len(files) < 5:
            print(drive)
    error_id = ['0001','0023','0035','0048','0052','0079','0086','0087','0091']
    return

if __name__ == '__main__':
    # separate_motion()
    generate_raw()