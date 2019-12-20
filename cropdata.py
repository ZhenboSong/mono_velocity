from torch.utils.data import Dataset
import torchvision.transforms as transforms
import argparse
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from utils import *


class AugmentImagePair(object):
    def __init__(self, augment_parameters):
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

    def __call__(self, sample):
        first_image = sample['first_image']
        second_image = sample['second_image']
        for iter_img in range(4):
            first_image_i = first_image[iter_img]
            second_image_i = second_image[iter_img]
            p = np.random.uniform(0, 1, 1)
            if p > 0.5:
                # randomly shift gamma
                random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
                first_image_aug = first_image_i ** random_gamma
                second_image_aug = second_image_i ** random_gamma

                # randomly shift brightness
                random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
                first_image_aug = first_image_aug * random_brightness
                second_image_aug = second_image_aug * random_brightness

                # randomly shift color
                random_colors = np.random.uniform(self.color_low, self.color_high, 3)
                for i in range(3):
                    first_image_aug[i, :, :] *= random_colors[i]
                    second_image_aug[i, :, :] *= random_colors[i]

                # saturate
                first_image_aug = torch.clamp(first_image_aug, 0, 1)
                second_image_aug = torch.clamp(second_image_aug, 0, 1)
                first_image[iter_img] = first_image_aug
                second_image[iter_img] = second_image_aug

        return sample


def generate_flow_rois(obj_rois, ratios, img_w, img_h):
    flow_rois = np.zeros(obj_rois.shape, dtype=obj_rois.dtype)
    new_rois = np.zeros(obj_rois.shape, dtype=obj_rois.dtype)
    size_rois = np.zeros((obj_rois.shape[0], 2), dtype=obj_rois.dtype)
    size_rois[:, 0] = (obj_rois[:, 2] - obj_rois[:, 0] + 20.0) * ratios
    size_rois[:, 1] = (obj_rois[:, 3] - obj_rois[:, 1] + 20.0) * ratios
    flow_rois[:, 0] = (obj_rois[:, 0] + obj_rois[:, 2] - size_rois[:, 0]) / 2.0
    flow_rois[:, 1] = (obj_rois[:, 1] + obj_rois[:, 3] - size_rois[:, 1]) / 2.0
    flow_rois[:, 2] = (obj_rois[:, 0] + obj_rois[:, 2] + size_rois[:, 0]) / 2.0
    flow_rois[:, 3] = (obj_rois[:, 1] + obj_rois[:, 3] + size_rois[:, 1]) / 2.0

    left_i = flow_rois[:, 0] < 0
    flow_rois[left_i, 0] = 0

    top_i = flow_rois[:, 1] < 0
    flow_rois[top_i, 1] = 0

    right_i = flow_rois[:, 2] >= img_w
    flow_rois[right_i, 2] = img_w - 1

    bottom_i = flow_rois[:, 3] >= img_h
    flow_rois[bottom_i, 3] = img_h - 1

    new_rois[:, 0:2] = obj_rois[:, 0:2] - flow_rois[:, 0:2]
    new_rois[:, 2:4] = obj_rois[:, 2:4] - flow_rois[:, 0:2]

    return flow_rois.astype(np.int), new_rois


class CropLoader(Dataset):
    def __init__(self, args, root_dir, pair_dir):
        self.args = args
        self.dicts = []
        f = open(pair_dir, "r")
        for line in f:
            line = line.strip("\n")
            fields = line.split(" ")
            self.dicts.append(
                {"first_image": root_dir + '/' + fields[0], "second_image": root_dir + '/' + fields[1],
                 "velocity": root_dir + '/' + fields[2]})
        f.close()

        augment_parameters = [0.8, 1.2, 0.5, 2.0, 0.8, 1.2]
        self.transform = transforms.Compose([
            transforms.Resize((384, 448)),
            transforms.ToTensor()
        ])
        self.augment = AugmentImagePair(augment_parameters)
        print('data processing Croper', args.mode)

    def __len__(self):
        return len(self.dicts)

    def __getitem__(self, idx):
        first_image = Image.open(self.dicts[idx]["first_image"])
        second_image = Image.open(self.dicts[idx]["second_image"])

        with open(self.dicts[idx]["velocity"], 'r') as load_f:
            json_dict = json.load(load_f)
        data_num = len(json_dict)

        perspective_size = []
        velocity = []
        location = []
        bbox = []

        if self.args.mode == 'train':
            max_pos = 0
            max_pos_id = 0
            for i in range(data_num):
                t_bbox = json_dict[i]['bbox']
                t_bbox = [t_bbox['left'], t_bbox['top'], t_bbox['right'], t_bbox['bottom']]
                bbox.append(t_bbox)
                velocity.append(json_dict[i]['velocity'])
                location.append(json_dict[i]['position'])
                perspective_size.append([self.args.cam_fx / (t_bbox[2] - t_bbox[0]),
                                         self.args.cam_fy / (t_bbox[3] - t_bbox[1]),
                                         (t_bbox[0] - self.args.cam_cx) / self.args.cam_fx,
                                         (t_bbox[1] - self.args.cam_cy) / self.args.cam_fy,
                                         (t_bbox[2] - self.args.cam_cx) / self.args.cam_fx,
                                         (t_bbox[3] - self.args.cam_cy) / self.args.cam_fy])
                if json_dict[i]['position'][0] > max_pos:
                    max_pos_id = i
                    max_pos = json_dict[i]['position'][0]

            if data_num == 1:
                bbox = bbox * 4
                location = location * 4
                velocity = velocity * 4
                perspective_size = perspective_size * 4
            elif data_num == 2:
                bbox = bbox * 2
                location = location * 2
                velocity = velocity * 2
                perspective_size = perspective_size * 2
            elif data_num == 3:
                t_bbox = json_dict[max_pos_id]['bbox']
                t_bbox = [t_bbox['left'], t_bbox['top'], t_bbox['right'], t_bbox['bottom']]
                bbox.append(t_bbox)
                velocity.append(json_dict[max_pos_id]['velocity'])
                location.append(json_dict[max_pos_id]['position'])
                perspective_size.append([self.args.cam_fx / (t_bbox[2] - t_bbox[0]),
                                         self.args.cam_fy / (t_bbox[3] - t_bbox[1]),
                                         (t_bbox[0] - self.args.cam_cx) / self.args.cam_fx,
                                         (t_bbox[1] - self.args.cam_cy) / self.args.cam_fy,
                                         (t_bbox[2] - self.args.cam_cx) / self.args.cam_fx,
                                         (t_bbox[3] - self.args.cam_cy) / self.args.cam_fy])
            elif data_num > 4:
                data_num = 4
                location = location[0:4]
                velocity = velocity[0:4]
                bbox = bbox[0:4]
                perspective_size = perspective_size[0:4]

            bbox = np.array(bbox).astype(np.float32)
            location = np.array(location).astype(np.float32)
            velocity = np.array(velocity).astype(np.float32)
            perspective_size = np.array(perspective_size).astype(np.float32)

            flow_bbox, n_bbox = generate_flow_rois(bbox, 2.0, first_image.size[0], first_image.size[1])

            rescale = np.zeros((4, 2), dtype=np.float32)
            rescale[0, 1] = (flow_bbox[0, 3] - flow_bbox[0, 1]) / 384.0
            rescale[0, 0] = (flow_bbox[0, 2] - flow_bbox[0, 0]) / 448.0
            first_flow_img = self.transform(first_image.crop(flow_bbox[0])).unsqueeze(0)
            second_flow_img = self.transform(second_image.crop(flow_bbox[0])).unsqueeze(0)
            for i in range(1, 4):
                rescale[i, 1] = (flow_bbox[i, 3] - flow_bbox[i, 1]) / 384.0
                rescale[i, 0] = (flow_bbox[i, 2] - flow_bbox[i, 0]) / 448.0
                i_first = self.transform(first_image.crop(flow_bbox[i])).unsqueeze(0)
                i_second = self.transform(second_image.crop(flow_bbox[i])).unsqueeze(0)
                first_flow_img = torch.cat((first_flow_img, i_first))
                second_flow_img = torch.cat((second_flow_img, i_second))

            n_bbox[:, 0] = n_bbox[:, 0] / rescale[:, 0]
            n_bbox[:, 1] = n_bbox[:, 1] / rescale[:, 1]
            n_bbox[:, 2] = n_bbox[:, 2] / rescale[:, 0]
            n_bbox[:, 3] = n_bbox[:, 3] / rescale[:, 1]

            perspective_size = torch.from_numpy(perspective_size)
            n_bbox = torch.from_numpy(n_bbox)
            location = torch.from_numpy(location)
            velocity = torch.from_numpy(velocity)
            rescale = torch.from_numpy(rescale)

            sample = {'first_image': first_flow_img, 'second_image': second_flow_img,
                      'rois': n_bbox, 'rois_size': perspective_size, 'rescale': rescale,
                      'location': location, 'velocity': velocity}
            sample = self.augment(sample)
        else:
            for i in range(data_num):
                t_bbox = json_dict[i]['bbox']
                t_bbox = [t_bbox['left'], t_bbox['top'], t_bbox['right'], t_bbox['bottom']]
                if self.args.dataset == 'Kitti':
                    velocity.append(json_dict[i]['velocity'])
                    location.append(json_dict[i]['position'])
                bbox.append(t_bbox)
                perspective_size.append([self.args.cam_fx / (t_bbox[2] - t_bbox[0]),
                                         self.args.cam_fy / (t_bbox[3] - t_bbox[1]),
                                         (t_bbox[0] - self.args.cam_cx) / self.args.cam_fx,
                                         (t_bbox[1] - self.args.cam_cy) / self.args.cam_fy,
                                         (t_bbox[2] - self.args.cam_cx) / self.args.cam_fx,
                                         (t_bbox[3] - self.args.cam_cy) / self.args.cam_fy])

            bbox = np.array(bbox).astype(np.float32)
            perspective_size = np.array(perspective_size).astype(np.float32)
            flow_bbox, n_bbox = generate_flow_rois(bbox, 2.0, first_image.size[0], first_image.size[1])

            # draw1 = ImageDraw.Draw(second_image)
            # draw2 = ImageDraw.Draw(first_image)
            # for i in range(data_num):
            #     draw1.rectangle([flow_bbox[i, 0], flow_bbox[i, 1], flow_bbox[i, 2], flow_bbox[i, 3]], outline='blue', width=5)
            #     draw1.rectangle([bbox[i, 0], bbox[i, 1], bbox[i, 2], bbox[i, 3]], outline='red', width=5)
            #     draw2.rectangle([flow_bbox[i, 0], flow_bbox[i, 1], flow_bbox[i, 2], flow_bbox[i, 3]], outline='blue', width=5)
            # first_image.save("/data/songzb/projects/velocity/velocity-final/crop/first_%d.png" % idx)
            # second_image.save("/data/songzb/projects/velocity/velocity-final/crop/second_%d.png" % idx)

            rescale = np.zeros((data_num, 2), dtype=np.float32)
            rescale[0, 1] = (flow_bbox[0, 3] - flow_bbox[0, 1]) / 384.0
            rescale[0, 0] = (flow_bbox[0, 2] - flow_bbox[0, 0]) / 448.0
            first_flow_img = self.transform(first_image.crop(flow_bbox[0])).unsqueeze(0)
            second_flow_img = self.transform(second_image.crop(flow_bbox[0])).unsqueeze(0)
            for i in range(1, data_num):
                rescale[i, 1] = (flow_bbox[i, 3] - flow_bbox[i, 1]) / 384.0
                rescale[i, 0] = (flow_bbox[i, 2] - flow_bbox[i, 0]) / 448.0
                i_first = self.transform(first_image.crop(flow_bbox[i])).unsqueeze(0)
                i_second = self.transform(second_image.crop(flow_bbox[i])).unsqueeze(0)
                first_flow_img = torch.cat((first_flow_img, i_first))
                second_flow_img = torch.cat((second_flow_img, i_second))

            n_bbox[:, 0] = n_bbox[:, 0] / rescale[:, 0]
            n_bbox[:, 1] = n_bbox[:, 1] / rescale[:, 1]
            n_bbox[:, 2] = n_bbox[:, 2] / rescale[:, 0]
            n_bbox[:, 3] = n_bbox[:, 3] / rescale[:, 1]

            perspective_size = torch.from_numpy(perspective_size)
            n_bbox = torch.from_numpy(n_bbox)
            rescale = torch.from_numpy(rescale)
            original_box = torch.from_numpy(np.array(bbox))

            if self.args.dataset == 'Kitti':
                location = np.array(location).astype(np.float32)
                velocity = np.array(velocity).astype(np.float32)
                location = torch.from_numpy(location)
                velocity = torch.from_numpy(velocity)
                sample = {'first_image': first_flow_img, 'second_image': second_flow_img,
                          'rois': n_bbox, 'rois_size': perspective_size, 'rescale': rescale,
                          'original_box': original_box, 'original_dir': self.dicts[idx]["second_image"],
                          'car_num': data_num, 'location': location, "velocity": velocity}
            else:
                sample = {'first_image': first_flow_img, 'second_image': second_flow_img,
                      'rois': n_bbox, 'rois_size': perspective_size, 'rescale': rescale,
                      'original_box': original_box, 'original_dir': self.dicts[idx]["second_image"], 'car_num':data_num}
        return sample


def crop_dataloader(args, root_dir, data_directory, mode, batch_size, num_workers):
    dataset = CropLoader(args, root_dir, data_directory)
    data_num = len(dataset)
    print('Use a dataset with', data_num, 'images')
    if mode == 'train':
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    return data_num, loader


def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Mono Velocity')

    # dir info
    parser = argparse.ArgumentParser(description='PyTorch Mono Velocity')

    parser.add_argument('--val_data', default='data/ts_valid.txt')
    parser.add_argument('--train_data', default='data/ts_train.txt')
    parser.add_argument('--trained_model', default='pwc_net_chairs.pth.tar')
    parser.add_argument('--car_depth_model', default='results/cardepth_ts_3supp/CarDepthS_cpt.pth')
    parser.add_argument('--dataset', default='Tusimple')
    parser.add_argument('--output_directory', default='./results')

    # network parameters
    parser.add_argument('--basedir', default='/data/songzb/velocity/benchmark_velocity_train')
    parser.add_argument('--md', default=4)
    parser.add_argument('--image_h', default=576)
    parser.add_argument('--image_w', default=1280)
    parser.add_argument('--cam_fx', default=714.1526)
    parser.add_argument('--cam_fy', default=710.3725)
    parser.add_argument('--cam_cx', default=675.5754)
    parser.add_argument('--cam_cy', default=304.2563)
    parser.add_argument('--pre_height', default=1.80)
    parser.add_argument('--time_inv', default=1.0)

    parser.add_argument('--mode', default='train')
    parser.add_argument('--pre_trained', default=True)
    parser.add_argument('--epochs', default=120)
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--adjust_lr', default=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--num_workers', default=4)
    parser.add_argument('--use_multiple_gpu', default=False)

    parser.add_argument('--div_distance', default=1.0 / 10.0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = return_arguments()
    n_img, loader = crop_dataloader(args, args.train_data, args.mode, args.batch_size, args.num_workers)
    for (i, data) in enumerate(loader):
        f_img = data['first_image'].squeeze(0)
        s_img = data['second_image'].squeeze(0)
        bbox = data['rois'].squeeze(0)
        num = f_img.shape[0]
        for j in range(num):
            image_a = transforms.ToPILImage()(f_img[j]).convert('RGB')
            image_b = transforms.ToPILImage()(s_img[j]).convert('RGB')
            draw = ImageDraw.Draw(image_b)
            draw.rectangle([bbox[j, 0], bbox[j, 1], bbox[j, 2], bbox[j, 3]], outline='red')
            image_b.save("crop/%d_%03d_b.png" % (i, j))
            image_a.save("crop/%d_%03d_a.png" % (i, j))

