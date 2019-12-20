import argparse
import time
from PIL import ImageDraw

import collections

# custom modules

from cropdata import *
from networks.cropvelocity import *

import os


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")


def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Mono Velocity')

    # dir info
    parser.add_argument('--test_data', default='data/tusimple_test.txt')
    parser.add_argument('--trained_model', default='results/cropflowandsupp/cropflowandsupp6_cpt.pth')
    parser.add_argument('--dataset', default='Tusimple')
    parser.add_argument('--output_directory', default='results')
    parser.add_argument('--res_file', default='crop_velocity.json')
    parser.add_argument('--basedir', default='/home/song/Documents/data/benchmark_velocity_test')
    parser.add_argument('--md', default=4)
    parser.add_argument('--image_h', default=576)
    parser.add_argument('--image_w', default=1280)
    parser.add_argument('--cam_fx', default=714.1526)
    parser.add_argument('--cam_fy', default=710.3725)
    parser.add_argument('--cam_cx', default=675.5754)
    parser.add_argument('--cam_cy', default=304.2563)
    parser.add_argument('--pre_height', default=1.80)
    parser.add_argument('--time_inv', default=1.0)

    # parser.add_argument('--test_data', default='data/original/kitti_motion_test.txt')
    # parser.add_argument('--basedir', default='/data/songzb')
    # parser.add_argument('--md', default=4)
    # parser.add_argument('--image_h', default=375)
    # parser.add_argument('--image_w', default=1242)
    # parser.add_argument('--cam_fx', default=721.5377)
    # parser.add_argument('--cam_fy', default=721.5377)
    # parser.add_argument('--cam_cx', default=609.5593)
    # parser.add_argument('--cam_cy', default=172.854)
    # parser.add_argument('--time_inv', default=0.1)
    # parser.add_argument('--trained_model', default='results/kitti/kitti_cropflowandsupp6_cpt.pth')
    # parser.add_argument('--output_directory', default='results')
    # parser.add_argument('--res_file', default='kitti_crop_velocity.json')

    # train and test parameters
    parser.add_argument('--mode', default='test')
    parser.add_argument('--pre_trained', default=False)
    parser.add_argument('--epochs', default=50)
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--adjust_lr', default=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', default=4)
    parser.add_argument('--use_multiple_gpu', default=False)

    parser.add_argument('--save_image', default=False)

    parser.add_argument('--div_distance', default=1.0 / 10.0)

    args = parser.parse_args()
    return args


class VelocityModel:

    def __init__(self, args):
        self.args = args

        # Set up model
        self.device = args.device
        self.model = crop_velocity_net(args)
        self.load(args.trained_model)

        self.dicts = []

        self.model = self.model.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)

        # Load data
        self.n_img, self.loader = crop_dataloader(args, args.basedir, args.test_data, args.mode, args.batch_size,
                                                      args.num_workers)

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def get_res(self, data, res):
        z_v_pred = res
        rois = data['original_box'].squeeze(0)

        u = (0.5 * (rois[:, 0] + rois[:, 2]))
        z = z_v_pred[:, 0] / self.args.div_distance + 2.0
        x = (u - self.args.cam_cx) * z / self.args.cam_fx
        rects = rois.data.cpu().numpy()
        frame_dicts = []
        for i in range(rois.shape[0]):
            frame_dicts.append({"velocity": [float(z_v_pred[i, 1]), float(z_v_pred[i, 2])],
                                "bbox": {"left": float(rects[i, 0]), "top": float(rects[i, 1]),
                                         "right": float(rects[i, 2]), "bottom": float(rects[i, 3])},
                                "position": [float(z[i]), float(x[i])]})

        return frame_dicts

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for (i, data) in enumerate(self.loader):
                # f_img = data['first_image'].squeeze(0)
                # s_img = data['second_image'].squeeze(0)
                # bbox = data['rois'].squeeze(0)
                # num = f_img.shape[0]
                # for j in range(num):
                #     image_a = transforms.ToPILImage()(f_img[j]).convert('RGB')
                #     image_b = transforms.ToPILImage()(s_img[j]).convert('RGB')
                #     draw = ImageDraw.Draw(image_b)
                #     # draw.rectangle([bbox[j, 0], bbox[j, 1], bbox[j, 2], bbox[j, 3]], outline='red', width=5)
                #     image_b.save("crop/%d_%03d_b.png" % (i, j))
                #     image_a.save("crop/%d_%03d_a.png" % (i, j))

                data = to_device(data, self.device)
                c_time = time.time()
                res = self.model(data)
                print(
                    'test:', i + 1,
                    'time:', round(time.time() - c_time, 3),
                    's'
                )
                frame_dict = self.get_res(data, res)
                self.dicts.append(frame_dict)

            with open("%s/%s" % (self.args.output_directory, self.args.res_file), "w") as f:
                json.dump(self.dicts, f)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


def main_test():
    args = return_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model = VelocityModel(args)
    model.test()


if __name__ == '__main__':
    main_test()

