import argparse
import time
import torch.optim as optim

import collections

from networks.cropvelocity import *
from cropdata import *

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

    parser.add_argument('--trained_model', default='results/cropflowandsupp/cropflowandsupp6_cpt.pth')
    parser.add_argument('--output_directory', default='./results')

    parser.add_argument('--val_data', default='data/tusimple_valid.txt')
    parser.add_argument('--train_data', default='data/tusimple_train.txt')
    parser.add_argument('--basedir', default='/data/songzb/velocity/benchmark_velocity_train')
    parser.add_argument('--valdir', default='/data/songzb/velocity/benchmark_velocity_train')
    parser.add_argument('--md', default=4)
    parser.add_argument('--cam_fx', default=714.1526)
    parser.add_argument('--cam_fy', default=710.3725)
    parser.add_argument('--cam_cx', default=675.5754)
    parser.add_argument('--cam_cy', default=304.2563)
    parser.add_argument('--time_inv', default=1.0)

    # parser.add_argument('--val_data', default='data/original/km_valid.txt')
    # parser.add_argument('--train_data', default='data/original/km_train.txt')
    # parser.add_argument('--basedir', default='/data/songzb')
    # parser.add_argument('--valdir', default='/data/songzb')
    # parser.add_argument('--md', default=4)
    # parser.add_argument('--image_h', default=375)
    # parser.add_argument('--image_w', default=1242)
    # parser.add_argument('--cam_fx', default=721.5377)
    # parser.add_argument('--cam_fy', default=721.5377)
    # parser.add_argument('--cam_cx', default=609.5593)
    # parser.add_argument('--cam_cy', default=172.854)
    # parser.add_argument('--time_inv', default=0.1)

    # train and test parameters
    parser.add_argument('--mode', default='train')
    parser.add_argument('--pre_trained', default=True)
    parser.add_argument('--epochs', default=120)
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--batch_size', default=2)
    parser.add_argument('--adjust_lr', default=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', default=4)
    parser.add_argument('--use_multiple_gpu', default=False)

    parser.add_argument('--div_distance', default=1.0 / 10.0)

    args = parser.parse_args()
    return args


class VelocityModel:

    def __init__(self, args):
        self.args = args

        # Set up model
        self.device = args.device
        self.model = crop_velocity_net(args)
        if not args.pre_trained:
            self.load(args.trained_model)

        self.model = self.model.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)

        self.loss_function = CropLoss(args).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.2)
        self.val_n_img, self.val_loader = crop_dataloader(args, args.valdir, args.val_data, args.mode, args.batch_size,
                                                              args.num_workers)

        # Load data
        self.n_img, self.loader = crop_dataloader(args, args.basedir, args.train_data, args.mode, args.batch_size,
                                                      args.num_workers)

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def train(self):
        best_loss = float('Inf')

        running_val_loss = 0.0
        self.model.eval()
        for data in self.val_loader:
            data = to_device(data, self.device)
            res = self.model(data)
            v_z_loss = self.loss_function(res, data)
            running_val_loss += v_z_loss.item()

        running_val_loss /= self.val_n_img / self.args.batch_size
        best_val_loss = running_val_loss
        print('Val_loss:', running_val_loss)

        for epoch in range(self.args.epochs):
            if self.args.adjust_lr:
                self.scheduler.step()
            c_time = time.time()
            run_v_loss = 0.0
            iter_per_epoch = int(0)
            self.model.train()
            for data in self.loader:
                # Load data
                data = to_device(data, self.device)

                # One optimization iteration
                self.optimizer.zero_grad()
                res = self.model(data)
                v_z_loss = self.loss_function(res, data)
                v_z_loss.backward()
                self.optimizer.step()

                run_v_loss += v_z_loss.item()

                iter_per_epoch += 1
                if iter_per_epoch % 500 == 0:
                    print('Epoch:', epoch + 1,
                          'process:', iter_per_epoch, '/1000',
                          'v_loss:', v_z_loss.data.cpu().numpy(),
                          'run_loss:', run_v_loss / float(iter_per_epoch))

            running_val_loss = 0.0
            self.model.eval()
            for data in self.val_loader:
                data = to_device(data, self.device)
                res = self.model(data)
                v_z_loss = self.loss_function(res, data)
                running_val_loss += v_z_loss.item()

            # Estimate loss per image
            run_v_loss /= self.n_img / self.args.batch_size
            running_val_loss /= self.val_n_img / self.args.batch_size
            print('Epoch:', epoch + 1,
                  'run_loss:', run_v_loss,
                  'val_loss:', running_val_loss,
                  'time:', round(time.time() - c_time, 3), 's')

            self.save(self.args.output_directory + '/cropflowandsupp6_last.pth')
            if running_val_loss < best_val_loss:
                self.save(self.args.output_directory + '/cropflowandsupp6_cpt.pth')
                best_val_loss = running_val_loss
                print('Model_saved')

        print('Finished Training. Best loss:', best_loss)
        self.save(self.args.output_directory + '/cropflowandsupp6_finished.pth')

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        print('load pre_trained model: ' + path)
        self.model.load_state_dict(torch.load(path))


def main():
    args = return_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model = VelocityModel(args)
    model.train()


if __name__ == '__main__':
    main()


