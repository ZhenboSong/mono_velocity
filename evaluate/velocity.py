import numpy as np
import json


class VeloEval(object):
    @staticmethod
    def load_json_file(file_list):
        data_list = []
        for file_name in file_list:
            with open(file_name) as f:
                raw_data = json.load(f)
            data_list.append(raw_data)
        if len(data_list) != 1:
            data_list = [data_list]
        return data_list

    @staticmethod
    def transform_annotation(raw_data_list):
        anno_list = []
        for raw_data in raw_data_list:
            # data = []
            for frame in raw_data:
                frame_data = []
                for instance in frame:
                    instance["bbox"] = np.array([[instance["bbox"]["top"],
                                                  instance["bbox"]["left"],
                                                  instance["bbox"]["bottom"],
                                                  instance["bbox"]["right"]]])
                    frame_data.append(instance)
                anno_list.append(frame_data)
            # anno_list.append(data)
        return anno_list

    @staticmethod
    def load_annotation(file_list):
        raw_data_list = VeloEval.load_json_file(file_list)
        anno_list = VeloEval.transform_annotation(raw_data_list)
        print("Finished loading {0:d} annotations.".format(len(anno_list)))
        return anno_list

    @staticmethod
    def calc_mse(a, b):
        try:
            error = np.linalg.norm(np.array(a) - np.array(b)) ** 2
        except BaseException as e:
            raise Exception('Error data format')
        return np.linalg.norm(np.array(a) - np.array(b)) ** 2

    @staticmethod
    def calc_rmse(a, b):
        try:
            error = np.linalg.norm(np.array(a) - np.array(b))
        except BaseException as e:
            raise Exception('Error data format')
        return np.linalg.norm(np.array(a) - np.array(b))

    @staticmethod
    def get_distance_label(gt):
        distance = np.linalg.norm(np.array(gt["position"]))
        if distance < 20:
            return 0
        elif distance < 45:
            return 1
        else:
            return 2

    @staticmethod
    def find_nearest_gt(pred, x_gt):
        bboxes = np.vstack([x["bbox"] for x in x_gt])
        difference = np.sum(np.abs(np.subtract(bboxes, pred["bbox"])), axis=1)
        if np.min(difference) > 10:
            raise Exception('We do not get all the predictions for a certain frame')
        return x_gt[np.argmin(difference)]

    @staticmethod
    def accuracy_velocity(pred_list, gt_list):
        velo_m_rmse = [[], [], []]
        velo_a_rmse = [[], [], []]
        for x_pred, x_gt in zip(pred_list, gt_list):
            for pred in x_pred:
                gt = VeloEval.find_nearest_gt(pred, x_gt)
                distance_label = VeloEval.get_distance_label(gt)
                velo_m_rmse[distance_label].append(VeloEval.calc_rmse(pred["velocity"][0], gt["velocity"][0]))
                velo_a_rmse[distance_label].append(VeloEval.calc_rmse(pred["velocity"][1], gt["velocity"][1]))

        vme0 = np.mean(np.array(velo_m_rmse[0]))
        vme1 = np.mean(np.array(velo_m_rmse[1]))
        vme2 = np.mean(np.array(velo_m_rmse[2]))
        vme_all = np.mean(np.array(velo_m_rmse[0] + velo_m_rmse[1] + velo_m_rmse[2]))

        vae0 = np.mean(np.array(velo_a_rmse[0]))
        vae1 = np.mean(np.array(velo_a_rmse[1]))
        vae2 = np.mean(np.array(velo_a_rmse[2]))
        vae_all = np.mean(np.array(velo_a_rmse[0] + velo_a_rmse[1] + velo_a_rmse[2]))

        print("Velocity Estimation error (Near): {:10.5f}, {:10.5f}".format(vme0, vae0))
        print("Velocity Estimation error (Medium): {:10.5f}, {:10.5f}".format(vme1, vae1))
        print("Velocity Estimation error (Far): {:10.5f}, {:10.5f}".format(vme2, vae2))
        print("Velocity Estimation error total: {:10.5f}, {:10.5f}".format(vme_all, vae_all))

        return (vme0, vme1, vme2, vme_all), (vae0, vae1, vae2, vae_all)

    @staticmethod
    def accuracy_position(pred_list, gt_list):
        pos_error_rmse = [[], [], []]
        for x_pred, x_gt in zip(pred_list, gt_list):
            for pred in x_pred:
                gt = VeloEval.find_nearest_gt(pred, x_gt)
                distance_label = VeloEval.get_distance_label(gt)
                pos_error_rmse[distance_label].append(VeloEval.calc_rmse(pred["position"], gt["position"]))
        pe0 = np.mean(np.array(pos_error_rmse[0]))
        pe1 = np.mean(np.array(pos_error_rmse[1]))
        pe2 = np.mean(np.array(pos_error_rmse[2]))
        pe_all = np.mean(np.array(pos_error_rmse[0] + pos_error_rmse[1] + pos_error_rmse[2]))
        print("Position Estimation rmse (Near): {0:.5f}".format(pe0))
        print("Position Estimation rmse (Medium): {0:.5f}".format(pe1))
        print("Position Estimation rmse (Far): {0:.5f}".format(pe2))
        print("Position Estimation rmse total: {0:.5f}".format(pe_all))
        return pe0, pe1, pe2, pe_all

    @staticmethod
    def accuracy_depth(pred_list, gt_list):
        pred_all = []
        gt_all = []
        pred_depth = [[], [], []]
        gt_depth = [[], [], []]
        for x_pred, x_gt in zip(pred_list, gt_list):
            for pred in x_pred:
                gt = VeloEval.find_nearest_gt(pred, x_gt)
                distance_label = VeloEval.get_distance_label(gt)
                pred_depth[distance_label].append(pred["position"][0])
                gt_depth[distance_label].append(gt['position'][0])
                pred_all.append(pred["position"][0])
                gt_all.append(gt['position'][0])
        rms = [[], [], [], []]
        log_rms = [[], [], [], []]
        abs_rel = [[], [], [], []]
        sq_rel = [[], [], [], []]
        a1 = [[], [], [], []]
        a2 = [[], [], [], []]
        a3 = [[], [], [], []]
        for i in range(3):
            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = VeloEval.compute_errors(
                np.array(gt_depth[i]), np.array(pred_depth[i]))
            if i == 0:
                print('near')
            elif i == 1:
                print('medium')
            else:
                print('far')
            print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".
                  format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
            print("{:10.4f} & {:10.4f} & {:10.3f} & {:10.3f} & {:10.3f} & {:10.3f} & {:10.3f}\n".
                  format(abs_rel[i], sq_rel[i].mean(), rms[i], log_rms[i], a1[i], a2[i], a3[i]))

        abs_rel[3], sq_rel[3], rms[3], log_rms[3], a1[3], a2[3], a3[3] = \
            VeloEval.compute_errors(np.array(gt_all), np.array(pred_all))
        print('all')
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".
              format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
        print("{:10.4f} & {:10.4f} & {:10.3f} & {:10.3f} & {:10.3f} & {:10.3f} & {:10.3f}\n".
              format(abs_rel[3], sq_rel[3].mean(), rms[3], log_rms[3], a1[3], a2[3], a3[3]))
        return log_rms, abs_rel, sq_rel, a1, a2, a3


    @staticmethod
    def benchmark(pred_list, gt_list):
        pos_error = [[], [], []]
        velo_error = [[], [], []]
        for x_pred, x_gt in zip(pred_list, gt_list):
            for pred in x_pred:
                gt = VeloEval.find_nearest_gt(pred, x_gt)
                distance_label = VeloEval.get_distance_label(gt)
                if 'position' not in pred or 'velocity' not in pred:
                    raise Exception('Missing position or velocity')
                pos_error[distance_label].append(VeloEval.calc_mse(pred["position"], gt["position"]))
                velo_error[distance_label].append(VeloEval.calc_mse(pred["velocity"], gt["velocity"]))

        ve0 = np.mean(np.array(velo_error[0]))
        ve1 = np.mean(np.array(velo_error[1]))
        ve2 = np.mean(np.array(velo_error[2]))
        pe0 = np.mean(np.array(pos_error[0]))
        pe1 = np.mean(np.array(pos_error[1]))
        pe2 = np.mean(np.array(pos_error[2]))

        print("Position and Velocity Estimation error (Near): {:10.5f}, {:10.5f}".format(ve0, pe0))
        print("Position and Velocity Estimation error (Medium): {:10.5f}, {:10.5f}".format(ve1, pe1))
        print("Position and Velocity Estimation error (Far): {:10.5f}, {:10.5f}".format(ve2, pe2))
        print("Position and Velocity Estimation error total: {:10.5f}, {:10.5f}".format((ve0 + ve1 + ve2) / 3, (pe0 + pe1 + pe2) / 3))

        return (ve0, ve1, ve2), (pe0, pe1, pe2)

    @staticmethod
    def calc_z_rate(pred, gt):
        try:
            error = np.abs((pred - gt) / gt)
        except BaseException as e:
            raise Exception('Error data format')
        return np.abs((pred - gt) / gt)


    @staticmethod
    def compute_errors(gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

    @staticmethod
    def calc_v_rate(a, b):
        try:
            error = np.linalg.norm(np.array(a) - np.array(b))
        except BaseException as e:
            raise Exception('Error data format')
        return np.linalg.norm(np.array(a) - np.array(b))

    @staticmethod
    def distribution(pred_list, gt_list):
        pos_error = [[], [], []]
        velo_error = [[], [], []]
        dist = [[], [], []]
        for x_pred, x_gt in zip(pred_list, gt_list):
            for pred in x_pred:
                gt = VeloEval.find_nearest_gt(pred, x_gt)
                distance_label = VeloEval.get_distance_label(gt)
                pos_error[distance_label].append(VeloEval.calc_z_rate(pred["position"][0], gt["position"][0]))
                velo_error[distance_label].append(VeloEval.calc_v_rate(pred["velocity"], gt["velocity"]))
                dist[distance_label].append(gt['position'][0])
        return velo_error, pos_error, dist

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            with open(pred_file, 'r') as f:
                json_pred = json.load(f)
            with open(gt_file, 'r') as f:
                json_gt = json.load(f)
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        pred_list = VeloEval.transform_annotation(json_pred)
        gt_list = VeloEval.transform_annotation(json_gt)
        pos_error = [[], [], []]
        velo_error = [[], [], []]
        for x_pred, x_gt in zip(pred_list, gt_list):
            for gt in x_gt:
                pred = VeloEval.find_nearest_gt(gt, x_pred)
                distance_label = VeloEval.get_distance_label(gt)
                if 'position' not in pred or 'velocity' not in pred:
                    raise Exception('Missing position or velocity')
                pos_error[distance_label].append(VeloEval.calc_mse(pred["position"], gt["position"]))
                velo_error[distance_label].append(VeloEval.calc_mse(pred["velocity"], gt["velocity"]))

        ve0 = np.mean(np.array(velo_error[0]))
        ve1 = np.mean(np.array(velo_error[1]))
        ve2 = np.mean(np.array(velo_error[2]))
        pe0 = np.mean(np.array(pos_error[0]))
        pe1 = np.mean(np.array(pos_error[1]))
        pe2 = np.mean(np.array(pos_error[2]))

        return json.dumps([
            {'name': 'EV', 'value': (ve0 + ve1 + ve2) / 3, 'order': 'asc'},
            {'name': 'EVNear', 'value': ve0, 'order': 'asc'},
            {'name': 'EVMed', 'value': ve1, 'order': 'asc'},
            {'name': 'EVFar', 'value': ve2, 'order': 'asc'},
            {'name': 'EP', 'value': (pe0 + pe1 + pe2) / 3, 'order': 'asc'},
            {'name': 'EPNear', 'value': pe0, 'order': 'asc'},
            {'name': 'EPMed', 'value': pe1, 'order': 'asc'},
            {'name': 'EPFar', 'value': pe2, 'order': 'asc'},
        ])


if __name__ == '__main__':
    import sys
    try:
        if len(sys.argv) != 3:
            raise Exception('Invalid input arguments')
        print(VeloEval.bench_one_submit(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(e.message)
        sys.exit(e.message)
