# -*- coding: utf-8 -*-
from __future__ import print_function, division
import json
import time
from torch.nn.functional import sigmoid
import yaml
import warnings
# from models.model import make_model
from tqdm import tqdm
import numpy as np
import torch
import argparse
import cv2
# from datasets.SiamUAV import SiamUAV_test
from tool.utils import load_network
from torchvision import transforms
import os
import glob
from PIL import Image

warnings.filterwarnings("ignore")

# 23.57

def get_opt():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--test_data_dir', default='/home/dmmm/FPI', type=str, help='training dir path')
    parser.add_argument('--num_worker', default=0, type=int, help='')
    parser.add_argument('--checkpoint', default="net_119.pth", type=str, help='')
    parser.add_argument('--k', default=10, type=int, help='')
    parser.add_argument('--SplitK', default=5, type=int, help='')
    parser.add_argument('--savename', default="result_filterR3.txt", type=str, help='')
    parser.add_argument('--GPS_output_filename', default="GPS_pred_gt_filterR3.json", type=str, help='')
    opt = parser.parse_args()
    config_path = 'opts.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)
    opt.stride = config['stride']
    opt.views = config['views']
    opt.transformer = config['transformer']
    opt.pool = config['pool']
    opt.views = config['views']
    opt.LPN = config['LPN']
    opt.block = config['block']
    opt.nclasses = config['nclasses']
    opt.droprate = config['droprate']
    opt.share = config['share']
    opt.h = config['h']
    opt.w = config['w']
    return opt


def create_hanning_mask(center_R):
    hann_window = np.outer(  # np.outer 如果a，b是高维数组，函数会自动将其flatten成1维 ，用来求外积
        np.hanning(center_R+2),
        np.hanning(center_R+2))
    hann_window /= hann_window.sum()
    return hann_window[1:-1,1:-1]

def create_model(opt):
    # model = make_model(opt)
    # state_dict = torch.load(opt.checkpoint)
    # model.load_state_dict(state_dict)
    model = load_network(opt)
    model = model.cuda()
    model.eval()
    return model

class Dataloader_SiamUAV:
    def __init__(self, root_dir, opt, mode="merge_test_700-1800_cr0.95_stride100"):
        '''
        :param root_dir: root of SiamUAV
        :param transform: a dict, format as {"UAV":Compose(),"Satellite":Compose()}
        '''
        super(Dataloader_SiamUAV, self).__init__()
        self.root_dir = root_dir
        self.opt = opt
        self.opt.UAVhw = [256,256]
        self.opt.Satellitehw = [400,400]
        self.K = opt.SplitK
        self.transform = self.get_transformer()
        self.root_dir_train = os.path.join(self.root_dir, mode)
        self.seq = glob.glob(os.path.join(self.root_dir_train, "*"))
        self.list_all_info = self.get_total_info()

    def get_total_info(self):
        list_all_info = []
        for seq in self.seq:
            UAV = os.path.join(seq, "UAV/0.JPG")
            Satellite_list = glob.glob(os.path.join(seq, "Satellite/*"))
            with open(os.path.join(seq, "labels.json"), 'r', encoding='utf8') as fp:
                json_context = json.load(fp)
            with open(os.path.join(seq, "GPS_info.json"), "r", encoding='utf8') as fp:
                gps_info_context = json.load(fp)
            for s in Satellite_list:
                single_dict = {}
                single_dict["UAV"] = UAV
                single_dict["UAV_GPS"] = gps_info_context["UAV"]
                single_dict["Satellite"] = s
                name = os.path.basename(s)
                single_dict["position"] = json_context[name]
                single_dict["Satellite_INFO"] = gps_info_context["Satellite"][name]
                list_all_info.append(single_dict)
        return list_all_info

    def split_to_KxK_parts(self,input_img,K):
        img = cv2.cvtColor(np.asarray(input_img), cv2.COLOR_RGB2BGR)
        # img = cv2.resize(img,(self.opt.Satellitehw[0], self.opt.Satellitehw[1]))
        img = cv2.resize(img,(K*self.opt.UAVhw[0],K*self.opt.UAVhw[1]))
        img_list = []
        for i in range(K):
            for j in range(K):
                part_img = img[
                               i*self.opt.UAVhw[0] : (i+1)*self.opt.UAVhw[0],
                               j*self.opt.UAVhw[1] : (j+1)*self.opt.UAVhw[1],
                               :
                           ]
                image_pil = Image.fromarray(cv2.cvtColor(part_img, cv2.COLOR_BGR2RGB))
                pos_rate = [ ( i+0.5 ) / K, ( j+0.5 ) / K ]
                img_list.append([image_pil,pos_rate])
        return img_list


    def get_transformer(self):
        transform_uav_list = [
            transforms.Resize(self.opt.UAVhw, interpolation=3),
            transforms.ToTensor()
        ]

        transform_satellite_list = [
            transforms.Resize(self.opt.UAVhw, interpolation=3),
            transforms.ToTensor()
        ]

        data_transforms = {
            'UAV': transforms.Compose(transform_uav_list),
            'satellite': transforms.Compose(transform_satellite_list)
        }

        return data_transforms

    def __len__(self):
        return len(self.list_all_info)

    def __getitem__(self, index):
        single_info = self.list_all_info[index]
        UAV_image_path = single_info["UAV"]
        UAV_image_ = Image.open(UAV_image_path)
        UAV_image = self.transform["UAV"](UAV_image_)

        Satellite_image_path = single_info["Satellite"]
        Satellite_image_ = Image.open(Satellite_image_path)
        Satellite_images = self.split_to_KxK_parts(Satellite_image_,self.K)
        pos_infos = []
        KxK_Satellite_images = []
        for splited_img, pos_info in Satellite_images:
            Satellite_image = self.transform["satellite"](splited_img)
            KxK_Satellite_images.append(Satellite_image)
            pos_infos.append(pos_info)

        X, Y = single_info["position"]
        X = int(X / Satellite_image_.height * self.opt.Satellitehw[0])
        Y = int(Y / Satellite_image_.width * self.opt.Satellitehw[1])

        UAV_GPS = single_info["UAV_GPS"]
        # tl_E,tl_N,br_E,br_N,center_distribute_X,center_distribute_Y,map_size
        Satellite_INFO = single_info["Satellite_INFO"]

        return [UAV_image, KxK_Satellite_images, X, Y, UAV_image_path, Satellite_image_path, UAV_GPS, Satellite_INFO, pos_infos]


def create_dataset(opt):
    dataset_test = Dataloader_SiamUAV(opt.test_data_dir, opt)
    dataloaders = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=opt.num_worker,
                                              pin_memory=True)
    return dataloaders


def evaluate(opt, pred_XY, label_XY):
    pred_X, pred_Y = pred_XY
    label_X, label_Y = label_XY
    x_rate = (pred_X - label_X) / opt.Satellitehw[0]
    y_rate = (pred_Y - label_Y) / opt.Satellitehw[1]
    distance = np.sqrt((np.square(x_rate) + np.square(y_rate)) / 2)  # take the distance to the 0-1
    result = np.exp(-1 * opt.k * distance)
    return result


def euclideanDistance(query, gallery):
    query = np.array(query, dtype=np.float32)
    gallery = np.array(gallery, dtype=np.float32)
    A = gallery - query
    A_T = A.transpose()
    distance = np.matmul(A, A_T)
    mask = np.eye(distance.shape[0], dtype=np.bool8)
    distance = distance[mask]
    distance = np.sqrt(distance.reshape(-1))
    return distance


def SDM_evaluateSingle(distance,K):
    # maxDistance = max(distance) + 1e-14
    # weight = np.ones(K) - np.log(range(1, K + 1, 1)) / np.log(opts.M * K)
    weight = np.ones(K) - np.array(range(0,K,1))/K
    # m1 = distance / maxDistance
    m2 = 1 / np.exp(distance*5e3)
    m3 = m2 * weight
    result = np.sum(m3) / np.sum(weight)
    return result


def SDM_evaluate_score(opt,UAV_GPS,Satellite_INFO,UAV_image_path,Satellite_image_path,S_X,S_Y):
    # drone/groundtruth GPS info
    drone_GPS_info = [float(UAV_GPS["E"]), float(UAV_GPS["N"])]
    # Satellite_GPS_info format:[tl_E,tl_N,br_E,br_N]
    Satellite_GPS_info = [float(Satellite_INFO["tl_E"]), float(Satellite_INFO["tl_N"]), float(Satellite_INFO["br_E"]),
                          float(Satellite_INFO["br_N"])]
    drone_in_satellite_relative_position = [float(Satellite_INFO["center_distribute_X"]),
                                            float(Satellite_INFO["center_distribute_Y"])]
    mapsize = float(Satellite_INFO["map_size"])
    # pred GPS info
    pred_N = Satellite_GPS_info[1] - S_X * ((Satellite_GPS_info[1] - Satellite_GPS_info[3]) / opt.Satellitehw[0])
    pred_E = Satellite_GPS_info[0] + S_Y * ((Satellite_GPS_info[2] - Satellite_GPS_info[0]) / opt.Satellitehw[1])
    pred_GPS_info = [pred_E, pred_N]
    # calc euclidean Distance between pred and gt
    distance = euclideanDistance(drone_GPS_info, [pred_GPS_info])
    # json_output pred GPS and groundtruth GPS for save
    GPS_output_dict = {}
    GPS_output_dict["GT_GPS"] = drone_GPS_info
    GPS_output_dict["Pred_GPS"] = pred_GPS_info
    GPS_output_dict["UAV_filename"] = UAV_image_path
    GPS_output_dict["Satellite_filename"] = Satellite_image_path
    GPS_output_dict["mapsize"] = mapsize
    GPS_output_dict["drone_in_satellite_relative_position"] = drone_in_satellite_relative_position
    GPS_output_dict["Satellite_GPS_info"] = Satellite_GPS_info
    GPS_output_list.append(GPS_output_dict)
    SDM_single_score = SDM_evaluateSingle(distance, 1)
    return SDM_single_score

def norm_feat(opt,ff):
    # norm feature
    if len(ff.shape) == 3:
        # feature size (n,2048,6)
        # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.block)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
    else:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
    return ff

GPS_output_list = []
def test(model, dataloader, opt):
    total_score = 0.0
    total_score_b = 0.0
    flag_bias = 0
    start_time = time.time()
    SDM_scores = 0
    for uav, satellite, X, Y, UAV_image_path, Satellite_image_path, UAV_GPS, Satellite_INFO,pos_infos in tqdm(dataloader):
        z = uav.cuda()
        x = torch.cat(satellite,dim=0).cuda()
        z_feat = model(z,None)[0].data.cpu()
        x_feat = torch.FloatTensor()
        for x_single in x:
            x_single = x_single.unsqueeze(0)
            x_feat_single = model(None,x_single)[1].data.cpu()
            x_feat = torch.cat((x_feat, x_feat_single), 0)

        z_feat = norm_feat(opt,z_feat)
        x_feat = norm_feat(opt,x_feat)
        x_feat = x_feat.transpose(1,0)
        scores = torch.mm(z_feat,x_feat).detach().numpy()
        max_ind = np.argmax(scores)
        pred_pos = pos_infos[max_ind]
        X_rate,Y_rate = pred_pos
        S_X,S_Y = int(X_rate*opt.Satellitehw[0]),int(Y_rate*opt.Satellitehw[1])
        label_XY = np.array([X.squeeze().detach().numpy(), Y.squeeze().detach().numpy()])
        pred_XY = np.array([S_X, S_Y])

        loc_bias = None



        # response, loc_bias = model(z, x)
        # response = torch.sigmoid(response)
        # map = response.squeeze().cpu().detach().numpy()
        #
        # # kernel = np.ones((opt.filterR, opt.filterR), np.float32)
        # # hanning kernel
        # kernel = create_hanning_mask(opt.filterR)
        # map = cv2.filter2D(map, -1, kernel)
        #
        # label_XY = np.array([X.squeeze().detach().numpy(), Y.squeeze().detach().numpy()])
        #
        # satellite_map = cv2.resize(map, opt.Satellitehw)
        # id = np.argmax(satellite_map)
        # S_X = int(id // opt.Satellitehw[0])
        # S_Y = int(id % opt.Satellitehw[1])
        # pred_XY = np.array([S_X, S_Y])

        # calculate SDM1 critron
        SDM_single_score = SDM_evaluate_score(opt, UAV_GPS, Satellite_INFO, UAV_image_path, Satellite_image_path, S_X, S_Y)
        # SDM score
        SDM_scores+=SDM_single_score
        # RDS score
        single_score = evaluate(opt, pred_XY=pred_XY, label_XY=label_XY)
        total_score += single_score
        if loc_bias is not None:
            flag_bias = 1
            loc = loc_bias.squeeze().cpu().detach().numpy()
            id_map = np.argmax(map)
            S_X_map = int(id_map // map.shape[-1])
            S_Y_map = int(id_map % map.shape[-1])
            pred_XY_map = np.array([S_X_map, S_Y_map])
            pred_XY_b = (pred_XY_map + loc[:, S_X_map, S_Y_map]) * opt.Satellitehw[0] / loc.shape[-1]  # add bias
            pred_XY_b = np.array(pred_XY_b)
            single_score_b = evaluate(opt, pred_XY=pred_XY_b, label_XY=label_XY)
            total_score_b += single_score_b

    # print("pred: " + str(pred_XY) + " label: " +str(label_XY) +" score:{}".format(single_score))

    time_consume = time.time() - start_time
    print("time consume is {}".format(time_consume))

    score = total_score / len(dataloader)
    SDM_score = SDM_scores / len(dataloader)
    print("the final RDS score is {}".format(score))
    print("the final SDM score is {}".format(SDM_score))
    if flag_bias:
        score_b = total_score_b / len(dataloader)
        print("the final score_bias is {}".format(score_b))

    with open(opt.savename, "w") as F:
        F.write("the final score is {}\n".format(score))
        F.write("the SDM score is {}\n".format(SDM_score))
        F.write("time consume is {}".format(time_consume))

    with open(opt.GPS_output_filename,"w") as F:
        json.dump(GPS_output_list, F, indent=4, ensure_ascii=False)


def main():
    opt = get_opt()
    model = create_model(opt)
    dataloader = create_dataset(opt)
    test(model, dataloader, opt)


if __name__ == '__main__':
    main()
