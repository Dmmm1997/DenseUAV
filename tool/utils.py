import os
import torch
import yaml
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
from shutil import copyfile, copytree, rmtree
import logging
from models.taskflow import make_model
from thop import profile, clever_format
import math


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def copy_file_or_tree(path, target_dir):
    target_path = os.path.join(target_dir, path)
    if os.path.isdir(path):
        if os.path.exists(target_path):
            rmtree(target_path)
        copytree(path, target_path)
    elif os.path.isfile(path):
        copyfile(path, target_path)


def copyfiles2checkpoints(opt):
    dir_name = os.path.join('checkpoints', opt.name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    # record every run
    copy_file_or_tree('train.py', dir_name)
    copy_file_or_tree('test.py', dir_name)
    copy_file_or_tree('evaluate_gpu.py', dir_name)
    copy_file_or_tree('evaluateDistance.py', dir_name)
    copy_file_or_tree('datasets', dir_name)
    copy_file_or_tree('losses', dir_name)
    copy_file_or_tree('models', dir_name)
    copy_file_or_tree('optimizers', dir_name)
    copy_file_or_tree('tool', dir_name)
    copy_file_or_tree('train_test_local.sh', dir_name)

    # save opts
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# Get model list for resume


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s' % dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

######################################################################
# Save model
# ---------------------------


def save_network(network, dirname, epoch_label):
    if not os.path.isdir('./checkpoints/'+dirname):
        os.mkdir('./checkpoints/'+dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth' % epoch_label
    else:
        save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./checkpoints', dirname, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def check_box(images, boxes):
    # Unorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # images = Unorm(images)*255
    images = images.permute(0, 2, 3, 1).cpu().detach().numpy()
    boxes = (boxes.cpu().detach().numpy()/16*255).astype(np.int)
    for img, box in zip(images, boxes):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(img)
        rect = plt.Rectangle(box[0:2], box[2]-box[0], box[3]-box[1])
        ax.add_patch(rect)
        plt.show()


######################################################################
#  Load model for resume
# ---------------------------
def load_network(opt):
    save_filename = opt.checkpoint
    model = make_model(opt)
    # print('Load the model from %s' % save_filename)
    network = model
    network.load_state_dict(torch.load(save_filename))
    return network


def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    toogle_grad(model_src, True)


def get_preds(outputs, outputs2):
    if isinstance(outputs, list):
        preds = []
        preds2 = []
        for out, out2 in zip(outputs, outputs2):
            preds.append(torch.max(out.data, 1)[1])
            preds2.append(torch.max(out2.data, 1)[1])
    else:
        _, preds = torch.max(outputs.data, 1)
        _, preds2 = torch.max(outputs2.data, 1)
    return preds, preds2


def calc_flops_params(model,
                      input_size_drone,
                      input_size_satellite,
                      ):
    inputs_drone = torch.randn(input_size_drone).cuda()
    inputs_satellite = torch.randn(input_size_satellite).cuda()
    total_ops, total_params = profile(
        model, (inputs_drone, inputs_satellite,), verbose=False)
    macs, params = clever_format([total_ops, total_params], "%.3f")
    return macs, params


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    random.seed(seed)


def Distance(lata, loga, latb, logb):
    # EARTH_RADIUS = 6371.0
    EARTH_RADIUS = 6378.137
    PI = math.pi
    # // 转弧度
    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    a = lat_a - lat_b
    b = loga * PI / 180 - logb * PI / 180
    dis = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat_a)
                                  * math.cos(lat_b) * math.pow(math.sin(b / 2), 2)))

    distance = EARTH_RADIUS * dis * 1000
    return distance