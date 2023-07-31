from torchvision import transforms
from .Dataloader_University import Sampler_University, Dataloader_University, train_collate_fn
from .autoaugment import ImageNetPolicy
import torch
from .queryDataset import RotateAndCrop, RandomCrop, RandomErasing


def make_dataset(opt):
    transform_train_list = []
    transform_satellite_list = []
    if "uav" in opt.rr:
        transform_train_list.append(RotateAndCrop(0.5))
    if "satellite" in opt.rr:
        transform_satellite_list.append(RotateAndCrop(0.5))
    transform_train_list += [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]

    transform_satellite_list += [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]

    transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w),
                          interpolation=3),  # Image.BICUBIC
    ]

    if "uav" in opt.ra:
        transform_train_list = transform_train_list + \
            [transforms.RandomAffine(180)]
    if "satellite" in opt.ra:
        transform_satellite_list = transform_satellite_list + \
            [transforms.RandomAffine(180)]

    if "uav" in opt.re:
        transform_train_list = transform_train_list + \
            [RandomErasing(probability=opt.erasing_p)]
    if "satellite" in opt.re:
        transform_satellite_list = transform_satellite_list + \
            [RandomErasing(probability=opt.erasing_p)]

    if "uav" in opt.cj:
        transform_train_list = transform_train_list + \
            [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                    hue=0)]
    if "satellite" in opt.cj:
        transform_satellite_list = transform_satellite_list + \
            [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                    hue=0)]

    if opt.DA:
        transform_train_list = [ImageNetPolicy()] + transform_train_list

    last_aug = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_train_list += last_aug
    transform_satellite_list += last_aug
    transform_val_list += last_aug

    print(transform_train_list)
    print(transform_satellite_list)

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
        'satellite': transforms.Compose(transform_satellite_list)}

    # custom Dataset
    image_datasets = Dataloader_University(
        opt.data_dir, transforms=data_transforms)
    samper = Sampler_University(
        image_datasets, batchsize=opt.batchsize, sample_num=opt.sample_num)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
                                              sampler=samper, num_workers=opt.num_worker, pin_memory=True, collate_fn=train_collate_fn)
    dataset_sizes = {x: len(image_datasets) *
                     opt.sample_num for x in ['satellite', 'drone']}
    class_names = image_datasets.cls_names
    return dataloaders, class_names, dataset_sizes
