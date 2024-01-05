data_dir="/home/dmmm/Dataset/DenseUAV/data_2022/train" #"/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/data_2021/train"
# data_dir="/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/data_2021/train"
test_dir="/home/dmmm/Dataset/DenseUAV/data_2022/test" #"/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/data_2021/test"
# test_dir="/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/data_2021/test"
num_worker=8
gpu_ids=0

name="checkpoints/Backbone_Experiment_SENet"
cd $name
cd tool
python get_inference_time.py --name $name
cd ../../../

# name="checkpoints/Backbone_Experiment_ConvnextT"
# cd $name
# cd tool
# python get_inference_time.py --name $name
# cd ../../../

# name="checkpoints/Backbone_Experiment_DeitS"
# cd $name
# cd tool
# python get_inference_time.py --name $name
# cd ../../../

# name="checkpoints/Backbone_Experiment_EfficientNet-B2"
# cd $name
# cd tool
# python get_inference_time.py --name $name
# cd ../../../

# name="checkpoints/Backbone_Experiment_EfficientNet-B3"
# cd $name
# cd tool
# python get_inference_time.py --name $name
# cd ../../../

# name="checkpoints/Backbone_Experiment_PvTv2b2"
# cd $name
# cd tool
# python get_inference_time.py --name $name
# cd ../../../

# name="checkpoints/Backbone_Experiment_resnet50"
# cd $name
# cd tool
# python get_inference_time.py --name $name
# cd ../../../

# name="checkpoints/Backbone_Experiment_Swinv2T-256"
# cd $name
# cd tool
# python get_inference_time.py --name $name --test_h 256 --test_w 256
# cd ../../../

# name="checkpoints/Backbone_Experiment_VGG16"
# cd $name
# cd tool
# python get_inference_time.py --name $name
# cd ../../../

# name="checkpoints/Backbone_Experiment_ViTB"
# cd $name
# cd tool
# python get_inference_time.py --name $name
# cd ../../../

# name="checkpoints/Head_Experiment-FSRA2B"
# cd $name
# cd tool
# python get_inference_time.py --name $name
# cd ../../../

# name="checkpoints/Head_Experiment-FSRA3B"
# cd $name
# cd tool
# python get_inference_time.py --name $name
# cd ../../../

# name="checkpoints/Head_Experiment-GeM"
# cd $name
# cd tool
# python get_inference_time.py --name $name
# cd ../../../

# name="checkpoints/Head_Experiment-LPN2B"
# cd $name
# cd tool
# python get_inference_time.py --name $name
# cd ../../../

# name="checkpoints/Head_Experiment-LPN3B"
# cd $name
# cd tool
# python get_inference_time.py --name $name
# cd ../../../
