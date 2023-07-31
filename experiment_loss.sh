name="Loss_Experiment-CELoss-WeightedSoftTripletLoss-KLLoss"
data_dir="/home/dmmm/Dataset/DenseUAV/data_2022/train" #"/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/data_2021/train"
# data_dir="/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/data_2021/train"
test_dir="/home/dmmm/Dataset/DenseUAV/data_2022/test" #"/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/data_2021/test"
# test_dir="/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/data_2021/test"
gpu_ids=0
num_worker=8
lr=0.003
batchsize=16
sample_num=1
block=1
num_bottleneck=512
backbone="ViTS-224"
head="SingleBranch"
cls_loss="CELoss" # CELoss FocalLoss
feature_loss="WeightedSoftTripletLoss" # TripletLoss HardMiningTripletLoss WeightedSoftTripletLoss
kl_loss="no" # KLLoss
h=224
w=224
load_from="no"
ra="satellite"  # random affine
re="satellite"  # random erasing
cj="no"  # color jitter
rr="uav"  # random rotate

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --block $block --lr $lr --num_worker $num_worker --head $head  \
                --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss


cd checkpoints/$name
python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
python evaluate_gpu.py
python evaluateDistance.py
cd ../../


name="Loss_Experiment-CELoss-WeightedSoftTripletLoss_alpha1-KLLoss"
data_dir="/home/dmmm/Dataset/DenseUAV/data_2022/train" #"/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/data_2021/train"
# data_dir="/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/data_2021/train"
test_dir="/home/dmmm/Dataset/DenseUAV/data_2022/test" #"/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/data_2021/test"
# test_dir="/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/data_2021/test"
gpu_ids=0
num_worker=8
lr=0.003
batchsize=16 
cls_loss="CELoss" # CELoss FocalLoss
feature_loss="WeightedSoftTripletLoss" # TripletLoss HardMiningTripletLoss WeightedSoftTripletLoss
kl_loss="KLLoss" # KLLoss
h=224
w=224
load_from="no"
ra="satellite"  # random affine
re="satellite"  # random erasing
cj="no"  # color jitter
rr="uav"  # random rotate

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --block $block --lr $lr --num_worker $num_worker --head $head  \
                --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss


cd checkpoints/$name
python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
python evaluate_gpu.py
python evaluateDistance.py
cd ../../