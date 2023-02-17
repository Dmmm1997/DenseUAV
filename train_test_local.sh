name="***DeitS-B1-Tr-R-1"
data_dir="/media/dmmm/CE31-3598/DataSets/DenseCV_Data/高度数据集/newdata/train"
test_dir="/media/dmmm/CE31-3598/DataSets/DenseCV_Data/高度数据集/newdata/test/"
gpu_ids=0
num_worker=8
lr=0.01
sample_num=1
rotate=1
transformer=1
triplet_loss=1
block=2
LPN=0
WSTR=0
deit=1

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num --rotate $rotate \
--transformer $transformer --triplet_loss $triplet_loss --block $block --LPN $LPN --WSTR $WSTR --lr $lr --deit $deit
cd checkpoints/$name
python test_server.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
cd ../../
