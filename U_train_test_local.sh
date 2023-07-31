name="U-VitS-Tr"
data_dir="/home/dmmm/Dataset/University-Release/train/"
test_dir="/home/dmmm/Dataset/University-Release/test/"
gpu_ids=1
num_worker=4
lr=0.01
sample_num=1

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --num_worker $num_worker --views 2 --lr $lr --sample_num $sample_num --transformer
# cd checkpoints/$name
# python test_server.py --name $name --test_dir $test_dir --mode 1 --gpu_ids $gpu_ids --num_worker $num_worker

# cd ../../
# name="U-resnetLPN-Tr"
# data_dir="/home/dmmm/University-Release/train/"
# test_dir="/home/dmmm/University-Release/test/"
# gpu_ids=0
# num_worker=4
# lr=0.01
# sample_num=1

# python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --num_worker $num_worker --views 2 --lr $lr --sample_num $sample_num --LPN --block 4
# cd checkpoints/$name
# python test_server.py --name $name --test_dir $test_dir --mode 1 --gpu_ids $gpu_ids --num_worker $num_worker
