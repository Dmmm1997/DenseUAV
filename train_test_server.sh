#name="from_transreid_256_base_lr01_tripletloss"
name="from_transreid_256_small_lr01_tr1x_heatbranch"
data_dir="/home/dmmm/University-Release/train"
test_dir="/home/dmmm/University-Release/test"
gpu_ids=0
num_worker=4
lr=0.01
sample_num=1
block=4

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --num_worker $num_worker --views 2 --lr $lr --sample_num $sample_num --block $block
cd checkpoints/$name
for((i=119;i<=120;i+=10));
do
  for ((j = 1; j < 3; j++));
  do
      python test_server.py --name $name --test_dir $test_dir --checkpoint net_$i.pth --mode $j --gpu_ids $gpu_ids --num_worker $num_worker
  done
done