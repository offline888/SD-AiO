# enter screen session
#screen -r run

#conda activate multirestore

export CUDA_VISIBLE_DEVICES=0

export CUDA_VISIBLE_DEVICES=0&
python /home/yhmi/All_in_one/basicsr/all_in_one_train.py -opt /home/yhmi/All_in_one/options/train/degnet_classification_2.yaml &

wait