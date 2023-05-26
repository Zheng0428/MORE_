python /home/biao/MORE_/test/test_more.py \
--train train \
--datapath /home/biao/MORE_data/atari_data/ \
--loadLXMERT /home/biao/MORE_data/model/model \
--batchSize 2 \
--optim bert \
--lr 5e-7 \
--epochs 2 \
--tqdm \
--tiny \
--llayers 9 \
--xlayers 5 \
--rlayers 5 \
--output /home/biao/MORE_data/model/more_model/ \
--multiGPU \
> /home/biao/MORE_data/log/train`date +'%Y-%m-%d'`.log 2>&1 &