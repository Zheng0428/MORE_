output=/home/biao/MORE_
CUDA_VISIBLE_DEVICES="2,3,5,6" \
    python -m torch.distributed.run --nproc_per_node=4 $output/src/tasks/more.py \
    --train train,nominival --valid valid  \
    --datapath /home/biao/data/   \
    --batchSize 4 --optim bert --lr 5e-7 --epochs 4 \
    --tqdm --output $output/snap/more/more_lxr955_tiny/ \
    --multiGPU  > $output/log/train`date +'%Y-%m-%d'`.log 2>&1 &