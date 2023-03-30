output=/home/biao/MORE_
CUDA_VISIBLE_DEVICES="0,1,3,5" \
    python -m torch.distributed.run --nproc_per_node=4 $output/src/tasks/more.py \
    --train train,nominival --valid valid  \
    --datapath /home/biao/data/   \
    --batchSize 4 --optim bert --lr 5e-6 --epochs 4 \
    --tqdm --output $output/snap/more/more_lxr955_tiny/ \
    --multiGPU --tiny > $output/log/train`date +'%Y-%m-%d'`.log 2>&1 &