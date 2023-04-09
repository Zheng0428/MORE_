output=/home/biao/MORE_
CUDA_VISIBLE_DEVICES="2,3,4,5" \
    python -m torch.distributed.run --nproc_per_node=4 $output/src/tasks/more.py \
    --train train --valid valid  \
    --datapath /home/biao/data/   \
    --batchSize 4 --optim bert --lr 6e-6 --epochs 5 \
    --tqdm --output $output/snap/more/more_lxr955_tiny/ \
    --multiGPU --tiny > $output/log/train`date +'%Y-%m-%d'`.log 2>&1 &