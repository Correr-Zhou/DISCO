box_noise_level=0.2

loss_lambda=0.01
loss_gamma=0.3
temper_coef=0.01
mix_a=10
mix_b=0.7

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch  \
	--nproc_per_node=4  \
	--master_port=11500 \
    ./tools/train.py \
    ./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_disco.py \
    --work-dir ./outputs/COCO/COCO-noise${box_noise_level} \
    --launcher pytorch \
    --loss_lambda ${loss_lambda} \
    --loss_gamma ${loss_gamma} \
    --temper_coef ${temper_coef} \
    --mix_a ${mix_a} \
    --mix_b ${mix_b} \
    --cfg-options \
    box_noise_level=${box_noise_level} \
    data.train.box_noise_level=${box_noise_level} \
    data.val_n.box_noise_level=${box_noise_level} \
