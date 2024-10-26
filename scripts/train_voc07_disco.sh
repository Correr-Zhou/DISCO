box_noise_level=0.4

loss_lambda=0.1
loss_gamma=0.1
temper_coef=0.1
mix_a=5
mix_b=0.8

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch  \
	--nproc_per_node=2  \
	--master_port=18500 \
    ./tools/train.py \
    ./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc_disco.py \
    --work-dir ./outputs/VOC07/VOC07-noise${box_noise_level} \
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
