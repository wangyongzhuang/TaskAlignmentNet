# 0.0 deps
# TODO, depends to the docker

# 1.0 setup
python mmcv/setup.py develop
python mmdetection/setup.py develop
# 1.1 download pretrained backbone from https://download.pytorch.org/models/resnet50-19c8e357.pth to s3://bucket-8280/wyz/model_zoo?
# TODO copy from s3://bucket-8280/wyz/model_zoo to /cache

# 2.0 prepare dataset
python prepare_coco.py


# 3.0 train and test

# multi-train
#python ./mmdetection/tools/dist_train.py ./mmdetection/configs/mask_rcnn_align_r50_fpn_1x.py 8

python -m torch.distributed.launch --nproc_per_node=8 ./mmdetection/tools/train.py ./mmdetection/configs/mask_rcnn_align_r50_fpn_1x.py --launcher pytorch

# multi-test
cd 
python -m torch.distributed.launch --nproc_per_node=8 ./mmdetection/tools/test.py configs/mask_rcnn_align_r50_fpn_1x.py  /cache/mask_rcnn_align_r50_fpn_1x/latest.pth --launcher pytorch --out results.pkl --eval bbox segm

#python ./mmdetection/tools/test.py configs/mask_rcnn_align_r50_fpn_1x.py /cache/mask_rcnn_align_r50_fpn_1x/latest.pth 8 --out results.pkl --eval bbox segm


# single-train
#python ./mmdetection/tools/train.py ./mmdetection/configs/mask_rcnn_align_r50_fpn_1x.py 
# single-test
#python ./mmdetection/tools/test.py configs/mask_rcnn_fpn_1x.py /cache/mask_rcnn_align_r50_fpn_1x/latest.pth --out results.pkl --eval bbox segm

# for debug
# backbone:    https://download.pytorch.org/models/resnet50-19c8e357.pth
# img_per_gpu: 2->1
# sample:      512->128
# batch:       8->1
# lr:          0.02->0.00125
# epoch:       12->1
# annotations: instances_train2017.json->instances_train2017_sampled_0.01.json
#              instances_val2017.json->instances_val2017_sampled_0.01.json
# for htc:     delete 'gt_semantic_seg' from train_pipeline, with_seg=False, delete 'seg_prefix'
