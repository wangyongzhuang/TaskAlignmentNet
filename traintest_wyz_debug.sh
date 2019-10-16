# 0.0 deps
# TODO, depends to the docker

# 1.0 setup
#python mmcv/setup.py develop
#python mmdetection/setup.py develop
# 1.1 download pretrained backbone from https://download.pytorch.org/models/resnet50-19c8e357.pth to s3://bucket-8280/wyz/model_zoo?
# TODO copy from s3://bucket-8280/wyz/model_zoo to /cache

# 2.0 prepare dataset
#python prepare_coco.py


# 3.0 train and test

# single-train
python ./mmdetection/tools/train.py ./mmdetection/configs/mask_rcnn_align_r50_fpn_1x_debug.py

# single-test
python ./mmdetection/tools/test.py /home/wyz/mmdet/mmdetection/configs/mask_rcnn_align_r50_fpn_1x_debug.py /home/wyz/mmdet/work_dirs/mask_rcnn_align_r50_fpn_1x_debug/latest.pth --out results.pkl --eval bbox segm


