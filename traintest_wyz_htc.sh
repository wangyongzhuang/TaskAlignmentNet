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
python ./mmdetection/tools/dist_train.py ./mmdetection/configs/htc_align_r50_fpn_1x.py 8

# multi-test
python ./mmdetection/tools/test.py configs/htc_align_r50_fpn_1x.py /cache/htc_align_r50_fpn_1x/latest.pth 8 --out results.pkl --eval bbox segm


