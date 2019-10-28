
$epoch=0
$prefix="cityscapes/"
$backbone="r50_"
$config="${prefix}mask_rcnn_align_r50_fpn_cityscapes_local"

echo "train"
python ./mmdetection/tools/train.py ./mmdetection/configs/cityscapes/${config}.py

echo "test"
python ./mmdetection/tools/test.py ./mmdetection/configs/cityscapes/${config}.py work_dirs/${config}/latest.pth --out --eval bbox segm
