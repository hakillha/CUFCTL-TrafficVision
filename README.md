# CUFCTL-TrafficVision

[Setup on Palmetto](docs/SETUP.md)

TMP1=/home/yingges/Desktop/Research/2018Q1

export PYTHONPATH=${TMP1}/CUFCTL-TrafficVision/models/research:${TMP1}/CUFCTL-TrafficVision/models/research/slim

wget https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip  
unzip protoc-3.2.0-linux-x86_64.zip -d ${HOME}/bin/protoc/  
rm protoc-3.2.0-linux-x86_64.zip  
export PATH=${PATH}:${HOME}/bin/protoc/bin

python python/copy_data.py \  
--in_path=/media/yingges/TOSHIBA\ EXT/datasets/trafficvision/UADETRAC/Insight-MVT_Annotation_Train \  
--out_path=/media/yingges/TOSHIBA\ EXT/datasets/trafficvision/UADETRAC_TFODAPI

python python/create_tfrecords.py \  
--in_path=/media/yingges/TOSHIBA\ EXT/datasets/trafficvision/UADETRAC/Insight-MVT_Annotation_Train \  
--out_path=/media/yingges/TOSHIBA\ EXT/datasets/trafficvision/UADETRAC_TFODAPI

python python/create_tfrecords.py \  
--in_path=/media/yingges/TOSHIBA\ EXT/datasets/trafficvision/UADETRAC \  
--out_path=/media/yingges/TOSHIBA\ EXT/datasets/trafficvision/UADETRAC_TFODAPI

python python/create_tfrecords.py \  
	--in_path=/media/yingges/TOSHIBA\ EXT/datasets/trafficvision/UADETRAC \  
	--occ_ratio_threshold=0.4

# palmetto  
python models/research/object_detection/train.py \
	--logtostderr \
	--pipeline_config_path=data/pretrained/faster_rcnn_resnet50_coco_2018_01_28/faster_rcnn_resnet50_coco.config01 \
	--train_dir=data/checkpoints/faster_rcnn_resnet50_ua_detrac_2018_04_29_01

# local  
python models/research/object_detection/train.py \
	--logtostderr \
	--pipeline_config_path=data/pretrained/faster_rcnn_resnet50_coco_2018_01_28/faster_rcnn_resnet50_coco.config \
	--train_dir=data/checkpoints/faster_rcnn_resnet50_ua_detrac_2018_04_29_01
	
#SORT Implementation
![Alt Txt](https://github.com/hakillha/CUFCTL-TrafficVision/blob/master/2018-05-18_11h48_04.gif)

#Deep SORT implementation

![Alt Txt](https://github.com/hakillha/CUFCTL-TrafficVision/blob/master/2018-05-18_12h55_10.gif)
