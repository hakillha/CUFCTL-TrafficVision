#!/bin/sh

TRAFFICVISION=/home/yingges/Desktop/Research/2018Q1/CUFCTL-TrafficVision
cd ..
# python python/test/test01.py
python models/research/object_detection/train.py \
	--logtostderr \
	--pipeline_config_path=${TRAFFICVISION}/data/pretrained/faster_rcnn_resnet50_coco_2018_01_28/pipeline_02.config \
	--train_dir=${TRAFFICVISION}/data/checkpoints/test01