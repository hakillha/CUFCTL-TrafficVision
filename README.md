# CUFCTL-TrafficVision

TMP1=/home/yingges/Desktop/Research/2018Q1

export PYTHONPATH=${TMP1}/CUFCTL-TrafficVision/models/research:${TMP1}/CUFCTL-TrafficVision/models/research/slim

wget https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip  
unzip protoc-3.2.0-linux-x86_64.zip -d ${HOME}/bin/protoc/  
rm protoc-3.2.0-linux-x86_64.zip  
export PATH=${PATH}:${HOME}/bin/protoc/bin

python python/move_data.py \  
--in_path=/media/yingges/TOSHIBA\ EXT/datasets/trafficvision/UADETRAC \  
--out_path=/media/yingges/TOSHIBA\ EXT/datasets/trafficvision/UADETRAC_TFODAPI
