# CUFCTL-TrafficVision

## Installation
[Installation](docs/INSTALLATION.md)

## Data Conversion
Right now label map needs to be provided as an input instead of being generated in the process.
```
# Example
# Run with --help for more information on options
python python/create_tfrecords.py \
    --in_path=path/to/ua_detrac \
```
## Training
Download pretrained models:

Training:
```
python models/research/object_detection/train.py \
	--logtostderr \
	--pipeline_config_path=${PIPELINE_CONFIG} \
	--train_dir=${TRAIN_DIR}
```
## Evaluation
```
python models/research/object_detection/eval.py \
	--logtostderr \
	--pipeline_config_path=${PIPELINE_CONFIG} \
	--checkpoint_dir=${TRAIN_DIR} \
	--eval_dir=${EVAL_DIR}
```
## Inference
Frozen graph exportation:
```
python models/research/object_detection/export_inference_graph.py \
	--pipeline_config_path=${PIPELINE_CONFIG} \
	--trained_checkpoint_prefix=${TRAIN_PATH} \
	--output_directory=${OUTPUT_DIR}
```
Inference:
```
# Example
# Run with --help for more information on options
python inference101.py \
    --model_dir=data/checkpoints/ssd_mobilenet_v2 \
    --test_data_dir=path/to/ua_detrac/Insight-MVT_Annotation_Test \
    --video_name=MVI_40714
```
Create output video from detection output images:
```
# Example
python python/create_video.py \
    --image_dir=data/inference_output/MVI_40714 \
    --video_name=video01.avi
```
