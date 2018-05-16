# Installation
## Local Machine
### Dependencies
* Placeholder
### Steps
Set up environment variable
```
export PYTHONPATH=${PYTHONPATH}:path/to/CUFCTL-TrafficVision/models/research
export PYTHONPATH=${PYTHONPATH}:path/to/CUFCTL-TrafficVision/models/research/slim
```

Build Protobuf compiler

Compile the Protobuf files
```
# From CUFCTL-TrafficVision/models/research directory
protoc object_detection/protos/*.proto --python_out=.
```

Test run TFODAPI
```
# From CUFCTL-TrafficVision/models/research directory
python object_detection/builders/model_builder_test.py
```

## Palmetto
Add necessary modules:
```
module add anaconda3/4.3.0
module add cuda-toolkit/9.0.176
module add cuDNN/9.0v7
module add opencv/2.4.9
```

Activate the Conda environment:
```
source activate trafficv
```
