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
