# Setup on Clemson's Palmetto Cluster

*Note: Setup should only need to occur once.  Skip this step if you have already setup everything.*

Before being able to train, you will need to setup an anaconda environment with the appropriate packages as well as the manual install of [Google's Protocol Buffers](https://github.com/google/protobuf).

Retrieve a node with a GPU

	qsub -I -v cores=20 -l select=1:ncpus=20:mem=16gb:ngpus=2:gpu_model=k40,walltime=48:00:00
	qsub -I -v cores=20 -l select=1:ncpus=20:mem=16gb:ngpus=2:gpu_model=p100,walltime=48:00:00

Add necessary modules

	# Note: these can be added to the .bashrc (so as not to have to add everytime)
	module add anaconda3/4.3.0
	module add cuda-toolkit/8.0.44
	module add cuDNN/8.0v6

Add the following paths to your .bashrc file:

	export PATH=${PATH}:${HOME}/bin/protoc/bin
	export PYTHONPATH=${PYTHONPATH}:${HOME}/CUFCTL-Track/models/research:${HOME}/CUFCTL-Track/models/research/slim

Source the .bashrc file:
	
	source ~/.bashrc

Build the Protobuf Compiler (protoc)

	# From CUFCTL-Track directory
	sh scripts/buildProtoc.sh

Create Conda environment w/ necessary packages

	# From CUFCTL-Track directory
	sh scripts/createCondaEnv.sh

Activate the Conda Environment

	source activate dac

Compile the Protobuf libraries

	# From CUFCTL-Track/models/research directory
	protoc object_detection/protos/*.proto --python_out=.
	# NOTE: This should produce no output if everything is correct.

Test to make sure everything is installed correctly

	# From CUFCTL-Track/models/research directory
	python object_detection/builders/model_builder_test.py

The result should be ...

	.......
	----------------------------------------------------------------------
	Ran 7 tests in 0.044s

	OK