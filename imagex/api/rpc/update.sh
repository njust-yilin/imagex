#!/bin/bash

# get script path, follow symlink
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd $SCRIPTPATH

python -m grpc_tools.protoc -I protos --python_out=./deepx --pyi_out=./deepx --grpc_python_out=./deepx protos/deepx.proto
python -m grpc_tools.protoc -I protos --python_out=./imagex --pyi_out=./imagex --grpc_python_out=./imagex protos/imagex.proto
python -m grpc_tools.protoc -I protos --python_out=./ui --pyi_out=./ui --grpc_python_out=./ui protos/ui.proto
python -m grpc_tools.protoc -I protos --python_out=./lightx --pyi_out=./lightx --grpc_python_out=./lightx protos/lightx.proto