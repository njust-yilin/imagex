#!/bin/bash

# get script path, follow symlink
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

cd $SCRIPTPATH/imagex
python -m grpc_tools.protoc -I ../protos --python_out=. --pyi_out=. --grpc_python_out=. ../protos/imagex.proto