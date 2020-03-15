#!/bin/bash

source activate mlflow-env

echo "Python bin:"
which python
echo "Conda environments:"
conda info --envs

python /root/app.py