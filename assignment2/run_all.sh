#!/bin/bash

DIR="./configs"

# 使用 find 命令查找所有文件，并通过 for 循环读取
for file in $(find "$DIR" -type f); do
    echo "Processing $file"
    uv run python -m plant_pathology.train --config $file
done