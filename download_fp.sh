#!/bin/bash -e
array=(
    "http://www.philippe-fournier-viger.com/spmf/datasets/chess.txt"
    "http://www.philippe-fournier-viger.com/spmf/datasets/mushrooms.txt"
    "http://www.philippe-fournier-viger.com/spmf/datasets/pumsb.txt"
    "http://www.philippe-fournier-viger.com/spmf/datasets/connect.txt"
)

for v in "${array[@]}"
do
  wget -P ./data/fp "$v"
done