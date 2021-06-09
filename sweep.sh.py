#!/usr/bin/env bash

device=$(redis-cli RPOP device-queue)
while [ "$device" == "(nil)" ]
do
  sleep 1
  echo "Waiting for device-queue"
  device=$(redis-cli RPOP device-queue)
done

CUDA_VISIBLE_DEVICES="$device" python src/sweep.py $@