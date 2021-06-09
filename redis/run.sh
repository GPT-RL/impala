#!/bin/sh
/usr/local/bin/docker-entrypoint.sh redis-server --daemonize yes
sleep 1

N_DEVICES=$1
cat envs.redis | redis-cli > /dev/null
for i in $(seq 1 $N_DEVICES)
do
    redis-cli RPUSH env-queue 'DONE' > /dev/null
    redis-cli RPUSH device-queue "$((i - 1))" > /dev/null
done

redis-cli