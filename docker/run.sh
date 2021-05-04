#!/bin/sh

NAME=$1
PORT=$2
DATASETS_PATH=$3

docker container stop -t 0 $NAME

USER_NAME=$(basename $HOME)
echo "Run as user '$USER_NAME'"

HOST_PATH=$(readlink -f "$PWD/../../")
DOCKER_PATH="/root/$NAME"

cd $HOST_PATH

(docker container run \
    --rm \
    -dit \
    --dns 217.10.39.4 --dns 8.8.8.8 \
    --privileged \
    -v $HOST_PATH:/$DOCKER_PATH \
    -v $DATASETS_PATH:$DATASETS_PATH \
    -v $HOME:/home/$USER_NAME \
    --expose $PORT \
    -p $PORT:$PORT \
    -h $NAME \
    --name $NAME \
    $NAME) || true

echo Hello1
docker container exec -it -w $DOCKER_PATH $NAME bash -c \
    "cd /home/ && \
    jupyter notebook --port=${PORT} --ip=0.0.0.0 --no-browser --allow-root && \
    bash"
echo Hello2

