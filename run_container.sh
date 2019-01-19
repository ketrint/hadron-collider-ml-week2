#! /bin/bash
docker run -it --name container --net=host -v $PWD:/ds hadron-collider-ml-week2
