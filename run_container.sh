#! /bin/bash
CONT_LOC=/Users/katetrofimova/Downloads/Docker_Tutorial-master/basic_tutorial/
docker run -it --name container --net=host -v $CONT_LOC:/ds hadron-collider-ml-week2
