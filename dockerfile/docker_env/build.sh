#!/bin/bash
IMAGE=llm_structure_env

while getopts "t:h" o; do
    case "${o}" in
        t)
            TAG=${OPTARG}
            ;;
        h)
            echo "sh build -t [TAG NAME]"
            exit 1
    esac
done

IMAGE_URL=${IMAGE}:${TAG}
CUR_DIR=$(cd `dirname $0` && pwd -P)

docker build --network=host --no-cache --tag ${IMAGE_URL} ${CUR_DIR}