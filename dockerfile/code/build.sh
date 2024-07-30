#!bin/bash
IMAGE=llm_structure_code

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
ROOT_DIR=${CUR_DIR}/../..

mkdir ${CUR_DIR}/tmp
mkdir ${CUR_DIR}/tmp/code
mkdir ${CUR_DIR}/tmp/code/src
mkdir ${CUR_DIR}/tmp/code/utils
mkdir ${CUR_DIR}/tmp/code/log

cp ${ROOT_DIR}/*.py ${CUR_DIR}/tmp/code/
cp ${ROOT_DIR}/*.sh ${CUR_DIR}/tmp/code/
cp -r ${ROOT_DIR}/src ${CUR_DIR}/tmp/code/
cp -r ${ROOT_DIR}/utils ${CUR_DIR}/tmp/code/
cp -r ${ROOT_DIR}/log ${CUR_DIR}/tmp/code/

docker build --no-cache --tag ${IMAGE_URL} ${CUR_DIR}

rm -rf ${CUR_DIR}/tmp

exit 0