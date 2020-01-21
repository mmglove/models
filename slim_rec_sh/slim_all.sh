#!/bin/bash
"""
Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
This module run slim model

Authors: guomengmeng01(guomengmeng01@baidu.com)
Date: 2019/8/21 14:29
"""
#set param
#current_dir=/workspace
#fluid_path=/workspace
#data_path=/ssd1/xiege/data
#paddle_father_path=/ssd1/xiege/paddle_ce
#log_father_path=/ssd1/xiege
#result_path=/workspace/result
#log_path=/ssd1/xiege/logs
#______________________________________
#49
#current_dir=/paddle/slim16/auto_0107
#fluid_path=/paddle/slim16/auto_0107/models
#data_path=/paddle/all_data/slim
#log_father_path=/paddle/slim16/auto_0107
#result_path=/paddle/slim16/auto_0107/result
#log_path=/paddle/slim16/auto_0107/logs
# 48
current_dir=/ssd3/guomengmeng01/slim/auto_0115
fluid_path=/ssd3/guomengmeng01/slim/auto_0115/models
data_path=/paddle/all_data/slim
log_father_path=/ssd3/guomengmeng01/slim/auto_0115
result_path=/ssd3/guomengmeng01/slim/auto_0115/result
log_path=/ssd3/guomengmeng01/slim/auto_0115/logs
#set result dir
cd ${current_dir}
if [ ! -d "result" ];then
	mkdir result
fi
result_path=${current_dir}"/result"
cd ${result_path}
if [ -d "result.log" ];then
	rm -rf result.log
fi
#set log dir
cd ${log_father_path}
if [ -d "logs" ];then
    rm -rf logs
fi
mkdir logs && cd logs
mkdir SUCCESS
mkdir FAIL
log_path=${log_father_path}"/logs"
#提交时务必删掉下划线之间的，否则log会被重置
#______________________________________
# paddleslim
cd ${fluid_path}/PaddleSlim
#if ILSVRC2012 does not exist in folder Paddleslim/data,Run the following;
mkdir data
cd data
if [ -d "ILSVRC2012" ];then rm -rf ILSVRC2012
fi
#ln -s ${data_path}/ILSVRC2012 ILSVRC2012
#48
ln -s ${data_path}/ILSVRC2012_data/ILSVRC2012 ILSVRC2012

cd ${fluid_path}/PaddleCV/PaddleDetection/dataset
if [ -d "voc" ];then rm -rf voc
fi
#13
#ln -s ${data_path}/pascalvoc voc
#48
ln -s /ssd3/xiege/data/pascalvoc voc
cd -

# add yaml
cd ${fluid_path}/PaddleSlim
if [ -d "configs" ];then
    rm -rf configs
fi
#13
#ln -s ${data_path}/PaddleSlim/configs_13 configs
#ln -s ${data_path}/configs_13 configs
#48
ln -s /ssd3/guomengmeng01/all_data/slim/configs_13
# add light_nas yaml
cd ${fluid_path}/PaddleSlim/light_nas
rm -rf compress.yaml
cp ${data_path}/light_nas_configs/* ./
cp ${data_path}/PaddleSlim/light_nas_configs/* ./

# download pretrain model
pip install wget
pip install pycocotools

root_url="http://paddle-imagenet-models-name.bj.bcebos.com"
dete_root_url="https://paddlemodels.bj.bcebos.com/object_detection"

MobileNetV1="MobileNetV1_pretrained.tar"
ResNet50="ResNet50_pretrained.tar"
GoogleNet="GoogLeNet_pretrained.tar"
MobileNetV2="MobileNetV2_pretrained.tar"
ResNet34="ResNet34_pretrained.tar"
yolov3_r34_voc="yolov3_r34_voc.tar"
pretrain_dir=${fluid_path}/PaddleSlim/pretrain
calss_pretrain_dir=${fluid_path}/PaddleSlim/classification/pretrain
cd ${fluid_path}/PaddleSlim
if [ ! -d ${pretrain_dir} ]; then
  mkdir ${pretrain_dir}
fi

cd ${pretrain_dir}

if [ ! -f ${MobileNetV1} ]; then
    wget ${root_url}/${MobileNetV1}
    tar xf ${MobileNetV1}
fi

if [ ! -f ${ResNet50} ]; then
    wget ${root_url}/${ResNet50}
    tar xf ${ResNet50}
fi

if [ ! -f ${GoogleNet} ]; then
    wget ${root_url}/${GoogleNet}
    tar xf ${GoogleNet}
fi

cd -
cd ${fluid_path}/PaddleSlim/classification/
if [ ! -d ${calss_pretrain_dir} ]; then
    mkdir ${calss_pretrain_dir}
fi
cd ${calss_pretrain_dir}
if [ ! -f ${MobileNetV1} ]; then
    wget ${root_url}/${MobileNetV1}
    tar xf ${MobileNetV1}
fi
if [ ! -f ${MobileNetV2} ]; then
    wget ${root_url}/${MobileNetV2}
    tar xf ${MobileNetV2}
fi

if [ ! -f ${ResNet50} ]; then
    wget ${root_url}/${ResNet50}
    tar xf ${ResNet50}
fi

if [ ! -f ${ResNet34} ]; then
    wget ${root_url}/${ResNet34}
    tar xf ${ResNet34}
fi

# enable GC strategy
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0

# slim_1 for uniform filter pruning MobileNetV1
##-------------------------------------------------
cd ${fluid_path}/PaddleSlim
export CUDA_VISIBLE_DEVICES=0
yaml_name='filter_pruning_uniform'
model=slim_1_${yaml_name}
sed -i "s/epoch: 200/epoch: 1/g" configs/${yaml_name}.yaml
rm -rf ${model}_checkpoints
rm -rf checkpoints
time (python -u compress.py \
        --model MobileNet \
        --pretrained_model ./pretrain/MobileNetV1_pretrained \
        --compress_config ./configs/${yaml_name}.yaml >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
mv checkpoints ${model}_checkpoints

## slim_2 for sen filter pruning MobileNetV1 models='ResNet50,MobileNetV1'
##-------------------------------------------------
cd ${fluid_path}/PaddleSlim
export CUDA_VISIBLE_DEVICES=0
yaml_name='filter_pruning_sen'
model=slim_2_${yaml_name}
sed -i "s/epoch: 200/epoch: 1/g" configs/${yaml_name}.yaml
rm -rf ${model}_checkpoints
rm -rf checkpoints
time (python -u compress.py \
        --model MobileNet \
        --pretrained_model ./pretrain/MobileNetV1_pretrained \
        --compress_config ./configs/${yaml_name}.yaml >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
mv checkpoints ${model}_checkpoints

## slim_3_1 for auto filter pruning search
##-------------------------------------------------------------------
cd ${fluid_path}/PaddleSlim
export CUDA_VISIBLE_DEVICES=0
yaml_name='auto_prune'
model=slim_3_1_search_${yaml_name}
sed -i "s/epoch: 500/epoch: 1/g" configs/${yaml_name}.yaml
sed -i "s/end_epoch: 500/end_epoch: 1/g" configs/${yaml_name}.yaml
rm -rf ${model}_checkpoints
rm -rf checkpoints
time (python compress.py \
        --model "MobileNet" \
        --pretrained_model ./pretrain/MobileNetV1_pretrained \
        --compress_config ./configs/${yaml_name}.yaml >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
mv checkpoints_auto_pruning ${model}_checkpoints

## slim_3_2 for auto filter pruning train
##-------------------------------------------------------------------
cd ${fluid_path}/PaddleSlim
export CUDA_VISIBLE_DEVICES=0
yaml_name='auto_prune_train'
model=slim_3_2_${yaml_name}
sed -i "s/epoch: 500/epoch: 1/g" configs/${yaml_name}.yaml
sed -i "s/retrain_epoch: 0/retrain_epoch: 1/g" configs/${yaml_name}.yaml
sed -i "s/end_epoch: 500/end_epoch: 1/g" configs/${yaml_name}.yaml
rm -rf ${model}_checkpoints
rm -rf checkpoints
time (python compress.py \
        --model "MobileNet" \
        --pretrained_model ./pretrain/MobileNetV1_pretrained \
        --compress_config ./configs/${yaml_name}.yaml >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
mv checkpoints_auto_pruning ${model}_checkpoints

## slim_4 for quantization
##----------------------------------------------------------------------
cd ${fluid_path}/PaddleSlim
export CUDA_VISIBLE_DEVICES=0
yaml_name='quantization'
model=slim_4_${yaml_name}
sed -i "s/epoch: 20/epoch: 1/g" configs/${yaml_name}.yaml
sed -i "s/end_epoch: 19/end_epoch: 1/g" configs/${yaml_name}.yaml
rm -rf ${model}_checkpoints
rm -rf checkpoints
time (python compress.py \
        --batch_size 64 \
        --model "MobileNet" \
        --pretrained_model ./pretrain/MobileNetV1_pretrained \
        --compress_config ./configs/${yaml_name}.yaml >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
mv checkpoints_quan ${model}_checkpoints

# slim_5 for uniform filter pruning with quantization
#-----------------------------------------------------------------
cd ${fluid_path}/PaddleSlim
export CUDA_VISIBLE_DEVICES=0
yaml_name='quantization_pruning'
model=slim_5_${yaml_name}
rm -rf ${model}_checkpoints
rm -rf checkpoints
time (python compress.py \
        --model "MobileNet" \
        --pretrained_model ./pretrain/MobileNetV1_pretrained \
        --compress_config ./configs/${yaml_name}.yaml >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
mv checkpoints ${model}_checkpoints

# slim_6 for distillation
#---------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1
#Fixing name conflicts in distillation
cd ${pretrain_dir}/ResNet50_pretrained
mv conv1_weights res_conv1_weights
mv fc_0.w_0 res_fc.w_0
mv fc_0.b_0 res_fc.b_0
cd ..
cd ${fluid_path}/PaddleSlim
yaml_name='mobilenetv1_resnet50_distillation'
model=slim_6_${yaml_name}
sed -i "s/epoch: 130/epoch: 1/g" configs/${yaml_name}.yaml
sed -i "s/end_epoch: 130/end_epoch: 1/g" configs/${yaml_name}.yaml
rm -rf ${model}_checkpoints
rm -rf checkpoints
time (python compress.py \
        --model "MobileNet" \
        --teacher_model "ResNet50" \
        --teacher_pretrained_model ./pretrain/ResNet50_pretrained \
        --compress_config ./configs/${yaml_name}.yaml >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
mv checkpoints ${model}_checkpoints
cd ${pretrain_dir}/ResNet50_pretrained
mv res_conv1_weights conv1_weights
mv res_fc.w_0 fc_0.w_0
mv res_fc.b_0 fc_0.b_0
cd -

# slim_7 for distillation with quantization
#---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1
# Fixing name conflicts in distillation
cd ${pretrain_dir}/ResNet50_pretrained
mv conv1_weights res_conv1_weights
mv fc_0.w_0 res_fc.w_0
mv fc_0.b_0 res_fc.b_0
cd ..
cd ${fluid_path}/PaddleSlim
yaml_name='quantization_dist'
model=slim_7_${yaml_name}
rm -rf ${model}_checkpoints
rm -rf checkpoints
time (python compress.py \
        --model "MobileNet" \
        --teacher_model "ResNet50" \
        --teacher_pretrained_model ./pretrain/ResNet50_pretrained \
        --compress_config ./configs/${yaml_name}.yaml >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
mv checkpoints ${model}_checkpoints
cd ${pretrain_dir}/ResNet50_pretrained
mv res_conv1_weights conv1_weights
mv res_fc.w_0 fc_0.w_0
mv res_fc.b_0 fc_0.b_0
cd -

# slim_8_1  Classification model  mobilenet_v1 with quantization
#---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${fluid_path}/PaddleSlim/classification/quantization
yaml_name='mobilenet_v1'
model=slim_8_1_quan_classification_${yaml_name}
sed -i "s/end_epoch: 29/end_epoch: 1/g" configs/${yaml_name}.yaml
sed -i "s/epoch: 30/epoch: 2/g" configs/${yaml_name}.yaml
time (python compress.py \
    --model "MobileNet" \
    --use_gpu 1 \
    --batch_size 256 \
    --pretrained_model ../pretrain/MobileNetV1_pretrained \
    --config_file  "./configs/${yaml_name}.yaml"  > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
# slim_8_2  Classification model mobilenet_v2 with quantization
#---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${fluid_path}/PaddleSlim/classification/quantization
yaml_name='mobilenet_v2'
model=slim_8_2_quan_classification_${yaml_name}
sed -i "s/end_epoch: 29/end_epoch: 1/g" configs/${yaml_name}.yaml
sed -i "s/epoch: 30/epoch: 2/g" configs/${yaml_name}.yaml
time (python compress.py \
    --model "MobileNetV2" \
    --use_gpu 1 \
    --batch_size 256 \
    --pretrained_model ../pretrain/MobileNetV2_pretrained \
    --config_file  "./configs/${yaml_name}.yaml" > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
# slim_8_3  Classification model resnet34 with quantization
#---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${fluid_path}/PaddleSlim/classification/quantization
yaml_name='resnet34'
model=slim_8_3_quan_classification_${yaml_name}
sed -i "s/end_epoch: 29/end_epoch: 1/g" configs/${yaml_name}.yaml
sed -i "s/epoch: 30/epoch: 2/g" configs/${yaml_name}.yaml
time (python -u compress.py \
    --model "ResNet34" \
    --use_gpu 1 \
    --batch_size 32 \
    --pretrained_model ../pretrain/ResNet34_pretrained \
    --config_file "./configs/${yaml_name}.yaml" > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
# classification_quan_freeze and infer
classification_quan_models='mobilenet_v1 mobilenet_v2 resnet34'
i=1
for quan_model in ${classification_quan_models}
do
    cd ${fluid_path}/PaddleSlim/classification/quantization
    model=slim_8_${i}_2_quan_classification_${quan_model}_freeze
    time (python freeze.py \
        --model_path ./checkpoints/${quan_model}/0/eval_model \
        --weight_quant_type  abs_max\
        --save_path ./freeze/${quan_model} >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
    if [ $? -ne 0 ];then
	    mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	    echo -e "${model},freeze,FAIL" >>${result_path}/result.log;
    else
	    mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	    echo -e "${model},freeze,SUCCESS" >>${result_path}/result.log
    fi
    cd ${fluid_path}/PaddleSlim/classification
    model=slim_8_${i}_3_quan_classification_${quan_model}_infer
    time (python infer.py \
        --use_gpu 0 \
        --model_path ./quantization/output/${quan_model}/float \
        --model_name model \
        --params_name weights >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
    if [ $? -ne 0 ];then
	    mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	    echo -e "${model},infer,FAIL" >>${result_path}/result.log;
    else
	    mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	    echo -e "${model},infer,SUCCESS" >>${result_path}/result.log
    fi
    cd ${fluid_path}/PaddleSlim/classification
    model=slim_8_${i}_4_quan_classification_${quan_model}_freeze_infer
    time (python infer.py \
        --use_gpu 0 \
        --model_path ./quantization/freeze/${quan_model}/float \
        --model_name model \
        --params_name weights >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1

    if [ $? -ne 0 ];then
	    mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	    echo -e "${model},freeze_infer,FAIL" >>${result_path}/result.log;
    else
	    mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	    echo -e "${model},freeze_infer,SUCCESS" >>${result_path}/result.log
    fi
    cd ${fluid_path}/PaddleSlim/classification
    model=slim_8_${i}_5_quan_classification_${quan_model}_eval
    time (python eval.py \
        --use_gpu True \
        --model_path ./quantization/output/${quan_model}/float \
        --model_name model \
        --params_name weights >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1

    if [ $? -ne 0 ];then
	    mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	    echo -e "${model},eval,FAIL" >>${result_path}/result.log;
    else
	    mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	    echo -e "${model},eval,SUCCESS" >>${result_path}/result.log
    fi
let i+=1
done
# slim_9_1  Classification model mobilenet_v1 with pruning
#---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${fluid_path}/PaddleSlim/classification/pruning
yaml_name='mobilenet_v1'
model=slim_9_1_pruning_classification_${yaml_name}
sed -i "s/epoch: 121/epoch: 1/g" configs/${yaml_name}.yaml
rm -rf checkpoints
time (python -u compress.py \
    --model "MobileNet" \
    --use_gpu 1 \
    --batch_size 256 \
    --total_images 1281167 \
    --lr_strategy "piecewise_decay" \
    --num_epochs 1 \
    --lr 0.1 \
    --l2_decay 3e-5 \
    --pretrained_model ../pretrain/MobileNetV1_pretrained \
    --config_file ./configs/${yaml_name}.yaml > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi

# slim_9_2  Classification model mobilenet_v2 with pruning
#---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${fluid_path}/PaddleSlim/classification/pruning
yaml_name='mobilenet_v2'
model=slim_9_2_pruning_classification_${yaml_name}
sed -i "s/epoch: 241/epoch: 1/g" configs/${yaml_name}.yaml
time (python -u compress.py \
    --model "MobileNetV2" \
    --use_gpu 1 \
    --batch_size 256 \
    --total_images 1281167 \
    --lr_strategy "cosine_decay" \
    --num_epochs 1 \
    --lr 0.1 \
    --l2_decay 4e-5 \
    --pretrained_model ../pretrain/MobileNetV2_pretrained \
    --config_file ./configs/${yaml_name}.yaml > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi

# slim_9_3  Classification model resnet34 with pruning
#---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${fluid_path}/PaddleSlim/classification/pruning
yaml_name='resnet34'
model=slim_9_3_pruning_classification_${yaml_name}
sed -i "s/epoch: 121/epoch: 1/g" configs/${yaml_name}.yaml
time (python -u compress.py \
    --model "ResNet34" \
    --use_gpu 1 \
    --batch_size 256 \
    --total_images 1281167 \
    --lr_strategy "cosine_decay" \
    --lr 0.1 \
    --num_epochs 1 \
    --l2_decay 1e-4 \
    --pretrained_model ../pretrain/ResNet34_pretrained \
    --config_file ./configs/${yaml_name}.yaml > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
# slim_9_3 classification_prune eval and infer
classification_prune_models='mobilenet_v1 mobilenet_v2 resnet34'
i=1
for prune_model in ${classification_prune_models}
do
    cd ${fluid_path}/PaddleSlim/classification
    model=slim_9_${i}_2_prune_classification_${prune_model}_infer
    time (python infer.py \
        --use_gpu True \
        --model_path ./pruning/checkpoints/${prune_model}/0/eval_model/ \
        --model_name __model__.infer \
        --params_name __params__ >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
    if [ $? -ne 0 ];then
	    mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	    echo -e "${model},infer,FAIL" >>${result_path}/result.log;
    else
	    mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	    echo -e "${model},infer,SUCCESS" >>${result_path}/result.log
    fi
    cd ${fluid_path}/PaddleSlim/classification
    model=slim_9_${i}_3_prune_classification_${prune_model}_eval
    time (python eval.py --use_gpu True --model_path ./pruning/checkpoints/${prune_model}/0/eval_model/ --model_name __model__.infer --params_name __params__ >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
    if [ $? -ne 0 ];then
	    mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	    echo -e "${model},eval,FAIL" >>${result_path}/result.log;
    else
	    mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	    echo -e "${model},eval,SUCCESS" >>${result_path}/result.log
    fi
let i+=1
done
# slim_10_1  Classification model mobilenetv1_resnet50_distillation
#---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${fluid_path}/PaddleSlim/classification/distillation
yaml_name='mobilenetv1_resnet50_distillation'
model=slim_10_1_classification_${yaml_name}
sed -i "s/end_epoch: 130/end_epoch: 0/g" configs/${yaml_name}.yaml
sed -i "s/epoch: 130/epoch: 1/g" configs/${yaml_name}.yaml
cd -
cd ${calss_pretrain_dir}/ResNet50_pretrained
for files in $(ls res50_*)
    do mv $files ${files#*_}
done
for files in $(ls *)
    do mv $files "res50_"$files
done
cd -
cd ${fluid_path}/PaddleSlim/classification/distillation
rm -rf *_checkpoints
time (python -u compress.py \
    --model "MobileNet" \
    --teacher_model "ResNet50" \
    --teacher_pretrained_model ../pretrain/ResNet50_pretrained \
    --compress_config ./configs/${yaml_name}.yaml > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
# slim_10_1_2  Classification model mobilenetv1_resnet50_distillation infer
cd ${fluid_path}/PaddleSlim/classification
model=slim_10_1_2_classification_${yaml_name}_infer
time (python infer.py \
    --use_gpu True \
    --model_path ./distillation/checkpoints/0/eval_model/ \
    --model_name __model__.infer \
    --params_name __params__ > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},infer,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},infer,SUCCESS" >>${result_path}/result.log
fi
# slim_10_1_3  Classification model mobilenetv1_resnet50_distillation eval
cd ${fluid_path}/PaddleSlim/classification
model=slim_10_1_3_classification_${yaml_name}_eval
time (python eval.py \
    --use_gpu True \
    --model_path ./distillation/checkpoints/0/eval_model/ \
    --model_name __model__.infer \
    --params_name __params__ > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},eval,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},eval,SUCCESS" >>${result_path}/result.log
fi
cd ${fluid_path}/PaddleSlim/classification/distillation
mv checkpoints ${model}_checkpoints
# slim_10_2  Classification model mobilenetv2_resnet50_distillation
#---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${fluid_path}/PaddleSlim/classification/distillation
yaml_name='mobilenetv2_resnet50_distillation'
model=slim_10_2_classification_${yaml_name}
sed -i "s/end_epoch: 130/end_epoch: 0/g" configs/${yaml_name}.yaml
sed -i "s/epoch: 130/epoch: 1/g" configs/${yaml_name}.yaml
cd ${calss_pretrain_dir}/ResNet50_pretrained
for files in $(ls res50_*)
    do mv $files ${files#*_}
done
for files in $(ls *)
    do mv $files "res50_"$files
done
cd -
cd ${fluid_path}/PaddleSlim/classification/distillation
rm -rf *_checkpoints
time (python -u compress.py \
    --model "MobileNetV2" \
    --teacher_model "ResNet50" \
    --teacher_pretrained_model ../pretrain/ResNet50_pretrained \
    --compress_config ./configs/${yaml_name}.yaml > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
# slim_10_2_2  Classification model mobilenetv1_resnet50_distillation infer
cd ${fluid_path}/PaddleSlim/classification
model=slim_10_2_2_classification_${yaml_name}_infer
time  (python infer.py \
        --use_gpu True \
        --model_path ./distillation/checkpoints/0/eval_model/ \
        --model_name __model__.infer \
        --params_name __params__ > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},infer,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},infer,SUCCESS" >>${result_path}/result.log
fi
# slim_10_2_3  Classification model mobilenetv1_resnet50_distillation eval
cd ${fluid_path}/PaddleSlim/classification
model=slim_10_2_3_classification_${yaml_name}_eval
time  (python eval.py \
        --use_gpu True \
        --model_path ./distillation/checkpoints/0/eval_model/ \
        --model_name __model__.infer \
        --params_name __params__ > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},eval,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},eval,SUCCESS" >>${result_path}/result.log
fi
cd ${fluid_path}/PaddleSlim/classification/distillation
mv checkpoints ${model}_checkpoints
# slim_10_3  Classification model resnet34_resnet50_distillation
#---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${fluid_path}/PaddleSlim/classification/distillation
yaml_name='resnet34_resnet50_distillation'
model=slim_10_3_classification_${yaml_name}
sed -i "s/end_epoch: 130/end_epoch: 0/g" configs/${yaml_name}.yaml
sed -i "s/epoch: 130/epoch: 1/g" configs/${yaml_name}.yaml
cd ${calss_pretrain_dir}/ResNet50_pretrained
for files in $(ls res50_*)
    do mv $files ${files#*_}
done
for files in $(ls *)
    do mv $files "res50_"$files
done
cd -
cd ${fluid_path}/PaddleSlim/classification/distillation
rm -rf *_checkpoints
time (python -u compress.py \
    --model "ResNet34" \
    --teacher_model "ResNet50" \
    --teacher_pretrained_model ../pretrain/ResNet50_pretrained \
    --compress_config ./configs/${yaml_name}.yaml > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
# slim_10_3_2  Classification model resnet34_resnet50_distillation infer
cd ${fluid_path}/PaddleSlim/classification
model=slim_10_3_2_classification_${yaml_name}_infer
time  (python infer.py \
        --use_gpu True \
        --model_path ./distillation/checkpoints/0/eval_model/ \
        --model_name __model__.infer \
        --params_name __params__ > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},infer,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},infer,SUCCESS" >>${result_path}/result.log
fi
# slim_10_3_3  Classification model resnet34_resnet50_distillation eval
cd ${fluid_path}/PaddleSlim/classification
model=slim_10_3_3_classification_${yaml_name}_eval
time  (python eval.py \
        --use_gpu True \
        --model_path ./distillation/checkpoints/0/eval_model/ \
        --model_name __model__.infer \
        --params_name __params__ > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},eval,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},eval,SUCCESS" >>${result_path}/result.log
fi
cd ${fluid_path}/PaddleSlim/classification/distillation
mv checkpoints ${model}_checkpoints
# slim_11_1  detection model yolov3_mobilenet_v1_slim_quantiation
#---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${fluid_path}/PaddleCV/PaddleDetection/slim/quantization
yaml_name='yolov3_mobilenet_v1_slim'
model=slim_11_1_detection_quan_${yaml_name}
sed -i "s/end_epoch: 4/end_epoch: 1/g" ${yaml_name}.yaml
sed -i "s/epoch: 5/epoch: 2/g" ${yaml_name}.yaml
rm -rf checkpoints
time (python compress.py \
    -s yolov3_mobilenet_v1_slim.yaml  \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    -d "../../dataset/voc" \
    -o max_iters=258 \
    LearningRate.base_lr=0.0001 \
    LearningRate.schedulers='[!PiecewiseDecay {gamma: 0.1, milestones: [258, 516]}]' \
    pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar \
    YoloTrainFeed.batch_size=64 > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
#slim_11_1_2  detection model yolov3_mobilenet_v1_slim_quantiation freeze
cd ${fluid_path}/PaddleCV/PaddleDetection/slim/quantization
model=slim_11_1_2_detection_quan_${yaml_name}_freeze
time (python freeze.py \
    --model_path ./checkpoints/yolov3/0/eval_model/ \
    --weight_quant_type abs_max \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    --save_path ./freeze \
    -d "../../dataset/voc" > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},freeze,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},freeze,SUCCESS" >>${result_path}/result.log
fi
#slim_11_1_3  detection model yolov3_mobilenet_v1_slim_quantiation infer
cd ${fluid_path}/PaddleCV/PaddleDetection/slim/quantization
model=slim_11_1_3_detection_quan_${yaml_name}_infer
time (python ../infer.py \
    --model_path ./output/yolov3/float \
    --model_name model \
    --params_name weights \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    --infer_dir ../../demo > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},infer,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},infer,SUCCESS" >>${result_path}/result.log
fi
#slim_11_1_4  detection model yolov3_mobilenet_v1_slim_quantiation eval
cd ${fluid_path}/PaddleCV/PaddleDetection/slim/quantization
model=slim_11_1_4_detection_quan_${yaml_name}_eval
time (python ../eval.py \
    --model_path ./output/yolov3/float \
    --model_name model \
    --params_name weights \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    -d "../../dataset/voc" > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},eval,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},eval,SUCCESS" >>${result_path}/result.log
fi
# slim_11_2  detection model yolov3_mobilenet_v1_slim_prune
#---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${fluid_path}/PaddleCV/PaddleDetection/slim/prune
yaml_name='yolov3_mobilenet_v1_slim'
model=slim_11_2_detection_prune_${yaml_name}
sed -i "s/epoch: 271/epoch: 1/g" ${yaml_name}.yaml
sed -i "s/eval_epoch: 10/eval_epoch: 1/g" ${yaml_name}.yaml
rm -rf checkpoints
time (python compress.py \
    -s yolov3_mobilenet_v1_slim.yaml \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    -o max_iters=258 \
    YoloTrainFeed.batch_size=64 \
    -d "../../dataset/voc" > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
# slim_11_2_2  detection model yolov3_mobilenet_v1_slim_prune infer
cd ${fluid_path}/PaddleCV/PaddleDetection/slim/prune
model=slim_11_2_2_detection_prune_${yaml_name}_infer
time (python ../infer.py \
    --model_path ./checkpoints/0/eval_model/ \
    --model_name __model__.infer \
    --params_name __params__ \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    --infer_dir ../../demo > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},infer,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},infer,SUCCESS" >>${result_path}/result.log
fi
# slim_11_2_3  detection model yolov3_mobilenet_v1_slim_prune eval
cd ${fluid_path}/PaddleCV/PaddleDetection/slim/prune
model=slim_11_2_3_detection_prune_${yaml_name}_eval_infer
time (python ../eval.py \
    --model_path ./checkpoints/0/eval_model/ \
    --model_name __model__.infer \
    --params_name __params__ \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    -d "../../dataset/voc" > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},eval_infer,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},eval_infer,SUCCESS" >>${result_path}/result.log
fi
# slim_11_2_4  detection model yolov3_mobilenet_v1_slim_prune eval
cd ${fluid_path}/PaddleCV/PaddleDetection/slim/prune
model=slim_11_2_4_detection_prune_${yaml_name}_eval
time (python ../eval.py \
    --model_path ./checkpoints/0/eval_model/ \
    --model_name __model__ \
    --params_name __params__ \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    -d "../../dataset/voc" > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},eval,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},eval,SUCCESS" >>${result_path}/result.log
fi
# slim_11_3  detection model yolov3_mobilenet_v1_slim_dist
#---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${fluid_path}/PaddleCV/PaddleDetection/slim/distillation
yaml_name='yolov3_mobilenet_v1_yolov3_resnet34_distillation'
model=slim_11_3_detection_${yaml_name}
sed -i "s/epoch: 271/epoch: 1/g" ${yaml_name}.yml
sed -i "s/end_epoch: 270/end_epoch: 0/g" ${yaml_name}.yml
cd ${fluid_path}/PaddleCV/PaddleDetection/slim/distillation
if [ ! -d 'pretrain' ]; then
  mkdir pretrain
fi
cd pretrain
if [ ! -f ${yolov3_r34_voc} ]; then
    wget ${dete_root_url}/${yolov3_r34_voc}
    tar xf ${yolov3_r34_voc}
fi
cd -
cd ${fluid_path}/PaddleCV/PaddleDetection/slim/distillation/pretrain/yolov3_r34_voc
for files in $(ls teacher_*)
    do mv $files ${files#*_}
done
for files in $(ls *)
    do mv $files "teacher_"$files
done
cd -
time (python -u compress.py \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    -t yolov3_resnet34.yml \
    -s yolov3_mobilenet_v1_yolov3_resnet34_distillation.yml \
    -o YoloTrainFeed.batch_size=32 \
    -d ../../dataset/voc \
    --teacher_pretrained ./pretrain/yolov3_r34_voc > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
#slim_11_3_2  detection model yolov3_mobilenet_v1_slim_dist infer
cd ${fluid_path}/PaddleCV/PaddleDetection/slim/distillation
model=slim_11_3_2_detection_${yaml_name}_infer
time (python ../infer.py \
    --model_path ./checkpoints/0/eval_model/ \
    --model_name __model__.infer \
    --params_name __params__ \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    --infer_dir ../../demo  > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},infer,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},infer,SUCCESS" >>${result_path}/result.log
fi
#slim_11_3_3  detection model yolov3_mobilenet_v1_slim_dist eval
cd ${fluid_path}/PaddleCV/PaddleDetection/slim/distillation
model=slim_11_3_3_detection_${yaml_name}_eval
time (python ../eval.py \
    --model_path ./checkpoints/0/eval_model/ \
    --model_name __model__.infer \
    --params_name __params__ \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    -d "../../dataset/voc"  > ${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},eval,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},eval,SUCCESS" >>${result_path}/result.log
fi

# slim_12_1 light_nas_flops_search  not support windows
#---------------------------------------------------------------------------
apt-get install net-tools -y
ip=ifconfig eth0|grep "inet addr:"|awk -F":" '{print $2}'|awk '{print $1}'
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${fluid_path}/PaddleSlim/light_nas
yaml_name='compress'
model=slim_12_1_light_nas_flops_search_${yaml_name}
sed -i "s/epoch: 500/epoch: 2/g" ${fluid_path}/PaddleSlim/light_nas/${yaml_name}.yaml
sed -i "s/retrain_epoch: 5/retrain_epoch: 1/g" ${fluid_path}/PaddleSlim/light_nas/${yaml_name}.yaml
sed -i "s/end_epoch: 500/end_epoch: 2/g" ${fluid_path}/PaddleSlim/light_nas/${yaml_name}.yaml
# 13ip
sed -i "s/server_ip: ''/server_ip: '10.255.118.26'/g" ${fluid_path}/light_nas/${yaml_name}.yaml
#sed -i "s/server_ip: ''/server_ip: '${ip}'/g" ${fluid_path}/PaddleSlim/light_nas/${yaml_name}.yaml
rm -rf slim_LightNASStrategy_controller_server.socket
time (python search.py >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},search,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},search,SUCCESS" >>${result_path}/result.log
fi
cd -
mv compress.yaml search_compress.yaml
# slim_12_2 light_nas_flops_train not support windows
#---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${fluid_path}/PaddleSlim/light_nas
yaml_name='compress_train'
model=slim_12_2_light_nas_flops_train_${yaml_name}
sed -i "s/epoch: 500/epoch: 2/g" ${fluid_path}/PaddleSlim/light_nas/${yaml_name}.yaml
sed -i "s/retrain_epoch: 5/retrain_epoch: 1/g" ${fluid_path}/PaddleSlim/light_nas/${yaml_name}.yaml
sed -i "s/end_epoch: 500/end_epoch: 2/g" ${fluid_path}/PaddleSlim/light_nas/${yaml_name}.yaml
mv compress_train.yaml compress.yaml
time (python search.py >${log_path}/${model}.log) >>${log_path}/${model}.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/${model}.log ${log_path}/FAIL/${model}.log
	echo -e "${model},train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/${model}.log ${log_path}/SUCCESS/${model}.log
	echo -e "${model},train,SUCCESS" >>${result_path}/result.log
fi
cd -
