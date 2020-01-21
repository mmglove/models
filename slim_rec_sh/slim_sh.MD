## slim sh 的更新历史
1. 20200117
 - (1-7) slim/configs 文件缺失，解决:新的文件configs_13
 - (8-9) class/quantization/下的yaml文件在配置时必须end_epoch<epoch ,否则output不会保存;
 解决：修改配置文件中下的yaml文件在配置时必须end_epoch=1，epoch=2；
 - (10)  class/prune中的V2模型的配置文件误删，解决：无需替换yaml,直接修改epoch数据
 - (11) detection/dist中的yolov3_tar下载预训练模型的链接同分类模型的不一样；解决：已替换新的链接
 - (10) detection/下的模型运行时未安装pycocotools；解决：在运行detection前pip安装一下
 - (12) light_nas/search任务无法自行停止，还需要替换成固定的IP再确认一下
 