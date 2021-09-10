# 全卷积网络进行语音降噪

[论文](https://arxiv.org/pdf/1609.07132.pdf)

## 准备工作

- 搭建虚拟环境

建议使用 virtualenv 工具进行虚拟环境搭建；

```shell script
virtualenv -p python3.6 venv
source venv/bin/activate
pip insatll -r requirements.txt
```

- 数据处理

`Work` 文件夹下有数据处理例子，以aishell_1为例，需要生成干净音频manifest列表文件，以及噪声manifest列表文件；

```
manifest.aishell_1.train
   格式： {"audio_filepath": "/home/**/**/aishell_1/data_aishell/wav/dev/S0724/BAC009S0724W0417.wav", "duration": 5.90}

manifest.noise_train
   格式： {"audio_filepath": "/home/**/**/noise/train/noise_88.wav", "duration": 30.0}
```

处理脚本 [run_data.sh](Work/aishell_1/run_data.sh), 修改配置内容，同时处理干净数据和噪声数据。

```shell script
cd Work/aishell_1

$ sh run_data.sh
```

## 代码结构说明

代码包括五个部分，分别是`数据`、`训练`、`测试`、`推理`、`移植`；

Work文件夹下为shell执行脚本、数据集处理脚本用于数据处理、模型训练及测试等过程的执行控制。
此外还包括生成的数据、log等文件，

- 数据

[DataLoader](data_utils/data_loader.py) -> [Sampler](data_utils/data_loader.py) 
-> [DataSet](data_utils/data_loader.py) -> [AudioFeature](data_utils/audio_feature.py)

- 训练

[main()](train.py) -> [FullyCNNTrainer](model_utils/trainer.py)


- 测试

[main()](test.py) -> [FullyCNNTester](model_utils/tester.py)

- 推理

[main()](infer.py) -> [InferenceEngine](infer.py)

- freeze

[FreezeEngine](freeze.py)


## 案例

- 所有的案例都放在[Work](Work/)目录下，以aishell_1数据集为例，[cfg](Work/aishell_1/cfg)为配置文件;

- [run_train.sh](Work/aishell_1/run_train.sh) 执行训练指令，需要指定使用的cfg配置文件以及数据处理使用的进程数（多进程有问题，暂时建议设为1）

- [run_test.sh](Work/aishell_1/run_test.sh) 执行测试指令，同上；

- 训练过程中每5个epoch后会进行一次验证，验证的结果存放在[log](checkpoints/aishell_1/log)目录下；

- 保存的模型在[checkpoints](checkpoints/) 目录下；

## 实验结果

|数据集|PESQ|STOI|SDR|
|-:-|-:-|-:-|-:-|
|aishell_1 + 自录噪声|2.1504|0.7027|2.1972|


## 待完成

- 实验结果
- 实时demo


**注**： 数据处理机制参考pytorch的实现以及paddlepaddle的DeepSpeech2.
