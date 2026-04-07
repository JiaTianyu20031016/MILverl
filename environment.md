环境配置：首先按照verl的要求配置verl虚拟环境，然后
cd MIL
pip install -e .[dev]
以安装trl，MILdata和MILmodel。

注意，环境配置完成后，务必检查transformers版本，必须低于5.0.0，否则verl会报关于import AutoModelForVision2Text的错误。

然而，由于做mil实验时所用的transformers库版本较高，有些mil checkpoint中tokenizer相关文件保存的格式和低版本的transfomers不兼容。可能需要将这些checkpoint文件夹下的tokenizer_config.json中的'extra_special_tokens'字段删去。