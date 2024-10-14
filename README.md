# space_llm

## 自制大模型推理框架


## 编译方法
```shell
# 在项目目录下执行make
make
```

## 添加编译链接选项
```shell
# 由于cuda对compile_commands.json有点冲突，暂时不添加compile_commands.json配置文件
source source
```

## 单元测试方法
```shell
# 添加单元测试
make && make unittest
```
### 示例展示方法
```shell
$ make && make demo_tensor
Device NVIDIA GeForce RTX 3070 Ti Laptop GPU
Input_tensor size: [1, 12, 224, 224]
```

### 调试方法
```shell
make debug
```

