# 道路车辆行人感知——Mask R-CNN 实例分割项目

## 项目简介
本项目基于 PyTorch 和 torchvision 实现了道路场景下行人和车辆的实例分割。采用 COCO 格式数据集，利用 Mask R-CNN 进行训练和推理，支持对图片中的行人和多种车辆类别进行像素级分割。

## 主要功能
- 支持 COCO 格式数据集的加载与自定义类别筛选（行人+车辆）
- 数据增强与批量加载，适配目标检测/分割任务
- Mask R-CNN 训练、模型保存
- 独立推理脚本，支持模型评估与可视化

## 目录结构
```
项目根目录/
├── mask_rcnn_train.py         # 训练脚本
├── mask_rcnn_infer.py         # 推理与评估脚本
├── train.ipynb                # 交互式代码讲解与实验
├── requirements.txt           # 依赖包列表
├── maskrcnn_model.pth         # 训练得到的模型权重
├── 类别id.txt                 # COCO类别id与名称映射
├── 数据/                      # 数据集根目录
│   └── data/
│       └── coco_data/
│           └── coco2017/
│               ├── train2017/         # 训练图片
│               ├── val2017/           # 验证图片
│               └── annotations/       # 标注文件（instances_train2017.json等）
├── test/                      # 推理结果示例图片
└── ...
```

## 环境依赖
- Python 3.8 及以上
- PyTorch >= 1.10
- torchvision >= 0.11
- numpy
- pillow
- scikit-learn
- pycocotools

建议使用 Anaconda 环境，安装依赖：
```
pip install -r requirements.txt
```

## 数据准备
1. 下载 COCO 2017 数据集（或自定义 COCO 格式数据），解压到 `数据/data/coco_data/coco2017/` 目录下。
2. 目录结构需包含：
   - `train2017/` 训练图片
   - `val2017/` 验证图片
   - `annotations/instances_train2017.json`、`instances_val2017.json` 标注文件
3. 可根据 `类别id.txt` 文件了解类别映射。

## 训练说明
- 运行 `mask_rcnn_train.py` 进行模型训练，训练完成后会自动保存模型权重为 `maskrcnn_model.pth`。
- 训练脚本默认只保留行人和车辆相关类别，类别映射见代码注释。
- 可在 `train.ipynb` 中交互式理解和实验训练流程。

## 推理与评估
- 运行 `mask_rcnn_infer.py`，可加载训练好的模型，对新图片或验证集进行分割推理和评估。
- 推理结果可保存到 `test/` 目录下，便于可视化。

## 其他说明
- 代码结构清晰，便于二次开发和自定义扩展。
- 推荐在 GPU 环境下运行，显著提升训练和推理速度。
- 如需自定义类别或数据增强方式，请参考 `CocoDataset` 类和 `get_transform` 函数。

---

如有问题或建议，欢迎交流与反馈！
