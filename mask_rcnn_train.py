import os
import numpy as np
import torch
import torchvision  # 计算机视觉工具包，含模型和数据增强
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn  # Mask R-CNN 检测模型
from torchvision.transforms import functional as F  # 图像变换函数
import torchvision.transforms as T # 图像预处理与增强
from pycocotools.coco import COCO  # COCO 数据集解析工具
from PIL import Image  # 图像读取与处理
from sklearn.metrics import precision_score, recall_score  # 精度与召回率评估指标
from tqdm import tqdm


# ======================================函数功能测试======================================

# coco = COCO('数据/data/coco_data/coco2017/annotations/instances_train2017.json')

# # 查看图片ID列表
# print("图片ID列表:", coco.getImgIds()[:5])

# # 获取键
# keys = list(coco.imgs.keys())
# print('键名:', keys[:5])

# # 查看类别信息
# print("类别信息:", coco.loadCats(coco.getCatIds()))

# # 查看某张图片的信息
# img_id = coco.getImgIds()[0]
# print("图片信息:", coco.loadImgs(img_id))

# # 查看某张图片的标注ID
# ann_ids = coco.getAnnIds(imgIds=img_id)
# print("标注ID:", ann_ids)

# # 查看某个标注的内容
# if ann_ids:
#     ann = coco.loadAnns([ann_ids[0]])[0]
#     print("标注内容:", ann)
#     # 探索 annToMask 的功能：将标注转换为二值掩码（mask）
#     mask = coco.annToMask(ann)
#     print("annToMask 输出掩码 shape:", mask.shape)
#     print("掩码像素值统计:", np.unique(mask, return_counts=True))

# # 探索 img 的功能：读取并显示图片
# img_path = coco.loadImgs(img_id)[0]['file_name']
# img_path = os.path.join('数据/data/coco_data/coco2017/train2017/train2017', img_path)
# img = Image.open(img_path).convert('RGB') # 读取图片并转换为RGB格式
# # img.show()  # 显示图片
# print("图片尺寸:", img.size)

# # imgs 字典结构
# print("imgs 字典类型:", type(coco.imgs))
# print("imgs 字典样例:", list(coco.imgs.items())[:1])

# ========================================================================================

# ======================================数据集类定义======================================
class CocoDataset(torch.utils.data.Dataset):
    """
    CocoDataset 是一个用于加载 COCO 格式数据集的自定义 PyTorch Dataset 类
    支持实例分割任务，能够返回图像、目标框、类别标签、分割掩码等信息
    """
    def __init__(self, root, annFile, transforms=None):
        """
        参数:
            root (str): 图像文件夹的根目录
            annFile (str): COCO格式的标注文件路径
            transforms (callable, optional): 图像及目标的变换函数
        """
        self.root = root # 图像所在的根目录
        self.coco = COCO(annFile) # 加载 COCO 数据集标注文件
        self.ids = list(self.coco.imgs.keys()) # 获取所有图片的ID（self.ids 是图片的唯一标识符列表，每个元素对应一张图片）
        self.transforms = transforms # 图像变换函数

        # 只保留感兴趣的类别
        self.target_cat_ids = [1, 2, 3, 4, 6, 7, 8]  # 行人和车辆
        # 建立类别id到label的映射：1=行人, 2=车辆
        self.cat_id_to_label = {1: 1}
        for cid in [2, 3, 4, 6, 7, 8]:
            self.cat_id_to_label[cid] = 2
        self.num_classes = 3  # 0=背景, 1=行人, 2=车辆

        # 区别说明：
        # self.ids 保存的是图片ID列表，用于索引和遍历数据集中的每一张图片。
        # cat_ids 保存的是类别ID列表，用于类别标签的映射和模型输出的类别数设置。
        # 两者分别对应“图片”和“类别”两个不同的维度。

    def __getitem__(self, index):
        """
        根据索引 index 返回一张图片及其对应的目标信息（boxes、labels、masks 等）。
        包括以下步骤：
        1. 获取图片ID和对应的标注信息；
        2. 读取图片文件并转换为RGB格式；
        3. 提取每个目标的边界框、类别标签和分割掩码；
        4. 将所有目标信息转换为Tensor格式，并组装为target字典；
        5. 对图片进行变换；
        6. 返回处理后的图片和target字典。
        """
        coco = self.coco # 获取 COCO 实例
        img_id = self.ids[index] # 根据索引获取图片ID
        ann_ids = coco.getAnnIds(imgIds=img_id) # 获取该图片的标注ID
        anns = coco.loadAnns(ann_ids) # 加载标注信息
        anns = [ann for ann in anns if ann['category_id'] in self.target_cat_ids] # 过滤行人与车辆类别的标注信息
        if len(anns) == 0: # 如果没有标注信息，递归调用下一个索引
            return self.__getitem__((index + 1) % len(self.ids))
        
        path = coco.loadImgs(img_id)[0]['file_name'] # 获取图片文件名
        img_path = os.path.join(self.root, path) # 构建图片完整路径
        img = Image.open(img_path).convert('RGB') # 读取图片并转换为RGB格式
        
        boxes = [] # 存储目标边界框
        labels = [] # 存储目标类别标签
        masks = [] # 存储目标分割掩码
        for ann in anns:
            bbox = ann['bbox'] # 获取目标的边界框信息
            # COCO 的 bbox 格式为 [x, y, width, height]，转换为 [x1, y1, x2, y2]
            # 其中 (x1, y1) 是左上角坐标，(x2, y2) 是右下角坐标
            boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])

            labels.append(self.cat_id_to_label[ann['category_id']]) # 获取目标的类别标签，并映射到自定义的标签
            mask = coco.annToMask(ann) # 获取目标的分割掩码
            masks.append(mask) # 将掩码添加到列表中
        # 将 boxes、labels 和 masks 转换为 Tensor 格式
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        image_id = torch.tensor([img_id])
        area = torch.as_tensor([ann['area'] for ann in anns], dtype=torch.float32)
        iscrowd = torch.as_tensor([ann.get('iscrowd', 0) for ann in anns], dtype=torch.int64)
        
        target = {} # 创建一个字典来存储目标信息
        target['boxes'] = boxes 
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms: # 如果定义了图像变换函数，则对图像进行变换
            img = self.transforms(img)
        else: # 否则直接转换为Tensor格式
            img = F.to_tensor(img)
        return img, target

    def __len__(self):
        return len(self.ids)
    
# ======================================数据集类定义结束======================================

# ======================================数据增强与预处理函数定义======================================
def get_transform(train=True):
    """
    定义数据增强与预处理流程。
    参数:
        train (bool): 是否为训练模式。训练模式下会添加数据增强操作。
    返回:
        transforms (Compose): 图像变换操作的组合。
    """
    transforms = []
    # 基础转换：将PIL图像转换为Tensor，并归一化到[0,1]
    transforms.append(T.ToTensor())
    if train:
        # 随机水平翻转，概率为0.5
        transforms.append(T.RandomHorizontalFlip(0.5))
        # 随机颜色抖动（亮度、对比度、饱和度、色调）
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        # 随机仿射变换（旋转、平移、缩放）
        transforms.append(T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)))
    return T.Compose(transforms)

def collate_fn(batch):
    # 假设 batch 是一个列表，每个元素是 (img, target)，比如 batch = [(img1, target1), (img2, target2), ...]
    # zip(*batch) 的作用是把 batch 拆成两个元组：所有 img 和所有 target。
    # 这样返回的就是 (imgs, targets)，imgs 是一个元组，targets 也是一个元组。
    
    # 因此，我们也可以这么写：
    # imgs, targets = zip(*batch)
    # return imgs, targets
    return tuple(zip(*batch))

# ======================================数据增强与预处理函数定义结束======================================

# ======================================训练主函数定义======================================
def train():
    """
    训练主流程函数，包含以下步骤：
    1. 设置训练和验证数据集的路径。
    2. 构建训练和验证集的 Dataset 与 DataLoader。
    3. 初始化 Mask R-CNN 模型，并设置类别数。
    4. 配置优化器（SGD）。
    5. 进入训练模式，进行训练
    """
    # 路径设置
    base_dir = os.path.dirname(os.path.abspath('mask_rcnn_train.py'))
    train_img_dir = os.path.join(base_dir,'data', 'coco_data', 'coco2017', 'train2017', 'train2017')
    train_ann_file = os.path.join(base_dir, 'data', 'coco_data', 'coco2017', 'annotations', 'instances_train2017.json')
   
    # 创建数据集与Dataloader
    train_dataset = CocoDataset(root=train_img_dir, annFile=train_ann_file, transforms=get_transform())
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = maskrcnn_resnet50_fpn(num_classes=train_dataset.num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    # 使用预训练模型的参数
    optimizer = torch.optim.Adam(params, lr=0.0005, weight_decay=0.0005)

    # 只跑一个batch
    # 进入训练模式，仅跑一个 batch 理解流程
    # model.train()
    # for images, targets in train_loader:
    #     # 将图片和目标信息移动到指定设备上
    #     images = [img.to(device) for img in images]
    #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #     # 前向传播，计算损失字典（包含分类、框回归、掩码等多项损失）
    #     loss_dict = model(images, targets)
    #     losses = sum(loss for loss in loss_dict.values()) # 将所有损失项加和得到总损失
    #     optimizer.zero_grad() # 梯度清零
    #     losses.backward() # 反向传播
    #     optimizer.step() # 优化器更新参数
    #     # 打印当前 batch 的损失
    #     print(f'单batch Loss: {losses.item():.4f}')
    #     break  # 只跑一个 batch，便于调试和理解流程

    # 完整训练流程
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        # tqdm 包裹 train_loader，显示进度条
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}')
    torch.save(model.state_dict(), os.path.join(base_dir, 'maskrcnn_model.pth'))
    print('训练完成!')

# ======================================训练主函数定义结束======================================


if __name__ == '__main__':
    # 训练完成后自动评估验证集准确率和召回率
    train()