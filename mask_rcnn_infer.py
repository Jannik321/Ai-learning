import os
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from sklearn.metrics import precision_score, recall_score
from mask_rcnn_train import CocoDataset, get_transform
from torchvision.transforms import functional as F
from pycocotools.coco import COCO

class CocoTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.cat_id_to_label = {1: 1}
        for cid in [2, 3, 4, 6, 7, 8]:
            self.cat_id_to_label[cid] = 2
        self.transforms = transforms
        self.target_cat_ids = [1, 2, 3, 4, 6, 7, 8]
        self.num_classes = 3

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        else:
            img = F.to_tensor(img)
        return img, img_id  # 只返回图片和图片ID

    def __len__(self):
        return len(self.ids)
# ===================== 路径设置 =====================
base_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base_dir, 'data', 'coco_data', 'coco2017', 'val2017', 'val2017')
ann_file = os.path.join(base_dir, 'data', 'coco_data', 'coco2017', 'annotations', 'instances_val2017.json')

# ===================== 加载数据集和模型 =====================
val_dataset = CocoTestDataset(root=img_dir, annFile=ann_file, transforms=get_transform(train=False))
model = maskrcnn_resnet50_fpn(num_classes=val_dataset.num_classes)
model.load_state_dict(torch.load(os.path.join(base_dir, 'maskrcnn_model.pth'), map_location='cpu'))
model.eval()

# ===================== 单张图片推理与可视化 =====================
img_name = os.listdir(img_dir)[0]  # 取第一张
img_path = os.path.join(img_dir, img_name)
img = Image.open(img_path).convert('RGB')
img_tensor = torchvision.transforms.ToTensor()(img)

with torch.no_grad():
    prediction = model([img_tensor])[0]

img_arr = np.array(img).copy()
has_box = False
for i in range(len(prediction['boxes'])):
    score = prediction['scores'][i].item()
    if score < 0.2:
        continue
    has_box = True
    box = prediction['boxes'][i].cpu().numpy()
    mask = prediction['masks'][i, 0].cpu().numpy()
    mask = (mask > 0.5)
    color = (np.random.rand(3) * 255).astype(np.uint8)
    for c in range(3):
        img_arr[..., c][mask] = img_arr[..., c][mask] * 0.3 + color[c] * 0.7
img_pil = Image.fromarray(img_arr.astype(np.uint8))
draw = ImageDraw.Draw(img_pil)
try:
    font = ImageFont.truetype("arial.ttf", 28)
except:
    font = ImageFont.load_default()
for i in range(len(prediction['boxes'])):
    score = prediction['scores'][i].item()
    if score < 0.2:
        continue
    box = prediction['boxes'][i].cpu().numpy()
    label_id = int(prediction['labels'][i].cpu().numpy())
    cat_name = None
    for k, v in val_dataset.cat_id_to_label.items():
        if v == label_id:
            cat = val_dataset.coco.loadCats([k])
            if cat and 'name' in cat[0]:
                cat_name = str(cat[0]['name'])
            else:
                cat_name = str(k)
            break
    color = (255, 0, 0)
    draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=4)
    text = cat_name if cat_name else str(label_id)
    draw.text((box[0], max(0, box[1]-32)), text, fill=color, font=font)
save_path = os.path.join(base_dir, f"{os.path.splitext(img_name)[0]}_result.jpg")
img_pil.save(save_path)
print(f"已保存标注结果图片: {save_path}")
if not has_box:
    print('警告：该图片未检测到任何目标，建议调低score阈值或检查模型效果。')

# ===================== 批量评估准确率和召回率 =====================
all_true = []
all_pred = []
num_test = 20
for idx in range(num_test):
    img_name = os.listdir(img_dir)[idx]
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    img_tensor = torchvision.transforms.ToTensor()(img)
    with torch.no_grad():
        prediction = model([img_tensor])[0]
    ann_ids = val_dataset.coco.getAnnIds(imgIds=val_dataset.ids[idx])

    anns = [ann for ann in val_dataset.coco.loadAnns(ann_ids) if ann['category_id'] in val_dataset.target_cat_ids]

    true_labels = set([val_dataset.cat_id_to_label[ann['category_id']] for ann in anns])
    pred_labels = set()
    for i in range(len(prediction['boxes'])):
        score = prediction['scores'][i].item()
        if score < 0.5:
            continue
        pred_labels.add(int(prediction['labels'][i].cpu().numpy()))
    for label in range(1, val_dataset.num_classes):
        all_true.append(1 if label in true_labels else 0)
        all_pred.append(1 if label in pred_labels else 0)
precision = precision_score(all_true, all_pred)
recall = recall_score(all_true, all_pred)
print(f"测试集前{num_test}张图片的准确率: {precision:.4f}，召回率: {recall:.4f}")

# ===================== 批量保存推理结果图片 =====================
save_dir = os.path.join(base_dir, 'test')
os.makedirs(save_dir, exist_ok=True)
from tqdm import tqdm
for idx in tqdm(range(50), desc='保存测试结果图片'):
    img_name = os.listdir(img_dir)[idx]
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    img_tensor = torchvision.transforms.ToTensor()(img)
    with torch.no_grad():
        prediction = model([img_tensor])[0]
    img_arr = np.array(img).copy()
    for i in range(len(prediction['boxes'])):
        score = prediction['scores'][i].item()
        if score < 0.2:
            continue
        box = prediction['boxes'][i].cpu().numpy()
        mask = prediction['masks'][i, 0].cpu().numpy()
        mask = (mask > 0.5)
        color = (np.random.rand(3) * 255).astype(np.uint8)
        for c in range(3):
            img_arr[..., c][mask] = img_arr[..., c][mask] * 0.3 + color[c] * 0.7
    img_pil = Image.fromarray(img_arr.astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()
    for i in range(len(prediction['boxes'])):
        score = prediction['scores'][i].item()
        if score < 0.2:
            continue
        box = prediction['boxes'][i].cpu().numpy()
        label_id = int(prediction['labels'][i].cpu().numpy())
        cat_name = None
        for k, v in val_dataset.cat_id_to_label.items():
            if v == label_id:
                cat = val_dataset.coco.loadCats([k])
                if cat and 'name' in cat[0]:
                    cat_name = str(cat[0]['name'])
                else:
                    cat_name = str(k)
                break
        color = (255, 0, 0)
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=4)
        text = cat_name if cat_name else str(label_id)
        draw.text((box[0], max(0, box[1]-32)), text, fill=color, font=font)
    save_path = os.path.join(save_dir, f"{os.path.splitext(img_name)[0]}_result.jpg")
    img_pil.save(save_path)
print(f"已批量保存前50张图片的推理结果到: {save_dir}")