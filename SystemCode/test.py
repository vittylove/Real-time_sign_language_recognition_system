import os
import cv2
import torch
import model.detector as de
import utils.utils

def detect_image(ori_img):
    data = "./data/coco.data"
    weight = "./data/weight.pth"
    cfg = utils.utils.load_datafile(data)
    assert os.path.exists(weight), "请指定正确的模型路径"

    #模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = de.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(weight, map_location=device))

    #sets the module in eval node
    model.eval()
    
    #数据预处理
    res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR)
    img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
    img = torch.from_numpy(img.transpose(0,3, 1, 2))
    img = img.to(device).float() / 255.0

    #模型推理
    preds = model(img)

    #特征图后处理
    output = utils.utils.handel_preds(preds, cfg, device)
    output_boxes = utils.utils.non_max_suppression(output, conf_thres = 0.8, iou_thres = 0.1)

    #加载label names
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())
    
    h, w, _ = ori_img.shape
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]
    score = 0
    category = ""
    #绘制预测框
    for box in output_boxes[0]:
        box = box.tolist()
        obj_score = box[4]
        if obj_score > score:
            category = LABEL_NAMES[int(box[5])]
    return category

