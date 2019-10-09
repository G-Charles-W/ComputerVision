import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import os
import numpy
import cv2
import random
from MobileNetV2_depthwise import MobileNet2_7B48
from torch.nn.functional import softmax
from torchvision.transforms import ToTensor
import pickle as pkl


def evaluate(model_weight_name, model, inputs):
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        print(inputs)
        model = model.cuda()
        labels = torch.from_numpy(numpy.array([1])).cuda()

    criterion = nn.CrossEntropyLoss()

    if os.path.exists(model_weight_name):
        print(f"The best model {model_weight_name} has been loaded")
        model.load_state_dict(torch.load(model_weight_name))

    model.eval()

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        print(_)
        print(preds)
    loss = criterion(outputs, labels)
    print(loss)
    outputs = softmax(outputs, dim=1)
    print(outputs)
    outputs = outputs.squeeze(0)
    outputs = outputs.cpu().numpy()
    print(outputs)

    print(f'outputs: interest {outputs[0]}, not interested {outputs[1]}')
    return outputs


def transform(file):
    img = cv2.imread(file, 0)
    img = cv2.resize(img, (48, 48))
    input_img = ToTensor()(img)
    input_img = input_img.unsqueeze(0)
    return input_img


def show_results(file, detection):
    img = cv2.imread(file)

    a = round(detection[0], ndigits=2)
    b = round(detection[1], ndigits=2)
    label1 = f'  happiness:{a:.2}'
    label2 = f'unhappiness:{b:.2}'

    colors = pkl.load(open("pallete", "rb"))
    color = (255, 255, 255)
    color1 = (0, 0, 0)
    print(color)
    t_size = cv2.getTextSize(label1, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c1 = (10, 10)
    c2 = (6, 26)
    # c3 = (158, 38)
    # cv2.rectangle(qimg, c1, c3, color1, 2)
    cv2.putText(img, label1, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    cv2.putText(img, label2, (c2[0], c2[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        pass


if __name__ == "__main__":
    # batch picture detect
    model_ft = MobileNet2_7B48(2)
    model_weight_name_ft = 'mobilenet_v2_2C_7B_F960_adam_89%_2019_09_26_12_09_41'
    for i in range(7):
        img_name = f'{i}.jpeg'
        print(img_name)
        # img_name = '0_2.jpg'
        img = transform(img_name)
        outputs = evaluate(model_weight_name_ft, model_ft, img)
        show_results(img_name, outputs)

    # single picture detect
    # img_name = '0_10.jpeg'
    # img_name = '0_2.jpg'
    # img = transform(img_name)
    # outputs = evaluate(model_weight_name_ft, model_ft, img)
    # show_results(img_name, outputs)
