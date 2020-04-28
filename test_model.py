from augmentation import get_test_transforms
import torch
from main import Model
from PIL import Image
import numpy as np

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

if __name__ == "__main__":
    
    model = torch.load("./android/mixnet_84acc_24classes.pt")
    
    train_a_path =  "./dataset/train/A/A0 (2).jpg"
    test_a_path = "./dataset/val/A/A_FilipA_light_1_2020-01-24[phone]_1.jpg"

    train_a_path = "//home/odin/Programing/HandSight_PyTorchLightning/dataset/train/S/color_18_0002 (2).png"
    test_a_path = "/home/odin/Programing/HandSight_PyTorchLightning/dataset/val/S/S_FilipA_light_1_2020-01-24[phone]_1.jpg"

    transforms = get_test_transforms()

    data = transforms(Image.open(train_a_path))["image"].unsqueeze(0)
    pred = model(data)
    print(pred.shape)
    print("Train sample S vvvvv")
    pred = pred.detach().numpy()[0]
    pred_idices_argmax = pred.argsort()[-5:][::-1].astype(int)
    #print(pred_idices_argmax)
    for indice in pred_idices_argmax:
        indice_class = classes[indice]
        print(f"{indice_class} with proba {float(round(pred[indice], 5))}")

    print()

    data = transforms(Image.open(test_a_path))["image"].unsqueeze(0)
    pred = model(data)
    print(pred.shape)
    print("Test sample S vvvvv")
    pred = pred.detach().numpy()[0]
    pred_idices_argmax = pred.argsort()[-5:][::-1].astype(int)
    #print(pred_idices_argmax)
    for indice in pred_idices_argmax:
        indice_class = classes[indice]
        print(f"{indice_class} with proba {float(round(pred[indice], 5))}")

    print()