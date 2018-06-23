import os
from options.test_options import TestOptions
from models import create_model
from PIL import Image
import numpy as np 
import torchvision.transforms as transforms
import torch

def isSubString(Str,Filters):
    flag = False
    for substr in Filters:
        if substr in Str:
            flag = True
            break
    return flag

def getImageList(image_dir):
    filelist = []
    filters = [".jpg",".bmp",".png"]
    files = os.listdir(image_dir)
    for f in files:
        if isSubString(f,filters):
            filelist.append(f)
    return filelist

def loadImage(image_path,h,w):
    image = Image.open(image_path).convert("RGB")
    return image.resize((h,w))
    #nparray = np.asarray(image)
    #return nparrary.astype("float").reshape((1,c,h,w))

def getTransforms():
    t_list = []
    t_list.append(transforms.ToTensor())
    t_list.append(transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)))
    return transforms.Compose(t_list)

if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.nThreads = 1
    opt.batchSize = 1

    trans = getTransforms()
    model = create_model(opt)
    filelist = getImageList(opt.dataroot)
    for image_file in filelist:
        print image_file,"..."
        image = loadImage(opt.dataroot + "/" + image_file,opt.findSize,opt.findSize)
        image = trans(image)
        image = torch.unsqueeze(image, 0)
        if opt.which_direction == "AtoB":
            model.inferenceA2B(image)
            model.save_image(opt.results_dir + "/gb_" + image_file,model.fake_B)
        else:
            model.inferenceB2A(image)
            model.save_image(opt.results_dir + "/ga_" + image_file,model.fake_A) 
