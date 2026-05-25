from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import os
from pathlib import Path
from PIL import Image
import numpy as np
import random
import scipy.io
import math
import cv2
import ast

def fold_files(foldname):
    """All files in the fold should have the same extern"""
    jpg_files = [f for f in os.listdir(foldname) if f.lower().endswith(".png")]

    if len(jpg_files) < 1:
        raise ValueError('No images in the data folder')
        return None
    else:
        return jpg_files

class BSDS_Loader_orig(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/HED-BSDS', split='train', transform=False, threshold=0.1, ablation=False):
        self.root = root
        self.split = split
        #self.threshold = threshold * 256
        #print('Threshold for ground truth: %f on BSDS' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            if ablation:
                self.filelist = os.path.join(self.root, 'train_pair.lst')
            else:
                self.filelist = os.path.join(self.root, 'test.txt')
        elif self.split == 'test':
            if ablation:
                self.filelist = os.path.join(self.root, 'val.lst')
            else:
                self.filelist = os.path.join(self.root, 'train_raw_pair.txt')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
 
    def __getitem__(self, index):
    
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            #threshold = self.threshold
            #lb = lb[np.newaxis, :, :]
            #lb[lb == 0] = 0
            #lb[np.logical_and(lb>0, lb<threshold)] = 2
            #lb[lb >= threshold] = 1
            lb /= 255.

        else:
            img_file = self.filelist[index].rstrip() #.split(" ")[0].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, Path(img_file).stem
        else:
            img_name = Path(img_file).stem
            return img, img_name

class BSDS_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False, threshold=0.3, ablation=False):
        self.root = root
        self.split = split
        self.threshold = threshold * 256
        print('Threshold for ground truth: %f on BSDS' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            if ablation:
                self.filelist = os.path.join(self.root, 'train_pair.txt')
            else:
                self.filelist = os.path.join(self.root, 'train_my_bsds.txt')
        elif self.split == 'test':
            if ablation:
                self.filelist = os.path.join(self.root, 'val.txt')
            else:
                self.filelist = os.path.join(self.root, 'test.txt')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)


    def add_flipped_images(self,image,flag):
        temp = image
        if flag == "1_1":
          temp = cv2.flip(image,1)
        #print("here flip")
        return temp
    
    def rotatedRectWithMaxArea(self,w, h, angle):
      """
      Given a rectangle of size wxh that has been rotated by 'angle' (in
      radians), computes the width and height of the largest possible
      axis-aligned rectangle (maximal area) within the rotated rectangle.
      """
      if w <= 0 or h <= 0:
        return 0,0
    
      width_is_longer = w >= h
      side_long, side_short = (w,h) if width_is_longer else (h,w)
    
      # since the solutions for angle, -angle and 180-angle are all the same,
      # if suffices to look at the first quadrant and the absolute values of sin,cos:
      sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
      if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
      else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
    
      return wr,hr
      
    def rotate_bound(self,image, angle):
        # CREDIT: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))
    
    
    def rotate_max_area(self,image, angle):
        """ image: cv2 image matrix object
            angle: in degree
        """
        wr, hr = self.rotatedRectWithMaxArea(image.shape[1], image.shape[0],
                                        math.radians(angle))
        rotated = self.rotate_bound(image, angle)
        if rotated.ndim == 3:
            h, w, _ = rotated.shape
        else:
            h, w = rotated.shape
        y1 = h//2 - int(hr/2)
        y2 = y1 + int(hr)
        x1 = w//2 - int(wr/2)
        x2 = x1 + int(wr)
        return rotated[y1:y2, x1:x2]
    
    
    def __getitem__(self, index):
    
    
        img_processed = False
        scale = 1.0
        
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            #print(img_file)
            if "scale" in img_file:
                scale = 0.0
                
            if ".mat" in lb_file:
                lb = scipy.io.loadmat(os.path.join(self.root, lb_file))
                gt = [g.item()[1] for g in lb["groundTruth"][0]]

                
                #apply augmentation to all label
                flip_label = img_file.split("/")[3][-3:]
                rot_label = img_file.split("/")[3].split("_")[0]
                gt = [np.pad(g.astype(np.float32),((32,31),(32,31)), mode="symmetric") for g in gt]
                gt = [self.rotate_max_area(self.add_flipped_images(g,flip_label),360 -float(rot_label)) for g in gt]
                
                
                lb = gt          
                lb_sum = 1.0 * sum(lb) / len(lb)
                
                #lb_sum = np.squeeze(lb_sum[:, :, 0])
                threshold = self.threshold / 256.
                lb_sum[lb_sum == 0] = 0
                lb_sum[lb_sum >= threshold] = 1
                lb_sum[np.logical_and(lb_sum>0, lb_sum<threshold)] = 2
                lb.insert(0,lb_sum)
                #print(np.sum(lb_sum))

                lb  = [g[np.newaxis, :, :].astype(np.float32) for g in lb]
                if rot_label not in ["0.0","90.0","180.0","270.0","360.0"]:
                    scale = 0.0
                    lb = lb[0]
            else:
                lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
                
                if lb.ndim == 3:
                    lb = np.squeeze(lb[:, :, 0])
                assert lb.ndim == 2

                
                threshold = self.threshold
                lb = lb[np.newaxis, :, :]
                lb[lb == 0] = 0
                lb[np.logical_and(lb>0, lb<threshold)] = 2
                lb[lb >= threshold] = 1

            
        else:
            img_file = self.filelist[index].rstrip().split(" ")[0]

        #print(self.root , " " , img_file)
        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb,scale
        else:
            img_name = Path(img_file).stem
            return img, img_name



class BSDS_VOCLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False, threshold=0.3, ablation=False):
        self.root = root
        self.split = split
        self.threshold = threshold * 256
        print('Threshold for ground truth: %f on BSDS_VOC' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            if ablation:
                self.filelist = os.path.join(self.root, 'bsds_pascal_train200_pair.lst')
            else:
                self.filelist = os.path.join(self.root, 'train_my.txt')
        elif self.split == 'test':
            if ablation:
                self.filelist = os.path.join(self.root, 'val.lst')
            else:
                self.filelist = os.path.join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)


    def add_flipped_images(self,image,flag):
        temp = image
        if flag == "1_1":
          temp = cv2.flip(image,1)
        #print("here flip")
        return temp
    
    def rotatedRectWithMaxArea(self,w, h, angle):
      """
      Given a rectangle of size wxh that has been rotated by 'angle' (in
      radians), computes the width and height of the largest possible
      axis-aligned rectangle (maximal area) within the rotated rectangle.
      """
      if w <= 0 or h <= 0:
        return 0,0
    
      width_is_longer = w >= h
      side_long, side_short = (w,h) if width_is_longer else (h,w)
    
      # since the solutions for angle, -angle and 180-angle are all the same,
      # if suffices to look at the first quadrant and the absolute values of sin,cos:
      sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
      if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
      else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
    
      return wr,hr
      
    def rotate_bound(self,image, angle):
        # CREDIT: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))
    
    
    def rotate_max_area(self,image, angle):
        """ image: cv2 image matrix object
            angle: in degree
        """
        wr, hr = self.rotatedRectWithMaxArea(image.shape[1], image.shape[0],
                                        math.radians(angle))
        rotated = self.rotate_bound(image, angle)
        if rotated.ndim == 3:
            h, w, _ = rotated.shape
        else:
            h, w = rotated.shape
        y1 = h//2 - int(hr/2)
        y2 = y1 + int(hr)
        x1 = w//2 - int(wr/2)
        x2 = x1 + int(wr)
        return rotated[y1:y2, x1:x2]
    
    
    def __getitem__(self, index):
    
    
        img_processed = False
        scale = 1.0
        
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            #print(img_file)
            if "scale" in img_file:
                scale = 0.0
                
            if ".mat" in lb_file:
                lb = scipy.io.loadmat(os.path.join(self.root, lb_file))
                gt = [g.item()[1] for g in lb["groundTruth"][0]]
                

                
                #apply augmentation to all label
                flip_label = img_file.split("/")[3][-3:]
                rot_label = img_file.split("/")[3].split("_")[0]
                gt = [np.pad(g.astype(np.float32),((32,31),(32,31)), mode="symmetric") for g in gt]
                gt = [self.rotate_max_area(self.add_flipped_images(g,flip_label),360 -float(rot_label)) for g in gt]
                
                
                lb = gt          
                lb_sum = 1.0 * sum(lb) / len(lb)
                
                #lb_sum = np.squeeze(lb_sum[:, :, 0])
                threshold = self.threshold / 256.
                lb_sum[lb_sum == 0] = 0
                lb_sum[lb_sum >= threshold] = 1
                lb_sum[np.logical_and(lb_sum>0, lb_sum<threshold)] = 2
                lb.insert(0,lb_sum)
                #print(np.sum(lb_sum))

                lb  = [g[np.newaxis, :, :].astype(np.float32) for g in lb]
            else:
                lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
                
                if lb.ndim == 3:
                    lb = np.squeeze(lb[:, :, 0])
                assert lb.ndim == 2

                
                threshold = self.threshold
                lb = lb[np.newaxis, :, :]
                lb[lb == 0] = 0
                lb[np.logical_and(lb>0, lb<threshold)] = 2
                lb[lb >= threshold] = 1

            
        else:
            img_file = self.filelist[index].rstrip().split(" ")[0]

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb,scale
        else:
            img_name = Path(img_file).stem
            return img, img_name


class Multicue_Loader(data.Dataset):
    """
    Dataloader for Multicue
    """
    def __init__(self, root='data/', split='train', transform=False, threshold=0.3, setting=['boundary', '1']):
        """
        setting[0] should be 'boundary' or 'edge'
        setting[1] should be '1' or '2' or '3'
        """
        self.root = root
        self.split = split
        self.threshold = threshold * 256
        self.gt_type = ""
        if setting[0] == "boundary":
            self.gt_type = "boundaries"
        else:
            self.gt_type = "edges"
        print('Threshold for ground truth: %f on setting %s' % (self.threshold, str(setting)))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            self.filelist = os.path.join(
                    self.root, 'train_pair_%s_set_%s.lst' % (setting[0], setting[1]))
        elif self.split == 'test':
            self.filelist = os.path.join(
                    self.root, 'test_%s_set_%s.lst' % (setting[0], setting[1]))
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def add_flipped_images(self,image,flag):
        temp = image
        if flag == "1_1":
          temp = cv2.flip(image,1)
        #print("here flip")
        return temp
    
    def rotatedRectWithMaxArea(self,w, h, angle):
      """
      Given a rectangle of size wxh that has been rotated by 'angle' (in
      radians), computes the width and height of the largest possible
      axis-aligned rectangle (maximal area) within the rotated rectangle.
      """
      if w <= 0 or h <= 0:
        return 0,0
    
      width_is_longer = w >= h
      side_long, side_short = (w,h) if width_is_longer else (h,w)
    
      # since the solutions for angle, -angle and 180-angle are all the same,
      # if suffices to look at the first quadrant and the absolute values of sin,cos:
      sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
      if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
      else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
    
      return wr,hr
      
    def rotate_bound(self,image, angle):
        # CREDIT: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))
    
    
    def rotate_max_area(self,image, angle):
        """ image: cv2 image matrix object
            angle: in degree
        """
        wr, hr = self.rotatedRectWithMaxArea(image.shape[1], image.shape[0],
                                        math.radians(angle))
        rotated = self.rotate_bound(image, angle)
        if rotated.ndim == 3:
            h, w, _ = rotated.shape
        else:
            h, w = rotated.shape
        y1 = 0 #h//2 - int(hr/2)
        y2 = y1 + int(hr)
        x1 = 0 #w//2 - int(wr/2)
        x2 = x1 + int(wr)
        print(y1, " " , y2, " " ,x1 , " " , x2)
        return rotated[y1:y2, x1:x2]
        

    def rotate_max_area_opencv(self,image, angle):
        """ image: cv2 image matrix object
            angle: in degree
        """

        if angle == 0 or angle == 360:
            rotated = image
        elif angle == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return rotated

    def random_crop(self, img, gt, crop_size = (500,500) ):
        """
        img: 3xHxW or HxWx3 RGB image (tensor or PIL)
        gt: 1xHxW or HxW GT mask (tensor or PIL)
        crop_size: (crop_height, crop_width)
        """
        
        _,h, w = img.shape
        crop_h, crop_w = crop_size
    
        if h < crop_h or w < crop_w:
            raise ValueError("Crop size must be smaller than image size.")
    
        # Random crop coordinates
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
    
        img_cropped = img[:,top:top + crop_h, left:left + crop_w]
        if not isinstance(gt, list):
            gt_cropped  = gt[:,top:top + crop_h, left:left + crop_w]
        else:
            gt_cropped = [gt_i[:,top:top + crop_h, left:left + crop_w] for gt_i in gt]
    
        return img_cropped, gt_cropped
        
    def __getitem__(self, index):
        #images/s1.0/r90.0_flip_1/4710_left_0035.png mat/4710_left_0035.h5   
        img_processed = False
        scale = 1.0
        
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            
            if "s0.5" in img_file or "s0.75" in img_file:
                scale = 0.0
                
            if ".h5" in lb_file:
                with h5py.File(os.path.join(self.root, lb_file), 'r') as f:
                    data = f[self.gt_type][:]
                
                gt = [data[i] for i in range(data.shape[0])]

                
                #apply augmentation to all label
                flip_rot_label = img_file.split("/")[2]
                flip_label = "1_0"
                if "flip" in flip_rot_label:
                    flip_label = "1_1"
                rot_label = img_file.split("/")[2].split("_")[0][1:]
                gt = [self.add_flipped_images(self.rotate_max_area_opencv(g,360 -float(rot_label)),flip_label) for g in gt]
                #print("gt_0. shape ", gt[0].shape)
                
                lb = gt          
                lb_sum = 1.0 * sum(lb) / len(lb)
                
                #lb_sum = np.squeeze(lb_sum[:, :, 0])
                threshold = self.threshold / 256.
                lb_sum[lb_sum == 0] = 0
                lb_sum[lb_sum >= threshold] = 1
                lb_sum[np.logical_and(lb_sum>0, lb_sum<threshold)] = 2
                #print("lb sum shape: " ,lb_sum.shape)
                lb.insert(0,lb_sum)
                #print(np.sum(lb_sum))

                lb  = [g[np.newaxis, :, :].astype(np.float32) for g in lb]

            else:            
                lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
                if lb.ndim == 3:
                    lb = np.squeeze(lb[:, :, 0])
                assert lb.ndim == 2
                threshold = self.threshold
                lb = lb[np.newaxis, :, :]
                lb[lb == 0] = 0
                lb[np.logical_and(lb>0, lb<threshold)] = 2
                lb[lb >= threshold] = 1
                scale = 0.0
                            
        else:
            img_file = self.filelist[index].rstrip()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if scale != 0.0 and self.split == "train":
            img, lb = self.random_crop(img,lb)
                    
        if self.split == "train":           
            return img, lb,scale
        else:
            img_name = Path(img_file).stem
            return img, img_name

class BIPED_Loader(data.Dataset):
    """
    Dataloader for NYUDv2
    """
    def __init__(self, root='data/', split='train', transform=False, threshold=0.4, setting=['image']):
        """
        There is no threshold for NYUDv2 since it is singlely annotated
        setting should be 'image' or 'hha'
        """
        self.root = root
        self.split = split
        self.threshold = 128
        print('Threshold for ground truth: %f on setting %s' % (self.threshold, str(setting)))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            self.filelist = os.path.join(
                    self.root, 'train_pair.lst' )
        elif self.split == 'test':
            self.filelist = os.path.join(
                    self.root, 'test_pair.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.read()

        self.filelist = ast.literal_eval(self.filelist)

    def __len__(self):
        return len(self.filelist)

    def random_crop(self, img, gt, crop_size = (500,500) ):
        """
        img: 3xHxW or HxWx3 RGB image (tensor or PIL)
        gt: 1xHxW or HxW GT mask (tensor or PIL)
        crop_size: (crop_height, crop_width)
        """
        
        _,h, w = img.shape
        crop_h, crop_w = crop_size
    
        if h < crop_h or w < crop_w:
            raise ValueError("Crop size must be smaller than image size.")
    
        # Random crop coordinates
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
    
        img_cropped = img[:,top:top + crop_h, left:left + crop_w]
        if not isinstance(gt, list):
            gt_cropped  = gt[:,top:top + crop_h, left:left + crop_w]
        else:
            gt_cropped = [gt_i[:,top:top + crop_h, left:left + crop_w] for gt_i in gt]
    
        return img_cropped, gt_cropped
            
    def __getitem__(self, index):
        scale = 1.0
        if self.split == "train":
            file_names = self.filelist[index]
            img_file = file_names[0]
            lb_file = file_names[1]
            
            if "rot" in img_file:
                img_file_part = img_file.split("_")
                if img_file_part[2] not in ["90","180","270"]:
                    scale = 0.0                
            pil_image = Image.open(os.path.join(self.root, lb_file))

            lb = np.array(pil_image, dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb>0, lb<threshold)] = 2
            lb[lb >= threshold] = 1
            
        else:
            img_file = self.filelist[index][0]

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if scale != 0.0 and self.split == "train":
            img, lb = self.random_crop(img,lb)
            
        if self.split == "train":
            return img, lb, scale
        else:
            img_name = Path(img_file).stem
            return img, img_name

class NYUD_Loader(data.Dataset):
    """
    Dataloader for NYUDv2
    """
    def __init__(self, root='data/', split='train', transform=False, threshold=0.4, setting=['image']):
        """
        There is no threshold for NYUDv2 since it is singlely annotated
        setting should be 'image' or 'hha'
        """
        self.root = root
        self.split = split
        self.threshold = 128
        print('Threshold for ground truth: %f on setting %s' % (self.threshold, str(setting)))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':
            self.filelist = os.path.join(
                    self.root, '%s-train.txt' % (setting[0]))
        elif self.split == 'test':
            self.filelist = os.path.join(
                    self.root, '%s-test.txt' % (setting[0]))
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        scale = 1.0
        if self.split == "train":
            img_file, lb_file, scale = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            scale = float(scale.strip())
            pil_image = Image.open(os.path.join(self.root, lb_file))
            if scale < 0.99: # which means it < 1.0
                W = int(scale * pil_image.width)
                H = int(scale * pil_image.height)
                pil_image = pil_image.resize((W, H))
            lb = np.array(pil_image, dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb>0, lb<threshold)] = 2
            lb[lb >= threshold] = 1
            
        else:
            img_file, _ = self.filelist[index].rstrip().split()

        with open(os.path.join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            if scale < 0.9:
                img = img.resize((W, H))
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb, scale
        else:
            img_name = Path(img_file).stem
            return img, img_name

class Custom_Loader(data.Dataset):
    """
    Custom Dataloader
    """
    def __init__(self, root='data/'):
        self.root = root
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        self.imgList = fold_files(os.path.join(root))

    def __len__(self):
        return len(self.imgList)
    
    def __getitem__(self, index):

        with open(os.path.join(self.root, self.imgList[index]), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            #w, h = img.size
            #min_dim = min(w, h)
            #img = img.crop(
            #    ((w - min_dim) // 2, (h - min_dim) // 2,
            #     (w + min_dim) // 2, (h + min_dim) // 2)
            #).resize((512, 512))

        img = self.transform(img)

        filename = Path(self.imgList[index]).stem

        return img, filename
