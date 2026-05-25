import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
#import kornia as K

    
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,k=3):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=k,stride=1,padding=(k-1)//2,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2, k=3):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=k,stride=1,padding=(k-1)//2,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2, k =3):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t, k=k),
            Recurrent_block(ch_out,t=t, k =k)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
     
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = F.interpolate(d5, (x4.size(2), x4.size(3)), mode="bilinear", align_corners=False)

        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        
        d2 = F.interpolate(d2, (x1.size(2), x1.size(3)), mode="bilinear", align_corners=False)


        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return torch.sigmoid(d1)


class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,t=2,k = 3, useResize=True):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t, k=k)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t, k=k)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t, k=k)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t, k=k)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t, k=k)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512, k=k)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t, k=k)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256, k=k)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t, k=k)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128, k=k)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t, k=k)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64, k=k)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t, k=k)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)

        r_second = 4
        f_second = np.array([list(range(1, r_second + 1)) + [r_second + 1] + list(range(r_second, 0, -1))]) / (r_second + 1) ** 2        
        weighth_2_1 = torch.from_numpy(np.ascontiguousarray(f_second[::-1, ::-1])).float()
        weigth_2_1_t = torch.from_numpy(np.ascontiguousarray(f_second[::-1, ::-1]).T).float()
        self.conv_2_1_blur = nn.Conv2d(1, 1, (1, 9), padding='same', bias=False)
        self.conv_2_1_t_blur = nn.Conv2d(1, 1, (9, 1), padding='same', bias=False)
        self.conv_2_1_blur.weight = nn.Parameter(weighth_2_1.unsqueeze(0).unsqueeze(0))
        self.conv_2_1_t_blur.weight = nn.Parameter(weigth_2_1_t.unsqueeze(0).unsqueeze(0))
        
        for param in self.conv_2_1_t_blur.parameters():
            param.requires_grad = False

        for param in self.conv_2_1_blur.parameters():
            param.requires_grad = False
            
        self.useResize = useResize
            
    def forward(self,x,blurType):
    
        #x = self.conv_2_1_t_blur(self.conv_2_1_blur(x))
        
        with torch.no_grad():
            x *= 255        
            if blurType =="bilateral3x3_0.1":
              x = K.filters.bilateral_blur(x,(3,3),0.1, (1.5,1.5))
            
            elif blurType =="bilateral5x5_0.1":
              x = K.filters.bilateral_blur(x,(5,5),0.1, (1.5,1.5))
            
            elif blurType =="bilateral3x3_0.3":
              x = K.filters.bilateral_blur(x,(3,3),0.3, (1.5,1.5))
            
            elif blurType =="bilateral5x5_0.3":
              x = K.filters.bilateral_blur(x,(5,5),0.3, (1.5,1.5))           
            
            elif blurType =="box_blur3x3":
              x = K.filters.box_blur(x,(3,3))
            
            elif blurType =="box_blur5x5":
              x = K.filters.box_blur(x,(5,5))
              
            elif blurType =="gaussian_blur2d3x3":
              x = K.filters.gaussian_blur2d(x,(3,3),(1.5, 1.5))
            
            elif blurType =="gaussian_blur2d5x5":
              x = K.filters.gaussian_blur2d(x,(5,5),(1.5, 1.5))
            
            elif blurType =="median_blur3x3":
              x = K.filters.median_blur(x,(3,3))
            
            elif blurType =="median_blur5x5":
              x = K.filters.median_blur(x,(5,5))
            
            x /= 255.              
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path

        d5 = self.Up5(x5)
        
        if self.useResize:
          d5 = F.interpolate(d5, (x4.size(2), x4.size(3)), mode="bilinear", align_corners=False)
        elif d5.size(2) != x4.size(2):
          d5 = F.pad(d5,(0,0,1,0), "constant",0)
          
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        if self.useResize:
          d4 = F.interpolate(d4, (x3.size(2), x3.size(3)), mode="bilinear", align_corners=False)
        elif d4.size(2) != x3.size(2):
          d4 = F.pad(d4,(0,0,1,0), "constant",0)
        
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        if self.useResize:
          d3 = F.interpolate(d3, (x2.size(2), x2.size(3)), mode="bilinear", align_corners=False)
        elif d3.size(2) != x2.size(2):
          d3 = F.pad(d3,(0,0,1,0), "constant",0)
        
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        
        if self.useResize:
          d2 = F.interpolate(d2, (x1.size(2), x1.size(3)), mode="bilinear", align_corners=False)
        elif d2.size(2) != x1.size(2):
          d2 = F.pad(d2,(0,0,1,0), "constant",0)
        
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return torch.sigmoid(d1)
        
        
        
        
class conv_block_nested(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch, track_running_stats =False)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch, track_running_stats =False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class AttentionGroup(nn.Module):
    def __init__(self, num_channels):
        super(AttentionGroup, self).__init__()
        self.conv1 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv_1x1 = nn.Conv2d(num_channels, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        s = torch.softmax(self.conv_1x1(x), dim=1)

        att = s[:,0,:,:].unsqueeze(1) * x1 + s[:,1,:,:].unsqueeze(1) * x2 \
            + s[:,2,:,:].unsqueeze(1) * x3

        return x + att


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=1, out_ch=1, useResize = False):
        super(NestedUNet, self).__init__()

        n1 = 64 
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)
           
        self.useResize = useResize
        
        #self.sigmoid =PSigmoid()
        
    def padding(self,x,y):
        y_up = self.Up(y)
        
        if y_up.size(2) == x.size(2) and y_up.size(3) == x.size(3):
            return y_up
            
        _,_,pad_num_w,pad_num_h = x.size()
        _,_,pad_num_w_m,pad_num_h_m = y_up.size()
        return F.pad(y_up,(pad_num_h-pad_num_h_m,0,pad_num_w-pad_num_w_m,0), "constant",0)         

    def forward(self, x,blurType="median_blur5x5"):
        
        #x = self.conv_2_1_t_blur(self.conv_2_1_blur(x))

        
        with torch.no_grad():
                   
            if blurType =="bilateral3x3_0.1_1.5":
              x = K.filters.bilateral_blur(x,(3,3),0.1, (1.5,1.5))
            
            elif blurType =="bilateral5x5_0.2_1.5":
              x = K.filters.bilateral_blur(x,(5,5),0.2, (1.5,1.5))
            
            elif blurType =="bilateral3x3_0.3_1.5":
              x = K.filters.bilateral_blur(x,(3,3),0.3, (1.5,1.5))
            
            elif blurType =="bilateral5x5_0.3_1.5":
              x = K.filters.bilateral_blur(x,(5,5),0.3, (1.5,1.5))           
            
            elif blurType =="box_blur3x3":
              x = K.filters.box_blur(x,(3,3))
            
            elif blurType =="box_blur5x5":
              x = K.filters.box_blur(x,(5,5))
              
            elif blurType =="gaussian_blur2d3x3_0.8":
              x = K.filters.gaussian_blur2d(x,(3,3),(0.8, 0.8))
            
            elif blurType =="gaussian_blur2d5x5_1.1":
              x = K.filters.gaussian_blur2d(x,(5,5),(1.1, 1.1))

            elif blurType =="gaussian_blur2d3x3_1.5":
              x = K.filters.gaussian_blur2d(x,(3,3),(1.5, 1.5))

            elif blurType =="gaussian_blur2d5x5_1.5":
              x = K.filters.gaussian_blur2d(x,(5,5),(1.5, 1.5))
              
            elif blurType =="median_blur3x3":
              x = K.filters.median_blur(x,(3,3))
            
            elif blurType =="median_blur5x5":
              x_blur = K.filters.median_blur(x,(5,5))
              mask = (x >= 0.1) & (x_blur < 0.1)
              x_blur[mask] = x[mask]
              x = x_blur
        
            elif blurType =="median_blur7x7":
              x = K.filters.median_blur(x,(7,7))
              
                        
          
        x0_0 = torch.nn.functional.relu(self.conv0_0(x))
        x1_0 = self.conv1_0(x0_0) #self.pool(x0_0))
        
        if self.useResize:
          x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(self.Up(x1_0), (x0_0.size(2), x0_0.size(3)) ,mode="bilinear", align_corners=False) ], 1))
        else:        
          x0_1 = self.conv0_1(torch.cat([x0_0, self.padding(x0_0,x1_0) ], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))

        if self.useResize:
            x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        else:         
            x1_1 = self.conv1_1(torch.cat([x1_0, self.padding(x1_0,x2_0)], 1))

        
        if self.useResize:
          x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(self.Up(x1_1), (x0_0.size(2), x0_0.size(3)) ,mode="bilinear", align_corners=False)], 1))
        else:
          x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.padding(x0_0,x1_1)], 1))

        #x3_0 = self.att3(self.conv3_0(self.pool(x2_0)))
        x3_0 = self.conv3_0(self.pool(x2_0))
       
        x2_1 = self.conv2_1(torch.cat([x2_0, self.padding(x2_0,x3_0) ], 1))
       
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.padding(x1_0,x2_1)], 1))
        
        if self.useResize:
          x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(self.Up(x1_2), (x0_0.size(2), x0_0.size(3)) ,mode="bilinear", align_corners=False)], 1))
        else:        
          x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.padding(x0_1,x1_2)], 1))
          
          
        #x4_0 = self.att4(self.conv4_0(self.pool(x3_0)))
        x4_0 = self.conv4_0(self.pool(x3_0))
        if self.useResize:
          x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(self.Up(x4_0), (x3_0.size(2), x3_0.size(3)) ,mode="bilinear", align_corners=False)], 1))
        else:
          x3_1 = self.conv3_1(torch.cat([x3_0, self.padding(x3_0,x4_0)], 1))

          
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.padding(x2_0,x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.padding(x1_2,x2_2)], 1))
        if self.useResize:
          x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(self.Up(x1_3), (x0_0.size(2), x0_0.size(3)) ,mode="bilinear", align_corners=False)], 1))
        else:
          x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.padding(x0_0,x1_3)], 1))
          
        output = self.final(x0_4)
        
               
        return torch.sigmoid(output)



class conv_block_unet3p(nn.Module):
    def __init__(self, in_c, out_c, act=True):
        super().__init__()

        layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]

        if act == True:
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.Sequential(
            conv_block_unet3p(in_c, out_c),
            conv_block_unet3p(out_c, out_c)
        )
        self.p1 = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.c1(x)
        p = self.p1(x)
        return x, p

class unet3plus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.e5 = nn.Sequential(
            conv_block_unet3p(512, 1024),
            conv_block_unet3p(1024, 1024)
        )

        """ Decoder 4 """
        self.e1_d4 = conv_block_unet3p(64, 64)
        self.e2_d4 = conv_block_unet3p(128, 64)
        self.e3_d4 = conv_block_unet3p(256, 64)
        self.e4_d4 = conv_block_unet3p(512, 64)
        self.e5_d4 = conv_block_unet3p(1024, 64)

        self.d4 = conv_block_unet3p(64*5, 64)

        """ Decoder 3 """
        self.e1_d3 = conv_block_unet3p(64, 64)
        self.e2_d3 = conv_block_unet3p(128, 64)
        self.e3_d3 = conv_block_unet3p(256, 64)
        self.e4_d3 = conv_block_unet3p(64, 64)
        self.e5_d3 = conv_block_unet3p(1024, 64)

        self.d3 = conv_block_unet3p(64*5, 64)

        """ Decoder 2 """
        self.e1_d2 = conv_block_unet3p(64, 64)
        self.e2_d2 = conv_block_unet3p(128, 64)
        self.e3_d2 = conv_block_unet3p(64, 64)
        self.e4_d2 = conv_block_unet3p(64, 64)
        self.e5_d2 = conv_block_unet3p(1024, 64)

        self.d2 = conv_block_unet3p(64*5, 64)

        """ Decoder 1 """
        self.e1_d1 = conv_block_unet3p(64, 64)
        self.e2_d1 = conv_block_unet3p(64, 64)
        self.e3_d1 = conv_block_unet3p(64, 64)
        self.e4_d1 = conv_block_unet3p(64, 64)
        self.e5_d1 = conv_block_unet3p(1024, 64)

        self.d1 = conv_block_unet3p(64*5, 64)

        """ Output """
        self.y1 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self, x,blurType):
        
        
        with torch.no_grad():
            x *= 255        
            if blurType =="bilateral3x3_0.1":
              x = K.filters.bilateral_blur(x,(3,3),0.1, (1.5,1.5))
            
            elif blurType =="bilateral5x5_0.1":
              x = K.filters.bilateral_blur(x,(5,5),0.1, (1.5,1.5))
            
            elif blurType =="bilateral3x3_0.3":
              x = K.filters.bilateral_blur(x,(3,3),0.3, (1.5,1.5))
            
            elif blurType =="bilateral5x5_0.3":
              x = K.filters.bilateral_blur(x,(5,5),0.3, (1.5,1.5))           
            
            elif blurType =="box_blur3x3":
              x = K.filters.box_blur(x,(3,3))
            
            elif blurType =="box_blur5x5":
              x = K.filters.box_blur(x,(5,5))
              
            elif blurType =="gaussian_blur2d3x3":
              x = K.filters.gaussian_blur2d(x,(3,3),(1.5, 1.5))
            
            elif blurType =="gaussian_blur2d5x5":
              x = K.filters.gaussian_blur2d(x,(5,5),(1.5, 1.5))
            
            elif blurType =="median_blur3x3":
              x = K.filters.median_blur(x,(3,3))
            
            elif blurType =="median_blur5x5":
              x = K.filters.median_blur(x,(5,5))

            elif blurType =="median_blur7x7":
              x = K.filters.median_blur(x,(7,7))
              
            x /= 255.            
        
        
        """ Encoder """
        e1, p1 = self.e1(x)
        e2, p2 = self.e2(p1)
        e3, p3 = self.e3(p2)
        e4, p4 = self.e4(p3)

        """ Bottleneck """
        e5 = self.e5(p4)

        """ Decoder 4 """
        e1_d4 = F.max_pool2d(e1, kernel_size=8, stride=8)
        e1_d4 = self.e1_d4(e1_d4)

        e2_d4 = F.max_pool2d(e2, kernel_size=4, stride=4)
        e2_d4 = self.e2_d4(e2_d4)

        e3_d4 = F.max_pool2d(e3, kernel_size=2, stride=2)
        e3_d4 = self.e3_d4(e3_d4)

        e4_d4 = self.e4_d4(e4)

        e5_d4 = F.interpolate(e5, scale_factor=2, mode="bilinear", align_corners=True)
        e5_d4 = self.e5_d4(e5_d4)



        d4 = torch.cat([e1_d4, e2_d4, e3_d4, e4_d4, F.pad(e5_d4,(0,0,1,0), "constant",0)], dim=1)
        d4 = self.d4(d4)

        """ Decoder 3 """
        e1_d3 = F.max_pool2d(e1, kernel_size=4, stride=4)
        e1_d3 = self.e1_d3(e1_d3)

        e2_d3 = F.max_pool2d(e2, kernel_size=2, stride=2)
        e2_d3 = self.e2_d3(e2_d3)

        e3_d3 = self.e3_d3(e3)

        e4_d3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)
        e4_d3 = self.e4_d3(e4_d3)

        e5_d3 = F.interpolate(e5, scale_factor=4, mode="bilinear", align_corners=True)
        e5_d3 = self.e5_d3(e5_d3)

 

        
        d3 = torch.cat([e1_d3, e2_d3, e3_d3, e4_d3, F.pad(e5_d3,(0,0,1,1), "constant",0)], dim=1)
        d3 = self.d3(d3)

        """ Decoder 2 """
        e1_d2 = F.max_pool2d(e1, kernel_size=2, stride=2)
        e1_d2 = self.e1_d2(e1_d2)

        e2_d2 = self.e2_d2(e2)

        e3_d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)
        e3_d2 = self.e3_d2(e3_d2)

        e4_d2 = F.interpolate(d4, scale_factor=4, mode="bilinear", align_corners=True)
        e4_d2 = self.e4_d2(e4_d2)

        e5_d2 = F.interpolate(e5, scale_factor=8, mode="bilinear", align_corners=True)
        e5_d2 = self.e5_d2(e5_d2)


        
        d2 = torch.cat([e1_d2, e2_d2, e3_d2, e4_d2, F.pad(e5_d2,(0,0,2,2), "constant",0)], dim=1)
        d2 = self.d2(d2)

        """ Decoder 1 """
        e1_d1 = self.e1_d1(e1)

        e2_d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)
        e2_d1 = self.e2_d1(e2_d1)

        e3_d1 = F.interpolate(d3, scale_factor=4, mode="bilinear", align_corners=True)
        e3_d1 = self.e3_d1(e3_d1)

        e4_d1 = F.interpolate(d4, scale_factor=8, mode="bilinear", align_corners=True)
        e4_d1 = self.e4_d1(e4_d1)

        e5_d1 = F.interpolate(e5, scale_factor=16, mode="bilinear", align_corners=True)
        e5_d1 = self.e5_d1(e5_d1)
        
        d1 = torch.cat([e1_d1, F.pad(e2_d1,(0,0,1,0), "constant",0), F.pad(e3_d1,(0,0,1,0), "constant",0), F.pad(e4_d1,(0,0,1,0), "constant",0), F.pad(e5_d1,(0,0,5,4), "constant",0)], dim=1)
        d1 = self.d1(d1)

        """ Output """
        y1 = self.y1(d1)

        return torch.sigmoid(y1)
        
class MT_UNet(nn.Module):

    def __init__(self):
        super(MT_UNet, self).__init__()

        
        self.model = mtunet.MTUNet()

    def forward(self, x,blurType):
        
        x = self.model(x, blurType)
        return x

class Efficient_UPerNet(nn.Module):

    def __init__(self):
        super(Efficient_UPerNet, self).__init__()

        
        self.model = smp.UPerNet(
                   encoder_name="efficientnet-b6",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                   encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                   in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                   classes=1,                      # model output channels (number of classes in your dataset)
                   activation = None
                   )

    def forward(self, x, blurType):

        with torch.no_grad():
            x *= 255        
            if blurType =="bilateral3x3_0.1":
              x = K.filters.bilateral_blur(x,(3,3),0.1, (1.5,1.5))
            
            elif blurType =="bilateral5x5_0.1":
              x = K.filters.bilateral_blur(x,(5,5),0.1, (1.5,1.5))
            
            elif blurType =="bilateral3x3_0.3":
              x = K.filters.bilateral_blur(x,(3,3),0.3, (1.5,1.5))
            
            elif blurType =="bilateral5x5_0.3":
              x = K.filters.bilateral_blur(x,(5,5),0.3, (1.5,1.5))           
            
            elif blurType =="box_blur3x3":
              x = K.filters.box_blur(x,(3,3))
            
            elif blurType =="box_blur5x5":
              x = K.filters.box_blur(x,(5,5))
              
            elif blurType =="gaussian_blur2d3x3":
              x = K.filters.gaussian_blur2d(x,(3,3),(1.5, 1.5))
            
            elif blurType =="gaussian_blur2d5x5":
              x = K.filters.gaussian_blur2d(x,(5,5),(1.5, 1.5))
            
            elif blurType =="median_blur3x3":
              x = K.filters.median_blur(x,(3,3))
            
            elif blurType =="median_blur5x5":
              x = K.filters.median_blur(x,(5,5))

            elif blurType =="median_blur7x7":
              x = K.filters.median_blur(x,(7,7))
              
            x /= 255.   
             
        x = F.pad(x,(8,8,11,12), "constant",0)     
        x = torch.sigmoid(self.model(x))
        x = x[:,:,11:-12,8:-8]

        return x
        



class Efficient_UNetPlusPlus(nn.Module):

    def __init__(self):
        super(Efficient_UNetPlusPlus, self).__init__()

        
        self.model = smp.UnetPlusPlus(
                   encoder_name="efficientnet-b6",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                   encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                   in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                   classes=1,                      # model output channels (number of classes in your dataset)
                   activation = None
                   )

    def forward(self, x, blurType):

        with torch.no_grad():
            x *= 255        
            if blurType =="bilateral3x3_0.1":
              x = K.filters.bilateral_blur(x,(3,3),0.1, (1.5,1.5))
            
            elif blurType =="bilateral5x5_0.1":
              x = K.filters.bilateral_blur(x,(5,5),0.1, (1.5,1.5))
            
            elif blurType =="bilateral3x3_0.3":
              x = K.filters.bilateral_blur(x,(3,3),0.3, (1.5,1.5))
            
            elif blurType =="bilateral5x5_0.3":
              x = K.filters.bilateral_blur(x,(5,5),0.3, (1.5,1.5))           
            
            elif blurType =="box_blur3x3":
              x = K.filters.box_blur(x,(3,3))
            
            elif blurType =="box_blur5x5":
              x = K.filters.box_blur(x,(5,5))
              
            elif blurType =="gaussian_blur2d3x3":
              x = K.filters.gaussian_blur2d(x,(3,3),(1.5, 1.5))
            
            elif blurType =="gaussian_blur2d5x5":
              x = K.filters.gaussian_blur2d(x,(5,5),(1.5, 1.5))
            
            elif blurType =="median_blur3x3":
              x = K.filters.median_blur(x,(3,3))
            
            elif blurType =="median_blur5x5":
              x = K.filters.median_blur(x,(5,5))

            elif blurType =="median_blur7x7":
              x = K.filters.median_blur(x,(7,7))
              
            x /= 255.   
             
        x = F.pad(x,(8,8,11,12), "constant",0)     
        x = torch.sigmoid(self.model(x))
        x = x[:,:,11:-12,8:-8]

        return x
        


class UnetAttention(nn.Module):
    def __init__(self):
        super(UnetAttention, self).__init__()
        self.encoder = unet_att.Encoder()
        self.decoder = unet_att.Decoder()

    def forward(self, x,blurType):
    
        with torch.no_grad():
            x *= 255        
            if blurType =="bilateral3x3_0.1":
              x = K.filters.bilateral_blur(x,(3,3),0.1, (1.5,1.5))
            
            elif blurType =="bilateral5x5_0.1":
              x = K.filters.bilateral_blur(x,(5,5),0.1, (1.5,1.5))
            
            elif blurType =="bilateral3x3_0.3":
              x = K.filters.bilateral_blur(x,(3,3),0.3, (1.5,1.5))
            
            elif blurType =="bilateral5x5_0.3":
              x = K.filters.bilateral_blur(x,(5,5),0.3, (1.5,1.5))           
            
            elif blurType =="box_blur3x3":
              x = K.filters.box_blur(x,(3,3))
            
            elif blurType =="box_blur5x5":
              x = K.filters.box_blur(x,(5,5))
              
            elif blurType =="gaussian_blur2d3x3":
              x = K.filters.gaussian_blur2d(x,(3,3),(1.5, 1.5))
            
            elif blurType =="gaussian_blur2d5x5":
              x = K.filters.gaussian_blur2d(x,(5,5),(1.5, 1.5))
            
            elif blurType =="median_blur3x3":
              x = K.filters.median_blur(x,(3,3))
            
            elif blurType =="median_blur5x5":
              x = K.filters.median_blur(x,(5,5))

            elif blurType =="median_blur7x7":
              x = K.filters.median_blur(x,(7,7))
              
            x /= 255.   

        x = F.pad(x,(0,0,3,4), "constant",0)              
        out1, out2, out3, out4, x = self.encoder(x.float())
        x= self.decoder(out1, out2, out3, out4, x)
        x = x[:,:,3:-4,:]

        return torch.sigmoid(x)



import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock_m(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch, track_running_stats =False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch, track_running_stats =False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)

def pad_to_match(x, ref):
    """Pad x to match the spatial size of ref"""
    diffY = ref.size(2) - x.size(2)
    diffX = ref.size(3) - x.size(3)
    x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                  diffY // 2, diffY - diffY // 2])
    return x

class MiniUNetPP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=32):
        super().__init__()
        
        # Encoder
        self.conv00 = ConvBlock_m(in_channels, base_ch)
        self.pool0 = nn.MaxPool2d(2)
        self.conv10 = ConvBlock_m(base_ch, base_ch * 2)
        self.pool1 = nn.MaxPool2d(2)
        
        # Bottleneck (can be deeper if needed)
        self.bottleneck = ConvBlock_m(base_ch * 2, base_ch * 4)

        # Decoder path
        self.up10 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv01 = ConvBlock_m(base_ch + base_ch * 2, base_ch)

        self.up01 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(base_ch * 2, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x00 = self.conv00(x)                 # X_00
        x10 = self.conv10(self.pool0(x00))   # X_10

        # Bottleneck
        x20 = self.bottleneck(self.pool1(x10))  # Not used directly

        # Decoder node X_01
        x10_up = self.up10(x10)
        x10_up = pad_to_match(x10_up, x00)
        x01 = self.conv01(torch.cat([x00, x10_up], dim=1))  # X_01

        # Final output
        x01_up = self.up01(x01)
        x01_up = pad_to_match(x01_up, x00)
        x_out = self.final_conv(torch.cat([x01_up, x00], dim=1))

        return torch.sigmoid(x_out)
