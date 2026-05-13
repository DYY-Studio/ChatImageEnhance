import torch
import torch.nn as nn
import torch.nn.functional as F

'''
# --------------------------------------------
# Zero-DCE++
# --------------------------------------------
Reference:
@inproceedings{Zero-DCE++,
 author = {Li, Chongyi and Guo, Chunle Guo and Loy, Chen Change},
 title = {Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation},
 booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
 pages    = {},
 month = {},
 year = {2021}
 doi={10.1109/TPAMI.2021.3063604}
}
'''

class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class ZeroDCE_Extension_NET(nn.Module):

	def __init__(self):
		super(ZeroDCE_Extension_NET, self).__init__()

		self.relu = nn.ReLU(inplace=True)
		number_f = 32

#   zerodce DWC + p-shared
		self.e_conv1 = CSDN_Tem(3,number_f) 
		self.e_conv2 = CSDN_Tem(number_f,number_f) 
		self.e_conv3 = CSDN_Tem(number_f,number_f) 
		self.e_conv4 = CSDN_Tem(number_f,number_f) 
		self.e_conv5 = CSDN_Tem(number_f*2,number_f) 
		self.e_conv6 = CSDN_Tem(number_f*2,number_f) 
		self.e_conv7 = CSDN_Tem(number_f*2,3) 
		
	def forward(self, x):
		x_down = x

		x1 = self.relu(self.e_conv1(x_down))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(x2))
		x4 = self.relu(self.e_conv4(x3))
		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
		x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
		
		return x_r