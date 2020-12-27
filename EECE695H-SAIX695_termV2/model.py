import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
#  from protonets.models import register_model
from src.utils import square_euclidean_metric

def conv_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


def conv_block_EJK(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.MaxPool2d(2)
    )

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0),-1)


""" Define your own model """
class FewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim1=64, hid_dim2=128, hid_dim3=256, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
                conv_block(x_dim, hid_dim1),
                conv_block(hid_dim1, hid_dim1),
                conv_block(hid_dim1, hid_dim2),
                conv_block(hid_dim2, hid_dim3),
                conv_block(hid_dim3, hid_dim2),
                conv_block(hid_dim2, hid_dim1),
                conv_block(hid_dim1, hid_dim1),
                conv_block(hid_dim1, z_dim),
                Flatten()
                )
        self._initialize_weights3()
    def forward(self, x):        
        embedding_vector = self.encoder(x)

        return embedding_vector
    
    def _initialize_weights3(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0, 0.01)
                nn.init.constant_(m.bias, 0)


""" Define your own model """
class FewShotModel2(nn.Module):
    def __init__(self, x_dim=3, hid_dim1=64, hid_dim2=128, hid_dim3=256, hid_dim4=512, hid_dim5=1024, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
                conv_block(x_dim, hid_dim1),
                conv_block(hid_dim1, hid_dim1),
                conv_block(hid_dim1, hid_dim1),
                conv_block(hid_dim1, hid_dim1),
                conv_block(hid_dim1, z_dim),
                Flatten()
                )
        self._initialize_weights3()

    def forward(self, x):        
        embedding_vector = self.encoder(x)
        #  print(embedding_vector.shape)
        return embedding_vector
    
    def _initialize_weights3(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0, 0.01)
                nn.init.constant_(m.bias, 0)


class FewShotModel3(nn.Module):
    def __init__(self, x_dim=3, hid_dim1=64, hid_dim2=128, hid_dim3=256, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
                conv_block_EJK(x_dim, hid_dim1),
                conv_block_EJK(hid_dim1, hid_dim1),
                conv_block_EJK(hid_dim1, hid_dim2),
                conv_block_EJK(hid_dim2, hid_dim3),
                conv_block_EJK(hid_dim3, hid_dim2),
                conv_block_EJK(hid_dim2, hid_dim1),
                conv_block_EJK(hid_dim1, hid_dim1),
                conv_block_EJK(hid_dim1, z_dim),
                Flatten()
                )
    def forward(self, x):        
        embedding_vector = self.encoder(x)

        return embedding_vector
    
    def _initialize_weights3(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0, 0.01)
                nn.init.constant_(m.bias, 0)


class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64))
        
        self.layer2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64))
        self.Maxpool2d = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc1 = nn.Sequential(
                nn.Linear(2*2*64, 10),
                nn.BatchNorm1d(10))
        
        self._initialize_weights3()

    def forward(self, x):
        out = self.layer1(x)
        out = self.Maxpool2d(out)
        out = self.layer2(out)
        out = self.Maxpool2d(out)
        out = out.view(out.size(0), -1)
        #  out = self.fc1(out)
        return out

    def _initialize_weights3(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0, 0.01)
                nn.init.constant_(m.bias, 0)

class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm2d(64)
        self.Maxpool2d = nn.MaxPool2d(kernel_size=4, stride=4)
        self.Adaptivepool = nn.AdaptiveAvgPool2d(3)
        self.fc1 = nn.Linear(3*3*64, 10)
        self._initialize_weights()
    def forward(self, x):
        out = self.batch(self.relu(self.conv1(x)))
        out = self.Maxpool2d(out)
        out = self.batch(self.relu(self.conv2(out)))
        out = self.Adaptivepool(out)
        #  out = out.view(-1, 3*3*64)
        #  out = self.fc1(out)
        out = out.view(out.size(0), -1)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    

class Ensemble_Fewshot(nn.Module):
    def __init__(self, modelA, modelB, output_size):
        super(Ensemble_Fewshot, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(9280, 64)
    def forward(self, x):
        x1 = self.modelA(x.clone())
        x1 = x1.view(x1.size(0), -1)
        #  print(x1.shape)

        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        #  print(x2.shape)

        x = torch.cat((x1, x2), dim=1)
        #  print(x.shape)
        x = self.classifier(self.relu(x))
        out = x.view(x.size(0), -1)
    
        return out

class Ensemble_Fewshot2(nn.Module):
    def __init__(self, modelA, modelB, modelC, output_size):
        super(Ensemble_Fewshot2, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.relu = nn.ReLU()
        #  self.classifier = nn.Linear(593536, 64)

    def forward(self, x):
        x1 = self.modelA(x.clone())
        x1 = x1.view(x1.size(0), -1)
        #  print(x1.shape)

        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        #  print(x2.shape)

        x3 = self.modelB(x)
        x3 = x3.view(x3.size(0), -1)
        #  print(x3.shape)
        
        x = torch.cat((x1, x2, x3), dim=1)
        #  print(x.shape)
        #  x = self.classifier(self.relu(x))
        out = x.view(x.size(0), -1)
    
        return out




class IdentityPadding(nn.Module):
	def __init__(self, in_channels, out_channels, stride):
		super(IdentityPadding, self).__init__()
		
		self.pooling = nn.MaxPool2d(1, stride=stride)
		self.add_channels = out_channels - in_channels
    
	def forward(self, x):
		out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
		out = self.pooling(out)
		return out
	
	
class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
			                   stride=stride, padding=1, bias=False) 
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
			                   stride=1, padding=1, bias=False) 
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.stride = stride
		
		if down_sample:
			self.down_sample = IdentityPadding(in_channels, out_channels, stride)
		else:
			self.down_sample = None


	def forward(self, x):
		shortcut = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.down_sample is not None:
			shortcut = self.down_sample(x)

		out += shortcut
		out = self.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, num_layers, block, num_classes=10):
		super(ResNet, self).__init__()
		self.num_layers = num_layers
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 
							   stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.relu = nn.ReLU(inplace=True)
		
		# feature map size = 32x32x16
		self.layers_2n = self.get_layers(block, 16, 16, stride=1)
		# feature map size = 16x16x32
		self.layers_4n = self.get_layers(block, 16, 32, stride=2)
		# feature map size = 8x8x64
		self.layers_6n = self.get_layers(block, 32, 64, stride=2)

		# output layers
		self.avg_pool = nn.AvgPool2d(8, stride=1)
		self.fc_out = nn.Linear(64, num_classes)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', 
					                    nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
	
	def get_layers(self, block, in_channels, out_channels, stride):
		if stride == 2:
			down_sample = True
		else:
			down_sample = False
		
		layers_list = nn.ModuleList(
			[block(in_channels, out_channels, stride, down_sample)])
			
		for _ in range(self.num_layers - 1):
			layers_list.append(block(out_channels, out_channels))

		return nn.Sequential(*layers_list)
		
	def forward(self, x):
                #  print(x.shape)
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                #  print(x.shape)
                
                x = self.layers_2n(x)
                #  print(x.shape)
                x = self.layers_4n(x)
                #  print(x.shape)
                x = self.layers_6n(x)
                #  print(x.shape)
                
                x = self.avg_pool(x)
                #  print(x.shape)
                x = x.view(x.size(0), -1)
                #  print(x.shape)
                #  x = self.fc_out(x)
                #  print(x.shape)
                #  x = x.view(x.size(0), -1)
                return x


def resnet():
	block = ResidualBlock
	# total number of layers if 6n + 2. if n is 5 then the depth of network is 32.
	model = ResNet(5, block) 
	return model

def Fewshot_EJK():
    modelA = FewShotModel()
    modelB = FewShotModel2()
    model = Ensemble_Fewshot(modelA, modelB, 64)
    return model

