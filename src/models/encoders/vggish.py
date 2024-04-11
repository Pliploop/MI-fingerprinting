import torch.nn as nn
import torch
import torchaudio

class Melgram(nn.Module):
    
    def __init__(self, n_mels = 96, n_fft = 2048, window_len = 400, hop_length = 160, sample_rate = 16000, f_min = 300, f_max = 7000, power = 2):
        super(Melgram, self).__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.window_len = window_len
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length = window_len,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
        
        self.power = power
        
        stype = 'power' if self.power == 2 else 'magnitude'
        self.compressor = torchaudio.transforms.AmplitudeToDB(stype)
        
    def  forward(self, x):
        x = self.mel(x)
        x = self.compressor(x)
        return x

class VGGish(nn.Module):
    
    """A VGG-ish model that takes 1s of audio at 16000 Hz as input and outputs a 512-dimensional embedding.
    the frontend expects 1s (22050 samples) of mono audio at 16000 Hz. and returns a (100,96) melgram"""
    
    def __init__(self,channel_scale = 1):
        super(VGGish, self).__init__()

        self.embed_dim = 512
        self.sr = 16000
        self.n_samples = 1*self.sr

        self.frontend = Melgram()
            
        channels = int(16*channel_scale)
        all_channels = [1,channels,channels*2,channels*4,channels*8,channels*16]
        
        
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1 = nn.Conv2d(all_channels[0], all_channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        
        self.conv2 = nn.Conv2d(all_channels[1], all_channels[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        
        self.conv3_1 = nn.Conv2d(all_channels[2], all_channels[3], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(all_channels[3], all_channels[3], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        
        self.conv4_1 = nn.Conv2d(all_channels[3], all_channels[4], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(all_channels[4], all_channels[4], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        
        self.conv5_1 = nn.Conv2d(all_channels[4], all_channels[5], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(all_channels[5], all_channels[5], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        
        self.fc6 = nn.Linear(all_channels[5] * 3 * 3, 1024)
        self.relu6 = nn.ReLU()
        
        self.fc7 = nn.Linear(1024, 1024)
        self.relu7 = nn.ReLU()
        
        self.fc8 = nn.Linear(1024, 512)
        
    def forward(self, x):
        if self.frontend:
            x = self.frontend(x)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.pool3(x)
        
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.pool4(x)
        
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)
        
        x = self.fc8(x)
        
        return x