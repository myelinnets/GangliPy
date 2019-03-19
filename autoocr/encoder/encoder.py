from torch import nn
import torch
import numpy as np

class ConvolutionalEncoder(nn.Module):  # NOSONAR
    def __init__(self):
        super(ConvolutionalEncoder, self).__init__()
        self.fc = nn.Conv2d(1, 1000, kernel_size=(16, 16))
        self.fc_act = nn.LeakyReLU()
        self.conv = nn.ConvTranspose2d(1000, 1, kernel_size=(16, 16))
        self.conv_norm = nn.BatchNorm2d(1)
        self.conv_act = nn.ReLU(True)

    def forward(self, x):
        x = self.fc(x)
        x = self.fc_act(x)
        x = self.conv(x)
        x = self.conv_norm(x)
        x = self.conv_act(x)
        return x


loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-4

from autoocr.datasets import unifoundry as uni
from itertools import count
from cvpubsubs.webcam_pub import VideoHandlerThread, display_callbacks
from cvpubsubs.window_sub import SubscriberWindows
def train_from_unicode():
    it = uni.shuffled_iter()
    model = ConvolutionalEncoder()
    device = torch.device('cpu')

    x = torch.randn((16, 16), device=device)
    y = torch.randn((16, 16), device=device)
    arr = np.empty((2,16,16, 1))
    def train_display(frame, id):
        for s in count(0):
            inp = next(it)
            arr[1, :] = inp.copy().astype(np.float)[..., np.newaxis]
            y_pred = model(torch.Tensor(inp.astype(np.float)[np.newaxis, np.newaxis, ...]))
            loss = loss_fn(y_pred, y)
            #print(s, loss.item())
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param.data -= learning_rate * param.grad
            arr[0,:] = y_pred.detach().permute((0,2,3,1)).cpu().numpy()[0,...]
            return arr

    v = VideoHandlerThread(video_source=arr, callbacks=display_callbacks + [train_display])
    s = SubscriberWindows(video_sources=[arr], window_names=[str(x) for x in range(2)])

    v.start()
    s.loop()
    v.join()



if __name__ == '__main__':
    train_from_unicode()
