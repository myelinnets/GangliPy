from torch import nn
import torch
import numpy as np
from torch import autograd
from threading import Thread
import torch.optim as optim
import os
from ganglipy.sparse.sparse_variational_pooler import SparseVariationalPooler
from ganglipy.sparse.kwinners_boosted import KWinnersBoost


def dense_to_sparse(x):
    """ converts dense tensor x to sparse format """
    # from: https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809/3
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


stopper = False


class ConvolutionalEncoder(nn.Module):  # NOSONAR
    def __init__(self, displayable=False):
        super(ConvolutionalEncoder, self).__init__()
        self.fc = nn.Conv2d(1, 10000, kernel_size=(16, 16))
        self.booster = SparseVariationalPooler()
        self.conv = nn.ConvTranspose2d(10000, 1, kernel_size=(16, 16))
        self.displayable = displayable
        if self.displayable:
            self.display_list = [np.empty((16, 16)),
                                 np.empty((100, 100)),
                                 np.ones((100, 100)),
                                 np.empty((16, 16)), ]

    def _display_video_threads(self):
        if self.displayable:
            threads = []
            for d in self.display_list:
                v = VideoHandlerThread(video_source=d, callbacks=display_callbacks)
                threads.append(v)
            return threads

    def _display_windows(self):
        s = SubscriberWindows(video_sources=self.display_list,
                              window_names=[str(x) for x in range(len(self.display_list))])
        return s

    def display(self, training_function, *args, **kwargs):
        global stopper
        t = Thread(target=training_function, args=args, kwargs=kwargs)
        vs = self._display_video_threads()
        w = self._display_windows()
        for v in vs:
            v.start()
        t.start()
        w.loop()
        for v in vs:
            v.join(timeout=0.1)
        stopper = True
        t.join(timeout=0.1)

    def forward(self, x):
        if self.displayable: self.display_list[0][...] = x.view(16, 16).data.cpu().numpy()
        x = self.fc(x)

        x = self.booster(x)
        x_max = torch.max(self.booster.boost_tensor.data)
        x_min = torch.min(self.booster.boost_tensor.data)
        x_display = ((self.booster.boost_tensor.data - x_min) * (1.0 / (x_max - x_min))).view(100, 100).cpu().numpy()
        if self.displayable: self.display_list[1][...] = x_display * 1.0

        x_max = torch.max(x.data)
        x_min = torch.min(x.data)
        x_display = ((x.data - x_min) * (1.0 / (x_max - x_min))).view(100, 100).cpu().numpy()
        if self.displayable: self.display_list[2][...] = x_display * 1.0

        # x = dense_to_sparse(x) Todo: implement convTranspose on sparse for 5-10x speedup here.
        x = self.conv(x)
        if self.displayable: self.display_list[3][...] = x.view(16, 16).data.cpu().numpy()
        return x

    def forward_one_hot(self, one_hot, normalize=True):
        if isinstance(one_hot, int):
            one_hot = self.get_one_hot(one_hot)
        if not self.displayable:
            raise NotImplementedError("Must be marked displayable for forward_one_hot.")
        self.display_list[0][...] = np.ones((16, 16)) * .5
        self.display_list[1][...] = np.ones((100, 100)) * .5

        x_max = torch.max(one_hot.data)
        x_min = torch.min(one_hot.data)
        x_display = ((one_hot.data - x_min) * (1.0 / (x_max - x_min))).view(100, 100).cpu().numpy()
        self.display_list[2][...] = x_display * 1.0

        x = self.conv(one_hot)

        if normalize:
            x_max = torch.max(x.data)
            x_min = torch.min(x.data)
            x_display = ((x.data - x_min) * (1.0 / (x_max - x_min))).view(16, 16).cpu().numpy()
            self.display_list[3][...] = x_display * 1.0
        else:
            self.display_list[3][...] = x.view(16, 16).data.cpu().numpy()
        return x

    def get_one_hot(self, n):
        t = torch.zeros((1, 10000, 1, 1))
        t[0, n, 0, 0] = 1
        if next(self.parameters()).is_cuda:
            t = t.cuda()
        return t


loss_fn = nn.BCEWithLogitsLoss()
learning_rate = 1.0

from ganglipy.datasets import unifoundry as uni
from itertools import count
from cvpubsubs.webcam_pub import VideoHandlerThread, display_callbacks
from cvpubsubs.window_sub import SubscriberWindows


def train_from_unicode():
    it = uni.shuffled_iter()

    model = ConvolutionalEncoder(displayable=True)
    if os.path.isfile('ganglipy.torch'):
        model_dict = torch.load('ganglipy.torch')
        model_dict.pop('booster.boost_tensor')
        model.load_state_dict(model_dict)

    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), amsgrad=True)

    arr = np.empty((2, 16, 16, 1))

    def train_display(it):
        inp = next(it)
        for s in count(0):
            if s % 10 == 0:
                inp = next(it)
            arr[0, :] = inp.copy().astype(np.float)[..., np.newaxis]
            with torch.autograd.set_grad_enabled(True):
                inp_tens = torch.Tensor(inp.astype(np.float)[np.newaxis, np.newaxis, ...]).cuda()
                y_pred = model(inp_tens)
                loss = loss_fn(y_pred, inp_tens)
                # print(s, loss.item())
                model.zero_grad()
                loss.backward()
                optimizer.step()
            if stopper:
                break

    model.display(train_display, it)
    torch.save(model.state_dict(), "ganglipy.torch")


from itertools import cycle


def one_hot_display():
    model = ConvolutionalEncoder(displayable=True)
    if os.path.isfile('ganglipy.torch'):
        model_dict = torch.load('ganglipy.torch')
        model_dict.pop('booster.boost_tensor')
        model.load_state_dict(model_dict)

    model = model.cuda()
    it = cycle(range(10000))

    def test_display(it):
        for s in count(0):
            inp = next(it)
            with torch.autograd.set_grad_enabled(False):
                model.forward_one_hot(inp)
            if stopper:
                break

    model.display(test_display, it)


if __name__ == '__main__':
    train_from_unicode()
