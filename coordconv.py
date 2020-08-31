import torch
import torch.nn as nn

class AddLayers(nn.Module):
    def __init__(self, useCuda):
        super(AddLayers, self).__init__()
        self.useCuda = useCuda

    def forward(self, image):
        batch_size, _, x_dim, y_dim = image.shape

        ## Creating x channel
        x_ones = torch.ones(batch_size, x_dim, 1)
        x_range = torch.arange(y_dim)
        x_channel = (x_ones*x_range) / (x_dim - 1)
        x_channel = x_channel * 2 - 1
        x_channel.resize_(batch_size, 1, x_dim, y_dim)

        ## Creating y channel
        y_ones = torch.ones(batch_size, y_dim, 1)
        y_range = torch.arange(x_dim)
        y_channel = (y_ones*y_range).T / (y_dim - 1)
        y_channel = y_channel * 2 - 1
        y_channel = y_channel.permute(2, 0, 1)
        y_channel.unsqueeze_(1)

        ## Conbining channels with original image
        if self.useCuda:
            x_channel = x_channel.cuda()
            y_channel = y_channel.cuda()
        out_image = torch.cat([image, x_channel, y_channel], dim=1)
        return out_image

if __name__ == '__main__':
    coordConv = AddLayers()
    x = torch.ones(1, 1, 50, 50)
    x = coordConv(x)
    print(x)
    print(x.shape)