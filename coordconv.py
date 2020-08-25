import torch
import torch.nn as nn

class AddLayers(nn.Module):
    def __init__(self, x_dim=50, y_dim=50):
        super(AddLayers, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

    def forward(self, image):
        batch_size = image.shape[0]

        x_ones = torch.ones(batch_size, self.x_dim, 1)
        x_range = torch.arange(self.y_dim)
        x_channel = (x_ones*x_range).unsqueeze(-1)

        y_ones = torch.ones(batch_size, self.y_dim, 1)
        y_range = torch.arange(self.x_dim)
        y_channel = (y_ones*y_range).unsqueeze(-1)

        out_image = torch.cat([image, x_channel, y_channel], dim=-1)
        return out_image

if __name__ == '__main__':
    coordConv = AddLayers()
    x = torch.randn(1, 50, 50, 1)
    x = coordConv(x)
    print(x.shape)
    print(x.view(50, 50, 3))

