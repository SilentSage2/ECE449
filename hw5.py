import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


def get_data_loader(batch_size):
    """Build a data loader from training images of MNIST."""

    # Some pre-processing
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    # This will download MNIST to a 'data' folder where you run this code
    train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # Build the data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader


class Discriminator(nn.Module):
    """Discriminator network."""

    def __init__(self):
        """
        Define the layers in the discriminator network.
        """
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = nn.Conv2d   (1, 16, kernel_size=3,stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv2 = nn.Conv2d   (16, 32, kernel_size=3,stride=1, padding=1)
        self.conv3 = nn.Conv2d   (32, 64, kernel_size=3,stride=1, padding=1)
        self.conv4 = nn.Conv2d   (64, 128,kernel_size=3,stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=4,stride=4,padding=0)
        self.fc    = nn.Linear   (128, 1)
        self.relu  = nn.LeakyReLU(negative_slope=0.2)
        self.sig   = nn.Sigmoid  ()

    def forward(self, x):
        """
        Define forward pass in the discriminator network.

        Arguments:
            x: A tensor with shape (batch_size, 1, 32, 32).

        Returns:
            A tensor with shape (batch_size), predicting the probability of
            each example being a *real* image. Values in range (0, 1).
        """
        # Your code here
        N = x.shape[0]
        # first layer
        x = x.view(N, 1, 32, 32)
        x = self.pool1(self.relu(self.conv1(x)))
        # second layer
        x = self.pool1(self.relu(self.conv2(x)))
        # third layer
        x = self.pool1(self.relu(self.conv3(x)))
        # fourth layer
        x = self.pool2(self.relu(self.conv4(x)))
        # fully connected layer
        x = self.fc(torch.squeeze(x))
        x = self.sig(x)
        return x.view(N)


class Generator(nn.Module):
    """Generator network."""

    def __init__(self,):
        """
        Define the layers in the generator network.
        """
        super().__init__()
        # Your code here
        torch.manual_seed(0)
        self.convt1 = nn.ConvTranspose2d   (128, 64, kernel_size=4,stride=1, padding=0)
        self.convt2 = nn.ConvTranspose2d   (64 , 32, kernel_size=4,stride=2, padding=1)
        self.convt3 = nn.ConvTranspose2d   (32 , 16, kernel_size=4,stride=2, padding=1)
        self.convt4 = nn.ConvTranspose2d   (16 , 1 , kernel_size=4,stride=2, padding=1)
        self.relu   = nn.LeakyReLU(negative_slope=0.2)
        self.tanh   = nn.Tanh     ()

    def forward(self, z):
        """
        Define forward pass in the generator network.

        Arguments:
            z: A tensor with shape (batch_size, 128).

        Returns:
            A tensor with shape (batch_size, 1, 32, 32). Values in range (-1, 1).
        """
        # Your code here
        N = z.shape[0]
        # first layer
        z = z.view(N, 128, 1, 1)
        z = self.relu(self.convt1(z))
        # second layer
        z = self.relu(self.convt2(z))
        # third layer
        z = self.relu(self.convt3(z))
        # fourth layer
        z = self.relu(self.convt4(z))
        # fully connected layer
        z = self.tanh(z)
        return z.view(N,1,32,32)


class GAN(object):
    """Generative Adversarial Network."""

    def __init__(self):
        # This will use GPU if available - don't worry if you only have CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = get_data_loader(batch_size=64)
        self.D = Discriminator().to(self.device)
        self.G = Generator().to(self.device)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002)
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.001)

    def calculate_V(self, x, z):
        """
        Calculate the optimization objective for discriminator and generator
        as specified.

        Arguments:
            x: A tensor representing the real images,
                with shape (batch_size, 1, 32, 32).
            z: A tensor representing the latent variable (randomly sampled
                from Gaussian), with shape (batch_size, 128).

        Return:
            A tensor with only one element, representing the objective V.
        """
        # Your code here
        N = x.shape[0]
        D_x = self.D.forward(x)
        D_G_z = self.D.forward(self.G.forward(z))
        V = (torch.sum(torch.log(D_x))+torch.sum(torch.log(1-D_G_z)))/N
        return V

    def train(self, epochs):
        for epoch in range(epochs):
            print('Epoch {}'.format(epoch))

            # Data loader will also return the labels of the digits (_), but we will not use them
            for iteration, (x, _) in enumerate(self.data_loader):
                x = x.to(self.device)
                z = torch.randn(64, 128).to(self.device)

                # Train the discriminator
                # We want to maximize V <=> minimize -V
                self.D_optimizer.zero_grad()
                D_target = -self.calculate_V(x, z)
                D_target.backward()
                self.D_optimizer.step()

                # Train the generator
                # We want to minimize V
                self.G_optimizer.zero_grad()
                G_target = self.calculate_V(x, z)
                G_target.backward()
                self.G_optimizer.step()

                if iteration % 100 == 0:
                    print('Iteration {}, V={}'.format(iteration, G_target.item()))

            self.visualize('Epoch {}.png'.format(epoch))

    def visualize(self, save_path):
        # Let's generate some images and visualize them
        z = torch.randn(64, 128).to(self.device)
        fake_images = self.G(z)
        save_image(fake_images, save_path)


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=10)
