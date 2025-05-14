import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim = 100, img_chaanels = 3, fmap = 64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.ReLU(),

            nn.Linear(1024, 784),
            nn.Tanh()
        ).to(device)

        
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)
    
    def forward(self, img):
        img = img.view(img.size(0), -1)
        validity = self.model(img)
        return validity

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, z):
        img = self.generator(z)
        validity = self.discriminator(img)
        return img, validity
    
    def train(self, dataloader, epochs, optimizer_G, optimizer_D, criterion):
        for epoch in range(epochs):


if __name__ == "__main__":
    if cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    # data load (MNIST)


    # Hyperparameters


    # Loss function


    # Optimizers


    # Initialize
    generator = Generator()
    discriminator = Discriminator()

    ## training
    for epoch in range(epochs):
        for i, (ing, _) in enumerate(dataloader):

            # ground truth

            # train G

            # train D

            pass

    ## inference
    # generate img