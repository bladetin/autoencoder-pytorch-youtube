import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time 

device_gpu = torch.device("cuda:2")
# print(torch.cuda.is_available())

transform = transforms.ToTensor()
mnist_data = datasets.MNIST(
    root=".\data", train=True, download=True, transform=transform
)
data_loader = torch.utils.data.DataLoader(
    dataset=mnist_data, batch_size=128, shuffle=True, num_workers=4,pin_memory=True
)

# train_data -> data
# train_labels -> targets
# mnist_data.data = mnist_data.data.to(device)
# mnist_data.targets = mnist_data.targets.to(device)

# dataiter = iter(data_loader)
# images, labels = dataiter.next()
# # 了解取值范围，方便后续使用正确的激活函数，
# print(torch.min(images), torch.max(images))


class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # N(batch_size), 784
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),  # N,784->128
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),  # N,3
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  # N,3->N,784
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = Autoencoder()
model = model.to(device_gpu)
criterion = nn.MSELoss().to(device_gpu)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

num_epochs = 5
outputs = []


if __name__ == '__main__':
    for epoch in range(num_epochs):
        # for img in data_loader.dataset.data.cpu():
        since = time.time()
        for i, (img, labels) in enumerate(data_loader):
            # img = img.reshape(-1, 28 * 28).to(device)
            img = torch.reshape(img, (-1, 28 * 28)).to(device_gpu)
            recon = model(img)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss:{loss.item():.4f}")
        outputs.append((epoch, img, recon))
        print(f"Epoch {epoch+1}:{time.time()-since:.4f}")

    for k in range(0, num_epochs, 4):
        plt.figure(figsize=(9, 2))
        plt.gray()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9:
                break
            plt.subplot(2, 9, i + 1)
            item = item.reshape(-1, 28, 28)
            # plt.imshow(item[0])

        for i, item in enumerate(recon):
            if i >= 9:
                break
            plt.subplot(2, 9, 9 + i + 1)
            item = item.reshape(-1, 28, 28)
            # plt.imshow(item[0])

    myfig = plt.gcf()
    myfig.savefig(".\figure\figure.png", dpi=300)

