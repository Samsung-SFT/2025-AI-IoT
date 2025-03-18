class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate, batch_first=True):
        super(LSTM, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.l1=nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=False)
        self.l2=nn.Linear(hidden_size, output_size)
        self.dropout=nn.Dropout(dropout_rate)
        self.batch_first=batch_first
    def forward(self, x):
        if x.dim()==2:
            x=x.unsqueeze(1)
        batch_size=x.size(1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        #out, _ = self.l1(x)
        out, _=self.l1(x, (h0, c0))
        #out=torch.tensor(out)
        #out = self.relu(out)
        #out=self.dropout(out)
        #out=self.l2(out)
        out = self.l2(out[:, -1, :])
        out=torch.relu(out)
        #out=F.softmax(out, dim=-1)
        return out


# pip install torch torchvision matplotlib opencv-python scikit-learn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)

DATASET_PATH = '\My Documents\FLIR'


class ThermalImageDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.images = []

        for filename in os.listdir(dataset_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(dataset_path, filename)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0
                self.images.append(img)

        self.images = np.array(self.images).astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return torch.tensor(image).unsqueeze(0)


def load_data(dataset_path, test_size=0.2):
    dataset = ThermalImageDataset(dataset_path)
    train_data, test_data = train_test_split(dataset.images, test_size=test_size)

    train_dataset = [torch.tensor(x).unsqueeze(0) for x in train_data]
    test_dataset = [torch.tensor(x).unsqueeze(0) for x in test_data]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    train_loss = []
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs,) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return train_loss


def test_model(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, in test_loader:
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, inputs)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss


if __name__ == "__main__":
    train_loader, test_loader = load_data(DATASET_PATH)

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss = train_model(model, train_loader, criterion, optimizer, num_epochs=20)

    plt.plot(train_loss)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show("Urban Heat Island Mapping Technique")

    test_model(model, test_loader)

    torch.save(model.state_dict(), 'thermal_image_autoencoder_model.pth')

    new_img_path = 'path_to_new_thermal_image.jpg'
    new_img = cv2.imread(new_img_path, cv2.IMREAD_GRAYSCALE)
    new_img_resized = cv2.resize(new_img, IMG_SIZE)
    new_img_resized = new_img_resized / 255.0  # Normalize

    new_img_tensor = torch.tensor(new_img_resized).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dims
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        reconstructed_image = model(new_img_tensor)

    reconstructed_image = reconstructed_image.squeeze().numpy()  # Remove batch and channel dims
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title("Reconstructed Image")
    plt.show()

