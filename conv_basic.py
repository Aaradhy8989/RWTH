
# Dependencies
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import torch 
import torch.nn.functional as F
import torch.optim as optim


# GPU settings
# Leave as it is even if you don't have GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing function
def process_train_data(img_dir):
    filenames = os.listdir(img_dir)
    images = []
    for name in tqdm(filenames):
        img = cv2.imread(img_dir + '/' + name)
        img = cv2.resize(img, (64, 128))
        img = cv2.GaussianBlur(img, (5, 5), 0)
        images.append(np.transpose(img, (2, 0, 1)))

    images = torch.FloatTensor(images)
    return images 


# Network 

class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        # Encoder block
        self.conv_1 = torch.nn.Conv2d(3, 8, kernel_size=(5, 5), stride=(2, 2))
        self.conv_2 = torch.nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
        self.conv_3 = torch.nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
        self.dropout_1 = torch.nn.Dropout2d(0.2)
        self.dropout_2 = torch.nn.Dropout2d(0.2)
        self.dropout_3 = torch.nn.Dropout2d(0.2)
        self.flatten = torch.nn.Flatten()
        self.fc_latent = torch.nn.Linear(24128, 500)
        
        # Decoder block
        self.fc_decode = torch.nn.Linear(500, 8*8*16)
        self.conv_trans_1 = torch.nn.ConvTranspose2d(8, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv_trans_2 = torch.nn.ConvTranspose2d(16, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv_trans_3 = torch.nn.ConvTranspose2d(32, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv_out = torch.nn.Conv2d(32, 3, kernel_size=(1, 1), stride=(1, 1))
        
    def forward(self, x):
        # In shape: (batch_size, 4, 128, 64)
        x = self.conv_1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout_1(x)
        x = self.conv_2(x)
        x = F.leaky_relu(x)
        x = self.dropout_2(x)
        x = self.conv_3(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout_3(x)
        x = self.flatten(x)
        x = self.fc_latent(x)
        x = torch.tanh(x)
        x = self.fc_decode(x)
        x = torch.reshape(x, (-1, 8, 16, 8))
        x = F.leaky_relu(x, 0.1)
        x = self.conv_trans_1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv_trans_2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv_trans_3(x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv_out(x)
        x = torch.sigmoid(x)
        
        return x


# Model training function
def train_model_on_batch(model, optimizer, x, y):

    optimizer.zero_grad()
    preds = model(x)
    loss = F.mse_loss(preds, y, reduction="mean")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
    optimizer.step()

    return loss.item()


# Generator object to load data
def data_loader(input_data, output_data, batch_size):

    in_samples, out_samples = [], []
    count = 0

    while True:
        for in_img, out_img in zip(input_data, output_data):
            in_samples.append(in_img.numpy())
            out_samples.append(out_img.numpy())
            count += 1

            if count == batch_size:
                yield (
                    torch.FloatTensor(in_samples).to(device), 
                    torch.FloatTensor(out_samples).to(device)
                )
                count = 0
                in_samples, out_samples = [], []


# Main script
if __name__ == "__main__":

    oh_dir = "../data/OH_field/0"
    temp_dir = "../data/Temp_Field/0"

    oh_images = process_train_data(oh_dir)
    temp_images = process_train_data(temp_dir)

    # Preserve same number of images for both
    size = min(oh_images.shape[0], temp_images.shape[0])
    oh_images = oh_images[:size]
    temp_images = temp_images[:size]

    # Training params
    epochs = 2000
    batch_size = 10
    steps_per_epoch = oh_images.shape[0] // batch_size

    model = Network().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Loss history
    loss_history = []

    # Training loop
    for epoch in range(epochs):
        print("\n\nEPOCH {}".format(epoch+1))
        print("------------------------------------------")

        # Initialize data loader and loss counter
        d_gen = data_loader(oh_images, temp_images, batch_size)
        total_loss = 0

        for step in range(steps_per_epoch):
            
            # Generate a batch of samples
            x, y = next(d_gen)

            # Train model on this batch
            loss = train_model_on_batch(model, optimizer, x, y)

            # Record loss
            loss_history.append(loss)
            total_loss += loss

            # Every 10 steps output status
            if step % 10 == 0:
                print("Step {} - Loss {:.4f}".format(
                    step, loss
                ))

        # Print epoch status at end of epoch
        print("Epoch {} - Average loss {:.4f}".format(
            epoch+1, total_loss/steps_per_epoch
        ))

        # Save model
        torch.save(model.state_dict(), "../saved_data/model_{}".format(epoch+1))

    # Save the loss history
    with open("../saved_data/loss_history.pkl", "wb") as f:
        pickle.dump(loss_history, f)