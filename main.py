# main.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import collate_fn
from arcash_net import ArcAsh3DClassifier
from dataset import STLCategoryDataset
from data_utils import one_hot_encode
from sklearn.model_selection import train_test_split

def load_categories(root_dir):
    return [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

def validation_loss(model, num_classes, criterion, dataloader, device):
    total_loss = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            labels = one_hot_encode(labels, num_classes)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    model.train()
    return total_loss / total_samples

def main():
    root_dir = "/home/laptop/Projekty/wizard3d/Thingi10K_name_and_category"
    categories = load_categories(root_dir)
    voxel_res = 32
    batch_size = 4
    num_classes = len(categories)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    dataset = STLCategoryDataset(root_dir, categories, voxel_res)

    # Split the dataset into training and test sets
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42, shuffle=True)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    # Create DataLoaders for training and test sets
    train_dataloader = DataLoader(dataset, batch_size, sampler=train_sampler, collate_fn=lambda b: collate_fn(b, batch_size))
    test_dataloader = DataLoader(dataset, batch_size, sampler=test_sampler, collate_fn=lambda b: collate_fn(b, batch_size))

    # Initialize the model
    model = ArcAsh3DClassifier(batch_size=batch_size, in_channels=1, num_classes=num_classes, device=device).to(device)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(100):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader, 0):
            labels = one_hot_encode(labels, num_classes)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).to(device)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)
        test_loss = validation_loss(model, num_classes, criterion, test_dataloader, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}")

    print("Finished training")

    # Save the model
    torch.save(model.state_dict(), "arcash_3d_classifier.pth")

if __name__ == "__main__":
    main()

