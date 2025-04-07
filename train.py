import numpy as np
import math
import random
import os
import torch
from path import Path
from source import model
from source import dataset
from source import utils
from source.args import parse_args
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Set random seed for reproducibility
random.seed(42)

# Define the custom loss function for PointNet
def pointnet_loss(predictions, ground_truth, matrix_3x3, matrix_64x64, regularization_weight=0.0001):
    criterion = torch.nn.NLLLoss()
    batch_size = predictions.size(0)
    
    # Identity matrices for regularization
    identity_3x3 = torch.eye(3, requires_grad=True).repeat(batch_size, 1, 1)
    identity_64x64 = torch.eye(64, requires_grad=True).repeat(batch_size, 1, 1)
    
    if predictions.is_cuda:
        identity_3x3 = identity_3x3.cuda()
        identity_64x64 = identity_64x64.cuda()

    # Compute the difference between identity matrix and the transformed matrices
    diff_3x3 = identity_3x3 - torch.bmm(matrix_3x3, matrix_3x3.transpose(1, 2))
    diff_64x64 = identity_64x64 - torch.bmm(matrix_64x64, matrix_64x64.transpose(1, 2))

    # Calculate the loss (cross-entropy + regularization terms)
    loss = criterion(predictions, ground_truth) + regularization_weight * (torch.norm(diff_3x3) + torch.norm(diff_64x64)) / float(batch_size)
    return loss


# Training function
def train_model(args):
    data_path = Path(args.root_dir)
    
    # Prepare class labels
    class_folders = [folder for folder in sorted(os.listdir(data_path)) if os.path.isdir(data_path / folder)]
    class_labels = {folder: idx for idx, folder in enumerate(class_folders)}
    
    # Define data transformations
    data_transforms = transforms.Compose([
        utils.PointSampler(1024),
        utils.Normalize(),
        utils.RandRotation_z(),
        utils.RandomNoise(),
        utils.ToTensor()
    ])
    
    # Set device (GPU or CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize PointNet model
    model_net = model.PointNet().to(device)
    
    # Set optimizer
    optimizer = torch.optim.Adam(model_net.parameters(), lr=args.lr)
    
    # Initialize datasets
    train_dataset = dataset.PointCloudData(data_path, transform=data_transforms)
    validation_dataset = dataset.PointCloudData(data_path, valid=True, folder='test', transform=data_transforms)
    
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(validation_dataset)}')
    print(f'Number of classes: {len(train_dataset.classes)}')
    
    # Set up data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=args.batch_size * 2)
    
    # Create directory to save the model
    os.makedirs(args.save_model_path, exist_ok=True)
    
    print('Starting training...')
    
    # Train the model
    for epoch in range(args.epochs):
        model_net.train()  # Set model to training mode
        epoch_loss = 0.0
        
        for batch_idx, data in enumerate(train_loader, 0):
            pointcloud, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, matrix_3x3, matrix_64x64 = model_net(pointcloud.transpose(1, 2))
            
            # Calculate loss
            loss = pointnet_loss(outputs, labels, matrix_3x3, matrix_64x64)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Print statistics
            epoch_loss += loss.item()
            if batch_idx % 10 == 9:  # Print every 10 mini-batches
                print(f'[Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(train_loader)}], Loss: {epoch_loss / 10:.3f}')
                epoch_loss = 0.0
        
        # Validation phase
        model_net.eval()  # Set model to evaluation mode
        correct_predictions = total_samples = 0
        
        if validation_loader:
            with torch.no_grad():
                for data in validation_loader:
                    pointcloud, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, _, _ = model_net(pointcloud.transpose(1, 2))
                    _, predicted_labels = torch.max(outputs.data, 1)
                    total_samples += labels.size(0)
                    correct_predictions += (predicted_labels == labels).sum().item()

            validation_accuracy = 100. * correct_predictions / total_samples
            print(f'Validation Accuracy: {validation_accuracy:.2f}%')
        
        # Save model checkpoint
        model_checkpoint = Path(args.save_model_path) / f'save_epoch_{epoch}.pth'
        torch.save(model_net.state_dict(), model_checkpoint)
        print(f'Model saved to {model_checkpoint}')


# Entry point for the script
if __name__ == '__main__':
    # Parse command-line arguments
    arguments = parse_args()
    
    # Start training
    train_model(arguments)
