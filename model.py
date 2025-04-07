import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# TNet module for computing transformations
class TNet(nn.Module):
    def __init__(self, input_dim=3):
        super(TNet, self).__init__()
        self.input_dim = input_dim  # Number of input dimensions (e.g., 3 for point cloud)
        
        # Define 1D convolution layers
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_dim * input_dim)  # Output is a transformation matrix (input_dim x input_dim)

        # Define batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    
    def forward(self, input_points):
        # Input shape: (batch_size, num_points, input_dim)
        batch_size = input_points.size(0)
        
        # Apply conv layers with ReLU activations
        x = F.relu(self.bn1(self.conv1(input_points)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Apply max pooling to reduce dimensionality
        pooled_features = nn.MaxPool1d(x.size(-1))(x)
        
        # Flatten the pooled features
        flat_features = nn.Flatten(1)(pooled_features)
        
        # Apply fully connected layers with ReLU activations
        x = F.relu(self.bn4(self.fc1(flat_features)))
        x = F.relu(self.bn5(self.fc2(x)))
        
        # Initialize the transformation matrix as identity
        identity_matrix = torch.eye(self.input_dim, requires_grad=True).repeat(batch_size, 1, 1)
        if x.is_cuda:
            identity_matrix = identity_matrix.cuda()

        # Output the transformation matrix
        transformation_matrix = self.fc3(x).view(-1, self.input_dim, self.input_dim) + identity_matrix
        return transformation_matrix


# Transform module to handle both input and feature transformations
class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()
        self.input_transformer = TNet(input_dim=3)  # Transform for input points
        self.feature_transformer = TNet(input_dim=64)  # Transform for features
        
        # Define convolution layers for feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Define batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
    
    def forward(self, input_points):
        # Apply input transformation
        transformation_matrix_3x3 = self.input_transformer(input_points)
        
        # Apply the transformation to the input points
        transformed_points = torch.bmm(torch.transpose(input_points, 1, 2), transformation_matrix_3x3).transpose(1, 2)
        
        # Apply convolution layers for feature extraction
        x = F.relu(self.bn1(self.conv1(transformed_points)))
        
        # Apply feature transformation
        transformation_matrix_64x64 = self.feature_transformer(x)
        transformed_features = torch.bmm(torch.transpose(x, 1, 2), transformation_matrix_64x64).transpose(1, 2)
        
        # Apply remaining convolution layers
        x = F.relu(self.bn2(self.conv2(transformed_features)))
        x = self.bn3(self.conv3(x))
        
        # Apply max pooling and flatten the output
        pooled_features = nn.MaxPool1d(x.size(-1))(x)
        output = nn.Flatten(1)(pooled_features)
        
        return output, transformation_matrix_3x3, transformation_matrix_64x64


# PointNet model for point cloud classification
class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNet, self).__init__()
        self.transform = Transform()
        
        # Define fully connected layers for classification
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Define batch normalization layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)
        
        # LogSoftmax for output probabilities
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_points):
        # Apply transformation and feature extraction
        features, transformation_matrix_3x3, transformation_matrix_64x64 = self.transform(input_points)
        
        # Apply fully connected layers with ReLU activations and batch normalization
        x = F.relu(self.bn1(self.fc1(features)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        
        # Final classification layer
        logits = self.fc3(x)
        
        # Apply LogSoftmax for classification output
        output = self.logsoftmax(logits)
        
        return output, transformation_matrix_3x3, transformation_matrix_64x64
