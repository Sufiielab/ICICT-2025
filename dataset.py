import os
from path import Path
from source import utils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Default transformation pipeline for point cloud data
def get_default_transforms():
    return transforms.Compose([
        utils.PointSampler(1024),  # Sampling 1024 points from the mesh
        utils.Normalize(),          # Normalizing the point cloud
        utils.ToTensor()            # Converting to tensor
    ])

# Custom Dataset class for loading point cloud data
class PointCloudDataset(Dataset):
    def __init__(self, root_dir, is_valid=False, data_split="train", transform=get_default_transforms()):
        """
        Args:
            root_dir (str): Directory containing the dataset.
            is_valid (bool): Flag indicating if it's for validation (default is False).
            data_split (str): Either "train" or "test" indicating the dataset split.
            transform (callable): Optional transformation to be applied on a sample.
        """
        self.root_dir = root_dir
        # Get list of class folders in the root directory
        class_folders = [folder for folder in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/folder)]
        self.classes = {folder: idx for idx, folder in enumerate(class_folders)}  # Mapping of class names to ids
        
        # Set transformations
        self.transform = transform if not is_valid else get_default_transforms()
        self.is_valid = is_valid
        
        # List of files in the dataset
        self.files = []
        for category in self.classes.keys():
            class_dir = root_dir / Path(category) / data_split
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.off'):  # Only include .off files
                    sample = {
                        'pointcloud_path': class_dir / file_name,  # Path to the .off file
                        'category': category  # Class label
                    }
                    self.files.append(sample)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.files)

    def _preprocess(self, file):
        """
        Preprocess the file to extract the point cloud data.
        
        Args:
            file: File object representing the .off file.
            
        Returns:
            Processed point cloud data.
        """
        vertices, faces = utils.read_off(file)  # Read vertices and faces from .off file
        if self.transform:
            pointcloud = self.transform((vertices, faces))  # Apply transformations
        return pointcloud

    def __getitem__(self, idx):
        """
        Fetch a single item from the dataset by index.
        
        Args:
            idx (int): Index of the sample to fetch.
            
        Returns:
            dict: Contains the point cloud and corresponding class label.
        """
        pointcloud_path = self.files[idx]['pointcloud_path']
        category = self.files[idx]['category']
        
        # Open the file and preprocess
        with open(pointcloud_path, 'r') as f:
            pointcloud = self._preprocess(f)
        
        return {
            'pointcloud': pointcloud,  # Processed point cloud
            'category': self.classes[category]  # Class label as integer
        }
