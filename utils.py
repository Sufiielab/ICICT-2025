import numpy as np
import math
import random
import torch


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise ValueError('Not a valid OFF header')
    
    num_vertices, num_faces, _ = tuple([int(x) for x in file.readline().strip().split(' ')])
    
    vertices = [[float(x) for x in file.readline().strip().split(' ')] for _ in range(num_vertices)]
    faces = [[int(x) for x in file.readline().strip().split(' ')][1:] for _ in range(num_faces)]
    
    return vertices, faces


class PointCloudSampler:
    def __init__(self, num_samples):
        assert isinstance(num_samples, int), "num_samples must be an integer"
        self.num_samples = num_samples

    def compute_triangle_area(self, vertex1, vertex2, vertex3):
        side_a = np.linalg.norm(vertex1 - vertex2)
        side_b = np.linalg.norm(vertex2 - vertex3)
        side_c = np.linalg.norm(vertex3 - vertex1)
        semi_perimeter = 0.5 * (side_a + side_b + side_c)
        area = max(semi_perimeter * (semi_perimeter - side_a) * (semi_perimeter - side_b) * (semi_perimeter - side_c), 0)**0.5
        return area

    def generate_random_point(self, vertex1, vertex2, vertex3):
        s, t = sorted([random.random(), random.random()])
        interpolate = lambda i: s * vertex1[i] + (t - s) * vertex2[i] + (1 - t) * vertex3[i]
        return (interpolate(0), interpolate(1), interpolate(2))

    def __call__(self, mesh):
        vertices, faces = mesh
        vertices = np.array(vertices)
        triangle_areas = np.zeros(len(faces))

        for i, face in enumerate(faces):
            triangle_areas[i] = self.compute_triangle_area(vertices[face[0]], vertices[face[1]], vertices[face[2]])

        sampled_faces = random.choices(faces, weights=triangle_areas, k=self.num_samples)

        sampled_points = np.zeros((self.num_samples, 3))

        for i, face in enumerate(sampled_faces):
            sampled_points[i] = self.generate_random_point(vertices[face[0]], vertices[face[1]], vertices[face[2]])

        return sampled_points


class NormalizePointCloud:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2, "Pointcloud must be a 2D array"

        normalized_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        normalized_pointcloud /= np.max(np.linalg.norm(normalized_pointcloud, axis=1))

        return normalized_pointcloud


class RandomRotationZ:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2, "Pointcloud must be a 2D array"

        theta = random.random() * 2 * math.pi
        rotation_matrix = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ])
        
        rotated_pointcloud = rotation_matrix.dot(pointcloud.T).T
        return rotated_pointcloud


class AddRandomNoise:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2, "Pointcloud must be a 2D array"

        noise = np.random.normal(0, 0.02, pointcloud.shape)
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud


class ToTensor:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2, "Pointcloud must be a 2D array"
        
        return torch.from_numpy(pointcloud)
