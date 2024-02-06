import torch

import math
import numpy as np
import matplotlib.pyplot as plt


def read_txt_to_tensor(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        numbers = [float(num) for line in lines for num in line.split()]
        tensor = torch.tensor(numbers).view(4, 4)
    return tensor


def create_six_digit_strings():
    six_digit_strings = []
    for i in range(1, 250):
        six_digit_string = str(i).zfill(6)
        six_digit_strings.append(six_digit_string)
    return six_digit_strings


def main():
    ids = create_six_digit_strings()

    poses = []

    for id in ids:
        file_path = f"/Users/piotrwojcik/PycharmProjects/SSDNeRF/demo/example_pose/pose/{id}.txt"  # Path to your text file
        pose = read_txt_to_tensor(file_path)
        point = [0, 0, 0, 1]
        point = torch.tensor(point).float().view(4, 1)
        p_car = torch.matmul(pose, point)
        xyz_car = p_car.tolist()
        # Extract first three numbers and convert to tuple
        xyz_car = tuple(xyz_car[:3])
        poses.append(xyz_car)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for pose in poses:
        ax.scatter(*pose, color='b', s=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Poses on Sphere')

    plt.show()


if __name__ == "__main__":
    main()