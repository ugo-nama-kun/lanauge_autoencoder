import torch
import torchvision
import random
import matplotlib.pyplot as plt

class MNISTHandler:
    def __init__(self, train=True):
        """
        Initializes the MNIST dataset.

        Args:
            train (bool): If True, loads the training dataset; otherwise, loads the test dataset.
        """
        self.dataset = torchvision.datasets.MNIST(root="./data", train=train, download=True)
        self.class_indices = self._index_classes()

    def _index_classes(self):
        """
        Creates a dictionary mapping each class to the indices of its occurrences in the dataset.

        Returns:
            dict: A dictionary where keys are class labels and values are lists of indices.
        """
        class_indices = {i: [] for i in range(10)}
        for i, label in enumerate(self.dataset.targets):
            class_indices[label.item()].append(i)
        return class_indices

    def get_random_image(self, class_number):
        """
        Returns a random image from the specified class.

        Args:
            class_number (int): The class label (0-9) to retrieve an image from.

        Returns:
            PIL.Image: The image corresponding to the specified class.
        """
        if class_number not in self.class_indices:
            raise ValueError("Invalid class number. Choose a number between 0 and 9.")

        random_index = random.choice(self.class_indices[class_number])
        image, label = self.dataset[random_index]
        return image


if __name__ == '__main__':
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Example usage
    mnist_handler = MNISTHandler(train=True)
    for i in range(10):
        img = mnist_handler.get_random_image(i)

        # Display the image
        print(transform(img).max(), transform(img).min())
        plt.imshow(img, cmap="gray")
        plt.title(f"Random MNIST Image of {i}")
        plt.axis("off")
        plt.pause(0.3)
