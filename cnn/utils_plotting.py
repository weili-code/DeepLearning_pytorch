# imports from installed libraries
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_accuracy(train_acc_list, valid_acc_list, results_dir):
    num_epochs = len(train_acc_list)

    plt.plot(np.arange(1, num_epochs + 1), train_acc_list, label="Training")
    plt.plot(np.arange(1, num_epochs + 1), valid_acc_list, label="Testing")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # plt.tight_layout()

    if results_dir is not None:
        image_path = os.path.join(results_dir, "plot_acc_training_validation.pdf")
        plt.savefig(image_path)


def plot_loss(train_loss_list, valid_loss_list, results_dir):
    num_epochs = len(train_loss_list)

    plt.plot(np.arange(1, num_epochs + 1), train_loss_list, label="Training")
    plt.plot(np.arange(1, num_epochs + 1), valid_loss_list, label="Testing")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    if results_dir is not None:
        image_path = os.path.join(results_dir, "plot_loss_training_validation.pdf")
        plt.savefig(image_path)


def show_images(
    images,
    labels,
    predictions=None,
    num_images=10,
    normalize=True,
    mean=None,
    std=None,
    label_dict=None,
):
    """
    Displays images along with their true labels and, optionally, predicted labels.

    Args:
        images (numpy.ndarray): (N, C, H, W) Array of images.
        labels (numpy.ndarray): (N, ) Array of true numerical labels.
        predictions (numpy.ndarray, optional): (N, ) Array of predicted numerical labels.
        num_images (int, optional): Number of images to display.
        normalize (bool, optional): Indicates if the images were normalized.
        mean (tuple or float, optional): Mean used for normalization (per channel).
        std (tuple or float, optional): Standard deviation used for normalization (per channel).
        label_dict (dict, optional): Dictionary mapping numerical labels to string labels.

    e.g. if the images (N, C=3, H, W) are normalized through
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        i.e, normalize channel 0 with mean 0.485, and std 0.229;
             normalize channel 1 with mean 0.456, and std 0.224;
             etc.
        shall use
        show_images(images, labels, predictions, num_images=10, normalize=True,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        un-normalizing process is img = img * std + mean
    """

    def imshow(img, label, prediction=None):
        img = np.transpose(img, (1, 2, 0))  # Transpose to (H, W, C)
        if normalize and mean is not None and std is not None:
            if img.shape[2] == 1:  # Grayscale image with channel dimension
                img = img * std + mean
                img = img.squeeze()  # Remove the channel dimension
            else:  # RGB or multi-channel image
                img = (img * np.array(std)) + np.array(mean)
                # img: (H, W, C), np.array(mean): (C,)
                # broadcast as (H, W, C)* (1, 1, C)+ (1, 1, C)
        img = np.clip(img, 0, 1)  # Clip values to be within [0, 1] range
        plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
        plt.axis("off")
        true_label = label_dict[label] if label_dict and label in label_dict else label
        pred_label = (
            label_dict[prediction]
            if prediction is not None and label_dict and prediction in label_dict
            else prediction
        )
        title = f"True: {true_label}"
        if prediction is not None:
            title += f", Pred: {pred_label}"
        plt.title(title, fontsize=8)

    rows = num_images // 5 + (num_images % 5 != 0)
    plt.figure(figsize=(12, 2 * rows))
    for i in range(min(num_images, len(images))):
        plt.subplot(rows, 5, i + 1)
        imshow(
            images[i], labels[i], predictions[i] if predictions is not None else None
        )

    plt.subplots_adjust(hspace=0.4 if rows > 1 else 0, wspace=0.1)
    plt.show()


def plot_confusion_matrix(true_labels, predicted_labels, label_dict=None):
    """
    Plots a confusion matrix using true and predicted labels with class names.
    Allowing multiple-classes.

    Args:
        true_labels (numpy.ndarray): (N, ) Array of true numerical labels.
        predicted_labels (numpy.ndarray): (N, ) Array of predicted numerical labels.
        label_dict (dict, optional): Dictionary mapping numerical labels to class names.
    """
    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Use label_dict if provided, else use unique labels as class names
    if label_dict:
        class_names = [
            label_dict.get(label, label)
            for label in sorted(np.unique(true_labels))
            # label_dict.get fetches the string value associated with unique numerical labels in the true_labels
            # If label is not a key in label_dict, it defaults to just the numerical label.
        ]
    else:
        class_names = sorted(np.unique(np.concatenate((true_labels, predicted_labels))))
        # set to a list of all unique class labels found in both the true and predicted numerical labels.

    # Plotting using seaborn for better visualization
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
