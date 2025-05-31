import torch
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_tensor
from torchvision.models import resnet18
import numpy as np
from PIL import Image
import os

def evaluate_model(model_path, dataset_path, batch_size=128, device="cpu", subset_size=10000, output_dir="output"):
    """
    Evaluates the trained model and generates performance metrics, confusion matrix, and augmentation visuals.
    
    Parameters:
        model_path (str): Path to the saved PyTorch model file (.pt).
        dataset_path (str): Path to the dataset directory.
        batch_size (int): Batch size for evaluation.
        device (str): Device to run evaluation ('cpu' or 'cuda').
        subset_size (int): The number of images to use from the dataset.
        output_dir (str): Directory to save output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset with transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize for faster evaluation
        transforms.ToTensor()
    ])
    
    # Original dataset
    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    
    # Ensure we sample only the first 'subset_size' samples
    indices = np.random.choice(len(full_dataset), subset_size, replace=False)
    subset_dataset = torch.utils.data.Subset(full_dataset, indices)
    
    test_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load the trained model
    model = resnet18(weights=None)  # Initialize the architecture
    num_classes = len(full_dataset.classes)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, num_classes)
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Evaluation loop
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            true_labels.extend(targets.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    # Compute Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")

    # Classification Report
    report = classification_report(true_labels, predicted_labels, target_names=full_dataset.classes, output_dict=True)
    print("\nClassification Report:\n", pd.DataFrame(report).transpose())

    # Save Classification Report as CSV
    pd.DataFrame(report).transpose().to_csv(f"{output_dir}/classification_report.csv", index=True)
    

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.show()

    # Save Augmentation Examples
    def save_augmentation_examples():
        augmentations = transforms.Compose([
            transforms.RandomRotation(40),
            transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])
        
        # Example image path for augmentation demonstration
        sample_image_path = os.path.join(dataset_path, "Apple___Black_rot/image (1).jpg")  # Replace `class_name/sample_image.jpg` with an actual image path
        if not os.path.exists(sample_image_path):
            print(f"Sample image not found: {sample_image_path}")
            return

        image = Image.open(sample_image_path).convert("RGB")
        augmented_images = [augmentations(image) for _ in range(4)]

        plt.figure(figsize=(10, 5))
        for i, img in enumerate(augmented_images):
            plt.subplot(1, 4, i+1)
            plt.imshow(to_tensor(img).permute(1, 2, 0))  # Convert Tensor to image format
            plt.axis('off')
        plt.savefig(f"{output_dir}/augmentation_examples.png")
        plt.show()

    save_augmentation_examples()

    # Generate Metrics Plots
    def plot_metrics():
        metrics = ["precision", "recall", "f1-score"]
        for metric in metrics:
            values = [report[class_name][metric] for class_name in full_dataset.classes]
            plt.figure(figsize=(12, 6))
            plt.bar(full_dataset.classes, values)
            plt.title(f"{metric.capitalize()} by Class")
            plt.xlabel("Classes")
            plt.ylabel(metric.capitalize())
            plt.xticks(rotation=90)
            plt.savefig(f"{output_dir}/{metric}_plot.png")
            plt.show()

    plot_metrics()

if __name__ == "__main__":
    # Update these paths based on your setup
    MODEL_PATH = "optimized_plant_disease_model.pt"  # Path to your trained model
    DATASET_PATH = "Dataset"  # Path to your full dataset directory

    # Run evaluation
    evaluate_model(model_path=MODEL_PATH, dataset_path=DATASET_PATH, batch_size=128, device="cpu", subset_size=10000)
