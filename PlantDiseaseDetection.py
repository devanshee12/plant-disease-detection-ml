import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
from PIL import Image
import torchvision.transforms.functional as TF

def main():
    # Step 1: Dataset Preparation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Reduce resolution for faster processing
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder("Dataset", transform=transform)

    # Use a subset of the dataset
    subset_size = 10000
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    train_indices, validation_indices, test_indices = np.split(indices, [
        int(0.7 * subset_size),  # 70% for training
        int(0.85 * subset_size)  # 15% for validation
    ])

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # DataLoader
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=validation_sampler, num_workers=4)

    # Step 2: Load Pre-trained Model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False  # Freeze pre-trained layers

    # Replace the classification layer
    num_classes = len(dataset.class_to_idx)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Step 3: Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters())

    # Step 4: Training Loop with Early Stopping
    best_val_loss = float('inf')
    patience = 2
    trigger_times = 0
    epochs = 5

    for e in range(epochs):
        start_time = datetime.now()

        # Training Phase
        model.train()
        train_loss = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        # Validation Phase
        model.eval()
        val_loss = np.mean([
            criterion(model(inputs.to(device)), targets.to(device)).item()
            for inputs, targets in validation_loader
        ])

        print(f"Epoch {e+1}/{epochs} - Train Loss: {np.mean(train_loss):.3f}, Validation Loss: {val_loss:.3f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

        print(f"Epoch Duration: {datetime.now() - start_time}")

    # Step 5: Save the Model
    torch.save(model.state_dict(), 'optimized_plant_disease_model.pt')

    # Step 6: Load the Model for Inference
    model.load_state_dict(torch.load('optimized_plant_disease_model.pt', weights_only=True))
    model.eval()

    # Step 7: Prediction Function
    def single_prediction(image_path):
        image = Image.open(image_path).resize((128, 128))
        input_data = TF.to_tensor(image).unsqueeze(0).to(device)
        output = model(input_data)
        index = output.argmax(dim=1).item()
        return dataset.classes[index]

    # Test Prediction
    try:
        print(single_prediction("test_images/sample_image (1).jpg"))
    except FileNotFoundError:
        print("Error: Image file not found. Please provide a valid image path.")

if __name__ == "__main__":
    main()
