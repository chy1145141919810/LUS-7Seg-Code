import torch
import torch.nn as nn
import torch.optim as optim
from config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH
from dataset import get_data_loaders
from model import get_vit_model

# Fix the global random seed to ensure that the experiments are fully reproducible.
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def main():
    print(f"Currently using computing device: {DEVICE}")
    
    # 1. Get data
    train_loader, val_loader, _, _ = get_data_loaders()
    
    # 2. Get the model and send it to the device
    model = get_vit_model().to(DEVICE)
    
    # 3. Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0

    # 4. Training and validation loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Training phase
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)  
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # Save the model with the highest accuracy.
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Current best model saved, validation accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()