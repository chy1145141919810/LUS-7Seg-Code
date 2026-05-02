import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
from config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH
from dataset import get_data_loaders
from model import get_unet_model

# Fix the global random seed to ensure that the experiments are fully reproducible.
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
def main():
    print(f"Currently using computing device: {DEVICE}")
    
    # 1. Get data
    train_loader, val_loader, _, _ = get_data_loaders()
    
    # 2. Get the model and send it to the device
    model = get_unet_model().to(DEVICE)
    
    # 3. Define the loss function and optimizer
    # Combined use of Cross Entropy (CE) and Dice loss to address class imbalance in medical images.
    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_dice = smp.losses.DiceLoss(mode='multiclass')
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Segmentation tasks usually save the best model by monitoring the decrease in validation set loss.
    best_loss = float('inf') 

    # 4. Training and validation loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Training phase
        for i, (inputs, masks, _) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            masks = masks.to(DEVICE) 
            
            optimizer.zero_grad()
            
            # Model output shape: (Batch, NUM_CLASSES, H, W)
            outputs = model(inputs)
            
            # Calculate the combined loss
            loss_ce = criterion_ce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = loss_ce + loss_dice
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}")
                
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, masks, _ in val_loader:
                inputs = inputs.to(DEVICE)
                masks = masks.to(DEVICE)
                
                outputs = model(inputs)
                
                # Calculate the loss on the validation set
                loss_ce = criterion_ce(outputs, masks)
                loss_dice = criterion_dice(outputs, masks)
                val_loss += (loss_ce + loss_dice).item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} average loss on validation set: {avg_val_loss:.4f}")
        
        # Save the model with the minimum validation set loss.
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Current best model saved, validation set loss decreased to: {best_loss:.4f}")

if __name__ == '__main__':
    main()