import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from config import DEVICE, MODEL_SAVE_PATH
from dataset import get_data_loaders
from model import get_resnet50_model
import seaborn as sns

def evaluate():
    # 1. Get data and category names.
    _, _, test_loader, class_names = get_data_loaders()
    
    # 2. Load the model and its optimal weights.
    model = get_resnet50_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []

    # 3. Perform inference and collect results.
    print("Evaluating the model and collecting prediction results, please wait...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)

            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 4. Calculate the evaluation metrics required for the paper.
    acc = accuracy_score(all_labels, all_preds) 
    macro_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)

    auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

    print("="*40)
    print(f"Test set Accuracy (ACC):       {acc:.4f}")
    print(f"Test set Macro Precision:      {macro_prec:.4f}")
    print(f"Test set Macro Recall (Sens):  {macro_rec:.4f}")
    print(f"Test set Macro F1-Score:       {macro_f1:.4f}")
    print(f"Test set Cohen's Kappa:        {kappa:.4f}")
    print(f"Test set AUC-ROC:              {auc_roc:.4f}")
    print("="*40)

    # 5. Draw and save the confusion matrix.
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # (1) Calculate the normalized confusion matrix.
    cm_normalized = confusion_matrix(all_labels, all_preds, normalize='true')
    # Convert to percentage format.
    cm_percentage = cm_normalized * 100  
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax, 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 22, "weight": "bold"}, 
                cbar_kws={'label': 'Percentage (%)'})
    
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(22)
    cbar.ax.yaxis.label.set_weight('bold')
    cbar.ax.tick_params(labelsize=18)
    
    plt.title('ResNet50', fontsize=28, pad=20, weight='bold') 
    plt.xlabel('Predicted Label', fontsize=24, labelpad=15, weight='bold')
    plt.ylabel('True Label', fontsize=24, labelpad=15, weight='bold')
    
    plt.xticks(rotation=45, fontsize=20, weight='bold')
    plt.yticks(rotation=0, fontsize=20, weight='bold') 
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, 'confusion_matrix_resnet50.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"The updated confusion matrix has been saved: {save_path}")

if __name__ == '__main__':
    evaluate()