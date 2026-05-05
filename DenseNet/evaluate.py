import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample
from config import DEVICE, MODEL_SAVE_PATH
from dataset import get_data_loaders
from model import get_densenet_model
import seaborn as sns

# ================= Bootstrap Resampling to Calculate 95% Confidence Interval =================
def compute_bootstrap_ci(y_true, y_pred, y_prob, n_bootstraps=1000, ci=0.95):
    print(f"Performing {n_bootstraps} Bootstrap resampling to calculate 95% confidence interval...")
    boot_acc, boot_prec, boot_rec, boot_f1, boot_kappa, boot_auc = [], [], [], [], [], []
    
    indices = np.arange(len(y_true))
    for i in range(n_bootstraps):
        boot_idx = resample(indices, replace=True, n_samples=len(indices), random_state=i)
        
        y_true_boot = y_true[boot_idx]
        y_pred_boot = y_pred[boot_idx]
        y_prob_boot = y_prob[boot_idx]
        
        try:
            boot_acc.append(accuracy_score(y_true_boot, y_pred_boot))
            boot_prec.append(precision_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
            boot_rec.append(recall_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
            boot_f1.append(f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
            boot_kappa.append(cohen_kappa_score(y_true_boot, y_pred_boot))
            boot_auc.append(roc_auc_score(y_true_boot, y_prob_boot, multi_class='ovr', average='macro'))
        except ValueError:
            continue 
            
    alpha = (1.0 - ci) / 2.0
    lower_p, upper_p = alpha * 100, (1.0 - alpha) * 100
    
    def get_bounds(metric_list):
        return np.percentile(metric_list, lower_p), np.percentile(metric_list, upper_p)
        
    return {
        'acc_ci': get_bounds(boot_acc),
        'prec_ci': get_bounds(boot_prec),
        'rec_ci': get_bounds(boot_rec),
        'f1_ci': get_bounds(boot_f1),
        'kappa_ci': get_bounds(boot_kappa),
        'auc_ci': get_bounds(boot_auc)
    }

def evaluate():
    # 1. Get data and category names.
    _, _, test_loader, class_names = get_data_loaders()
    
    # 2. Load the model and its optimal weights.
    model = get_densenet_model().to(DEVICE)
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

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 4. Calculate the evaluation metrics required for the paper.
    acc = accuracy_score(all_labels, all_preds) 
    macro_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    ci_dict = compute_bootstrap_ci(all_labels, all_preds, all_probs, n_bootstraps=1000)

    print("="*40)
    print("Model evaluation results ( with 95% confidence interval):")
    print(f"Accuracy (ACC):       {acc:.4f} [{ci_dict['acc_ci'][0]:.4f}, {ci_dict['acc_ci'][1]:.4f}]")
    print(f"Macro Precision:      {macro_prec:.4f} [{ci_dict['prec_ci'][0]:.4f}, {ci_dict['prec_ci'][1]:.4f}]")
    print(f"Macro Recall (Sens):  {macro_rec:.4f} [{ci_dict['rec_ci'][0]:.4f}, {ci_dict['rec_ci'][1]:.4f}]")
    print(f"Macro F1-Score:       {macro_f1:.4f} [{ci_dict['f1_ci'][0]:.4f}, {ci_dict['f1_ci'][1]:.4f}]")
    print(f"Cohen's Kappa:        {kappa:.4f} [{ci_dict['kappa_ci'][0]:.4f}, {ci_dict['kappa_ci'][1]:.4f}]")
    print(f"AUC-ROC:              {auc_roc:.4f} [{ci_dict['auc_ci'][0]:.4f}, {ci_dict['auc_ci'][1]:.4f}]")
    print("="*40)

    # ================= Save the results for subsequent McNemar statistical tests =================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the DataFrame to save detailed inference results
    df_results = pd.DataFrame({
        'True_Label': all_labels,
        'Pred_Label': all_preds
    })
    for i in range(all_probs.shape[1]):
        df_results[f'Prob_Class_{i}'] = all_probs[:, i]
        
    csv_save_path = os.path.join(current_dir, 'densenet121_predictions.csv')
    df_results.to_csv(csv_save_path, index=False)
    print(f"Model predictions saved to: {csv_save_path}")

    # 5. Draw and save the confusion matrix
    cm_normalized = confusion_matrix(all_labels, all_preds, normalize='true')
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
    
    plt.title('DenseNet121', fontsize=28, pad=20, weight='bold') 
    plt.xlabel('Predicted Label', fontsize=24, labelpad=15, weight='bold')
    plt.ylabel('True Label', fontsize=24, labelpad=15, weight='bold')
    
    plt.xticks(rotation=45, fontsize=20, weight='bold')
    plt.yticks(rotation=0, fontsize=20, weight='bold') 
    
    plt.tight_layout()
    save_path = os.path.join(current_dir, 'confusion_matrix_densenet121.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"The updated confusion matrix has been saved: {save_path}")

if __name__ == '__main__':
    evaluate()