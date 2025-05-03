from sklearn.manifold import TSNE
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC
import networkx as nx
from torch_geometric.utils import to_networkx

# Perform t-SNE on the node embeddings
def visualize_embeddings_tsne(node_embeddings, labels=None, random_seed=42):
    tsne = TSNE(n_components=2, random_state=random_seed)
    reduced_embeddings = tsne.fit_transform(node_embeddings.detach().cpu().numpy())

    # Plot the embeddings
    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Labels')
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    plt.title('t-SNE Visualization of Node Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()

# Perform UMAP on the node embeddings
def visualize_embeddings_umap(node_embeddings, labels=None, random_seed=42):
    umap = UMAP(n_components=2, random_state=random_seed)
    reduced_embeddings = umap.fit_transform(node_embeddings.detach().cpu().numpy())

    # Plot the embeddings
    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Labels')
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    plt.title('UMAP Visualization of Node Embeddings')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()

def visualize_whole_graph(data):
    G = to_networkx(data, to_undirected=True, node_attrs=['x'], edge_attrs=['edge_attr'])
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)  # Use spring layout for visualization
    nx.draw(G, pos, with_labels=False, node_size=20, alpha=0.7, edge_color='gray')
    plt.title('Graph Visualization')
    plt.show()

def evaluate_model(model, test_data, random_seed=42):
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        x, edge_index, edge_attr = test_data.x, test_data.edge_index, test_data.edge_attr
        edge_label_index, edge_label = test_data.edge_label_index, test_data.edge_label

        # Get predictions
        node_emb = model.gnn(x, edge_index, edge_attr)
        u_emb = node_emb[edge_label_index[0]]
        v_emb = node_emb[edge_label_index[1]]
        preds = model.link_predictor(u_emb, v_emb)

        # Calculate metrics
        preds_binary = (preds > 0.5).int()
        accuracy = Accuracy(task='binary')(preds_binary, edge_label.int())
        precision = Precision(task='binary')(preds_binary, edge_label.int())
        recall = Recall(task='binary')(preds_binary, edge_label.int())
        f1 = F1Score(task='binary')(preds_binary, edge_label.int())
        auc_value = AUROC(task='binary')(preds, edge_label.int())

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_value:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(edge_label.cpu(), preds_binary.cpu())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(edge_label.cpu(), preds.cpu())
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Predicted vs True Values
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(preds)), preds.cpu().numpy(), label='Predicted', alpha=0.7)
    plt.scatter(range(len(edge_label)), edge_label.cpu().numpy(), label='True', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Predicted vs True Values in Test Dataset')
    plt.legend()
    plt.show()

    # Histogram of Predictions
    plt.figure(figsize=(10, 6))
    plt.hist(preds.cpu().numpy(), bins=50, alpha=0.7, label='Predicted')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Values')
    plt.legend()
    plt.show()

    # Visualize embeddings using t-SNE and UMAP
    visualize_embeddings_tsne(node_emb, random_seed=random_seed)
    visualize_embeddings_umap(node_emb, random_seed=random_seed)