import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GINEConv, GINConv
from torchmetrics import Accuracy, AUROC

class GNNModel(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, dropout_rate):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout_rate = dropout_rate
        for _ in range(num_layers):
            conv = GINConv(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                train_eps=True,
            )
            self.convs.append(conv)
            input_dim = hidden_dim
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

# Link Prediction Decoder with Edge Attributes
class LinkPredictor(nn.Module):
    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        mlp_layers = []
        current_dim = 2 * hidden_dim
        for _ in range(num_layers):
            mlp_layers.append(nn.Linear(current_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            current_dim = hidden_dim
        mlp_layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, u_emb, v_emb):
        # Ensure u_emb and v_emb are at least 2-dimensional
        if u_emb.dim() == 1:
            u_emb = u_emb.unsqueeze(0)
        if v_emb.dim() == 1:
            v_emb = v_emb.unsqueeze(0)
        
        concatenated = torch.cat([u_emb, v_emb], dim=1)
        return self.mlp(concatenated).squeeze()

# Lightning Module for Training
class LitGNN(pl.LightningModule):
    def __init__(self,
                input_dim,
                # Conv layer parameters
                conv_layers=2,
                conv_hidden_dim=64,
                # Link prediction parameters
                link_predictor_layers=1,
                link_hidden_dim=64,
                # Training parameters
                learning_rate=0.001,
                weight_decay=0.0001,
                dropout_rate=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.gnn = GNNModel(input_dim, conv_layers, conv_hidden_dim, dropout_rate)
        self.link_predictor = LinkPredictor(link_predictor_layers, link_hidden_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Initialize metrics
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')
        self.train_auroc = AUROC(task='binary')
        self.val_auroc = AUROC(task='binary')
        self.test_auroc = AUROC(task='binary')
    
    def forward(self, x, edge_index):
        return self.gnn(x, edge_index)
    
    def training_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        edge_label_index, edge_label = batch.edge_label_index, batch.edge_label

        node_emb = self.gnn(x, edge_index)
        u_emb = node_emb[edge_label_index[0]]
        v_emb = node_emb[edge_label_index[1]]
        preds = self.link_predictor(u_emb, v_emb)

        loss = self.loss_fn(preds, edge_label.float())
        self.log('train_loss', loss, prog_bar=True)
        # print(f"Train Loss: {loss:.4f}")

        # Calculate accuracy
        probs = torch.sigmoid(preds)
        preds_binary = (probs > 0.5).int()
        acc = self.train_acc(preds_binary, edge_label.int())
        self.log('train_acc', acc, prog_bar=True)
        # print(f"Train Accuracy: {acc:.4f}")

        # Calculate AUC-ROC
        if len(torch.unique(edge_label)) > 1:
            auc = self.train_auroc(preds, edge_label.int())
            self.log('train_auc', auc, prog_bar=True)
            # print(f"Train AUC: {auc:.4f}")

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        edge_label_index, edge_label = batch.edge_label_index, batch.edge_label

        node_emb = self.gnn(x, edge_index)
        u_emb = node_emb[edge_label_index[0]]
        v_emb = node_emb[edge_label_index[1]]
        preds = self.link_predictor(u_emb, v_emb)

        loss = self.loss_fn(preds, edge_label.float())
        self.log('val_loss', loss, prog_bar=True)

        # Calculate accuracy
        probs = torch.sigmoid(preds)
        preds_binary = (probs > 0.5).int()
        acc = self.val_acc(preds_binary, edge_label.int())
        self.log('val_acc', acc, prog_bar=True)

        # Calculate AUC-ROC
        if len(torch.unique(edge_label)) > 1:  # Ensure both classes are present
            auc = self.val_auroc(preds, edge_label.int())
            self.log('val_auc', auc, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        edge_label_index, edge_label = batch.edge_label_index, batch.edge_label

        node_emb = self.gnn(x, edge_index)
        u_emb = node_emb[edge_label_index[0]]
        v_emb = node_emb[edge_label_index[1]]
        preds = self.link_predictor(u_emb, v_emb)

        loss = self.loss_fn(preds, edge_label.float())
        self.log('test_loss', loss, prog_bar=True)

        # Calculate accuracy
        probs = torch.sigmoid(preds)
        preds_binary = (probs > 0.5).int()
        acc = self.test_acc(preds_binary, edge_label.int())
        self.log('test_acc', acc, prog_bar=True)

        # Calculate AUC-ROC
        if len(torch.unique(edge_label)) > 1:
            auc = self.test_auroc(preds, edge_label.int())
            self.log('test_auc', auc, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
    
    def predict_winner(self, id1, id2, data):
        with torch.no_grad():
            # Get node embeddings using the GNN model
            x, edge_index = data.x, data.edge_index
            node_emb = self.gnn(x, edge_index)
            
            # Get embeddings for the lineups
            u_emb = node_emb[id1]
            v_emb = node_emb[id2]
            
            # Get prediction using the link predictor
            pred = self.link_predictor(u_emb, v_emb)
            probs = torch.sigmoid(pred)
            
        return probs.item()
    
    def predict_enemy_lineup(self, lineup_id, enemy_team_lineup_ids, data):
        preds = []
        for enemy_id in enemy_team_lineup_ids:
            pred = self.predict_winner(lineup_id, enemy_id, data)
            preds.append({
                'enemy_id': enemy_id,
                'pred': pred
            })
    
        if not preds:
            return None
            
        return preds
