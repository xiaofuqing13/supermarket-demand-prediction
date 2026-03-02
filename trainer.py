import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# PyTorch库
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class DemandTrainer:
    """
    模型训练器
    """

    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            if len(batch) == 3:
                X, y, sku_ids = batch
                X, y, sku_ids = X.to(self.device), y.to(self.device), sku_ids.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X, sku_ids)

            else:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X)

            loss = self.criterion(outputs, y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    X, y, sku_ids = batch
                    X, y, sku_ids = X.to(self.device), y.to(self.device), sku_ids.to(self.device)
                    outputs = self.model(X, sku_ids)
                else:
                    X, y = batch
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)

                loss = self.criterion(outputs, y)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs=50, early_stop_patience=10):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in tqdm(range(epochs)):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # 学习率调整
            self.scheduler.step(val_loss)

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"早停在 epoch {epoch + 1}")
                    break

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))

        return self.train_losses, self.val_losses

    def plot_training_history(self):
        """
        绘制训练曲线
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss曲线
        axes[0].plot(epochs, self.train_losses, 'b-', label='Training Loss')
        axes[0].plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # 对数坐标
        axes[1].plot(epochs, self.train_losses, 'b-', label='Training Loss')
        axes[1].plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        axes[1].set_title('Model Loss (Log Scale)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (log)')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
