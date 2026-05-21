import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


class PatchEmbedding(nn.Module):
    """高光谱图像的Patch Embedding层"""

    def __init__(self, img_size=13, patch_size=1, in_channels=103, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # 对于高光谱图像，使用1D卷积进行光谱维度的特征提取
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H, W)
        x = self.projection(x)
        # 展平为patches: (B, embed_dim, H*W)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, embed_dim=384, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """前馈网络"""

    def __init__(self, embed_dim=384, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, embed_dim=384, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        # 自注意力 + 残差连接
        x = x + self.attn(self.norm1(x))
        # MLP + 残差连接
        x = x + self.mlp(self.norm2(x))
        return x


class ImageReconstructor(nn.Module):
    """图像重建器：从特征向量重建原始图像
    用于对抗性特征纯化：重建器试图从特征重建原始图像，特征提取器试图阻止重建"""

    def __init__(self, embed_dim=256, img_size=13, in_channels=103):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.in_channels = in_channels

        # 从特征向量重建图像
        # 首先将特征向量扩展为空间特征图
        self.feature_expand = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim * img_size * img_size)
        )

        # 将特征图转换为原始图像
        self.reconstructor = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2,
                               kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim //
                               4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, in_channels,
                               kernel_size=1, stride=1),
            nn.Sigmoid()  # 确保输出在 [0, 1] 范围内
        )

    def forward(self, features):
        """
        Args:
            features: (B, embed_dim) 特征向量
        Returns:
            reconstructed: (B, in_channels, img_size, img_size) 重建的图像
        """
        B = features.shape[0]
        # 将特征向量扩展为空间特征图
        # (B, embed_dim * img_size * img_size)
        x = self.feature_expand(features)
        x = x.view(B, self.embed_dim, self.img_size, self.img_size)
        # 重建图像
        # (B, in_channels, img_size, img_size)
        reconstructed = self.reconstructor(x)
        return reconstructed


class Pureformer(nn.Module):
    """大幅简化的自蒸馏Vision Transformer，适用于高光谱图像
    核心改进：embed_dim=256, depth=3, num_heads=4, mlp_ratio=2
    显著减少参数量同时保留自蒸馏机制
    添加对抗性特征纯化：分类任务和重建任务共享特征，形成自对抗"""

    def __init__(self, img_size=13, patch_size=1, in_channels=103, num_classes=7,
                 embed_dim=256, depth=3, num_heads=4, mlp_ratio=2, dropout=0.1, use_reconstruction=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.depth = depth
        self.img_size = img_size
        self.in_channels = in_channels
        self.use_reconstruction = use_reconstruction

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim)

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(
            1, (img_size//patch_size)**2 + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 中间层分类头（用于自蒸馏）
        self.intermediate_heads = nn.ModuleList([
            nn.Linear(embed_dim, num_classes) for _ in range(depth)
        ])

        # 图像重建器（用于对抗性特征纯化）
        if self.use_reconstruction:
            self.reconstructor = ImageReconstructor(
                embed_dim, img_size, in_channels)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_features=False):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置编码
        x = x + self.pos_embed

        # 存储中间层输出（用于自蒸馏）
        intermediate_outputs = []

        # 通过Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            # 提取CLS token的输出
            cls_output = x[:, 0]  # (B, embed_dim)
            intermediate_outputs.append(cls_output)

        # 最终输出
        x = self.norm(x)
        final_features = x[:, 0]  # (B, embed_dim) 最终特征向量
        final_output = self.head(final_features)  # 使用CLS token

        # 中间层输出
        intermediate_logits = []
        for i, head in enumerate(self.intermediate_heads):
            intermediate_logits.append(head(intermediate_outputs[i]))

        # 如果需要返回特征用于重建
        if return_features:
            return final_output, intermediate_logits, final_features
        else:
            return final_output, intermediate_logits


class PureformerTrainer:
    """Pureformer训练器 - 对抗性特征纯化机制

    对抗性训练原理（信息论角度）：
    任务A（分类任务）：只优化"要保"信息 → 传统监督损失
      - 特征提取器 + 分类器：最小化分类损失（让特征包含分类信息）

    任务B（重建任务）：让网络无法重建"要扔"信息 → 对抗式重建
      - 重建器：最小化重建损失（试图从特征重建原始图像）
      - 特征提取器：最大化重建损失（对抗重建器，让特征不包含原始图像信息）

    两任务共享同一份特征 = 自对抗
    """

    def __init__(self, model, device, alpha=0.8, alpha_kl=2.0, beta=0.01,
                 num_epochs=100, warmup_epochs=5):
        self.model = model
        self.device = device
        self.alpha = alpha  # 自蒸馏损失权重
        self.alpha_kl = alpha_kl  # KL散度温度参数
        self.beta = beta  # 重建损失权重（对抗性特征纯化）
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs

        # 分离参数：特征提取器/分类器 vs 重建器
        feature_params = []
        recon_params = []
        for name, param in model.named_parameters():
            if 'reconstructor' in name:
                recon_params.append(param)
            else:
                feature_params.append(param)

        # 优化器 - 特征提取器和分类器
        self.optimizer = optim.AdamW(
            feature_params, lr=5e-4, weight_decay=1e-4, betas=(0.9, 0.999))

        # 重建器的优化器（单独优化）
        if model.use_reconstruction and len(recon_params) > 0:
            self.recon_optimizer = optim.AdamW(
                recon_params, lr=5e-4, weight_decay=1e-4, betas=(0.9, 0.999))
        else:
            self.recon_optimizer = None

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.recon_loss = nn.MSELoss()  # 重建损失

        # 梯度裁剪
        self.max_grad_norm = 1.0

    def train_step(self, x, y):
        """单步训练 - 对抗性特征纯化

        训练流程：
        1. 任务A（分类任务）：特征提取器 + 分类器，最小化分类损失
        2. 任务B（对抗性重建）：
           a. 重建器：最小化重建损失（试图从特征重建原始图像）
           b. 特征提取器：最大化重建损失（对抗重建器，让特征不包含原始图像信息）
        """
        self.model.train()
        # x, y = x.to(self.device), y.to(self.device)

        # 前向传播 - 返回特征向量用于重建
        final_output, intermediate_outputs, final_features = self.model(
            x, return_features=True)

        # ========== 任务A：分类任务（只优化"要保"信息） ==========
        # 基础分类损失：确保特征包含尽可能多的分类信息
        base_loss = self.ce_loss(final_output, y)

        # 自蒸馏损失
        distillation_loss = 0.0
        if len(intermediate_outputs) > 0:
            # 随机选择一个中间层
            random_idx = random.randint(0, len(intermediate_outputs) - 1)
            intermediate_output = intermediate_outputs[random_idx]

            # KL散度蒸馏损失
            distillation_loss = F.kl_div(
                F.log_softmax(intermediate_output / self.alpha_kl, dim=1),
                F.log_softmax(final_output / self.alpha_kl, dim=1),
                reduction='sum',
                log_target=True
            ) * (self.alpha_kl * self.alpha_kl) / intermediate_output.numel()

        # 分类任务总损失
        classification_loss = base_loss + self.alpha * distillation_loss

        # 反向传播分类损失（更新特征提取器和分类器）
        self.optimizer.zero_grad()
        classification_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # ========== 任务B：对抗性重建（让网络无法重建"要扔"信息） ==========
        reconstruction_loss = 0.0
        if self.model.use_reconstruction and self.beta > 0 and self.recon_optimizer is not None:
            # 步骤1：重建器优化 - 最小化重建损失（试图从特征重建原始图像）
            # 重新前向传播获取特征（基于更新后的特征提取器），但使用detach()断开梯度
            with torch.no_grad():
                _, _, final_features_for_recon = self.model(
                    x, return_features=True)

            # 使用detach()断开特征提取器的梯度，只更新重建器
            reconstructed_img = self.model.reconstructor(
                final_features_for_recon.detach())
            recon_loss_for_decoder = self.recon_loss(reconstructed_img, x)

            # 重建器优化：最小化重建损失
            self.recon_optimizer.zero_grad()
            recon_loss_for_decoder.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.reconstructor.parameters(), self.max_grad_norm)
            self.recon_optimizer.step()

            # 步骤2：特征提取器对抗 - 最大化重建损失（对抗重建器）
            # 重新前向传播获取新的特征（基于更新后的特征提取器和重建器）
            _, _, final_features_adv = self.model(x, return_features=True)
            reconstructed_img_adv = self.model.reconstructor(
                final_features_adv)
            reconstruction_loss = self.recon_loss(reconstructed_img_adv, x)

            reconstruction_loss = torch.clamp(
                reconstruction_loss, min=0.0, max=1.0)

            # 反向传播对抗损失（更新特征提取器，对抗重建器）
            # 这里改为最小化 (1 - reconstruction_loss)，reconstruction_loss ∈ [0, 1]
            # 等价于最大化 reconstruction_loss，梯度方向与 -reconstruction_loss 一致
            adv_loss = 1.0 - reconstruction_loss
            self.optimizer.zero_grad()
            (self.beta * adv_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # 总损失（用于记录）
        total_loss = classification_loss + \
            (self.beta * adv_loss if self.model.use_reconstruction else 0.0)

        return {
            'total_loss': total_loss.item(),
            'base_loss': base_loss.item(),
            'distillation_loss': distillation_loss.item(),
            'reconstruction_loss': reconstruction_loss.item() if self.model.use_reconstruction else 0.0,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device) - 1
                with torch.cuda.amp.autocast():
                    final_output, _ = self.model(x)
                preds = torch.argmax(final_output, dim=1)
                all_preds.append(preds)
                all_labels.append(y)

        # 在 GPU 上拼接并计算精度
        all_preds_tensor = torch.cat(all_preds)          # 仍在 self.device 上
        all_labels_tensor = torch.cat(all_labels)        # 仍在 self.device 上
        accuracy = (all_preds_tensor ==
                    all_labels_tensor).float().mean().item()

        # 为了兼容后续 numpy 计算（如混淆矩阵），再转成 numpy 返回
        all_preds_np = all_preds_tensor.cpu().numpy()
        all_labels_np = all_labels_tensor.cpu().numpy()
        return accuracy, all_preds_np, all_labels_np
