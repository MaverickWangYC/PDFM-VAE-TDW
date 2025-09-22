import os
os.environ["MKL_THREADING_LAYER"] = "GNU"  # 优先使用GNU线程层，避免冲突
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, matthews_corrcoef
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
from model import mlp, rnn, mamba, transformer  # 导入外部模型
from scipy.linalg import toeplitz, svd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold  # 新增交叉验证所需库
import warnings
import argparse
import json
warnings.filterwarnings('ignore')
plt.rcParams["font.family"] = ["Times New Roman"]
import time
import psutil
import torch.cuda as cuda


def measure_inference_time(model, data_loader, device, num_warmup=10, num_measure=100):
    """测量模型推理时间（ms/样本）"""
    model.eval()
    total_time = 0.0
    total_samples = 0

    # 热身阶段
    with torch.no_grad():
        for i, (features, _) in enumerate(data_loader):
            features = features.to(device)
            if i >= num_warmup:
                break
            model(features)

    # 测量阶段
    start_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None

    with torch.no_grad():
        count = 0
        for features, _ in data_loader:
            if count >= num_measure:
                break
            features = features.to(device)
            batch_size = features.size(0)

            if device.type == 'cuda':
                start_event.record()
                model(features)
                end_event.record()
                cuda.synchronize()
                batch_time = start_event.elapsed_time(end_event)  # ms
            else:
                start = time.perf_counter()
                model(features)
                batch_time = (time.perf_counter() - start) * 1000  # 转换为ms

            total_time += batch_time
            total_samples += batch_size
            count += 1

    return total_time / total_samples  # ms/样本


def measure_training_throughput(model, train_loader, optimizer, device, loss_fn, num_measure=10):
    """测量训练迭代速度（samples/sec）"""
    model.train()
    total_samples = 0
    total_time = 0.0

    with torch.set_grad_enabled(True):
        count = 0
        for features, labels in train_loader:
            if count >= num_measure:
                break
            features, labels = features.to(device), labels.to(device)
            batch_size = features.size(0)

            start = time.perf_counter()
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs['logits'], labels)
            loss.backward()
            optimizer.step()
            batch_time = time.perf_counter() - start  # sec

            total_time += batch_time
            total_samples += batch_size
            count += 1

    return total_samples / total_time  # samples/sec


def measure_memory_usage(model, data_loader, device, is_training=True, batch_size=32):
    """测量内存使用（GB）"""
    if device.type != 'cuda':
        return {
            'peak_memory': psutil.Process().memory_info().rss / 1024 ** 3,  # 系统内存
            'param_count': sum(p.numel() for p in model.parameters()) / 10 ** 6  # 百万参数
        }

    # 清理缓存
    cuda.empty_cache()
    cuda.reset_peak_memory_stats()

    model.train(is_training)
    with torch.set_grad_enabled(is_training):
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            if is_training:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs['logits'], labels)
                loss.backward()
            break  # 仅需一个批次

    peak_memory = cuda.max_memory_allocated() / 1024 ** 3  # 转换为GB
    param_count = sum(p.numel() for p in model.parameters()) / 10 ** 6  # 百万参数

    # 清理
    cuda.empty_cache()
    return {
        'peak_memory': peak_memory,
        'param_count': param_count
    }


def scalability_test(model, base_loader, device, scale_factors=[1.0, 2.0, 4.0], batch_size=32):
    """测试模型在不同数据规模下的可扩展性"""
    results = []
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for scale in scale_factors:
        # 构造缩放后的数据集（重复原始数据）
        scaled_dataset = []
        for _ in range(int(scale)):
            scaled_dataset.extend(base_loader.dataset)

        scaled_loader = DataLoader(
            scaled_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # 测量训练速度
        throughput = measure_training_throughput(
            model, scaled_loader, optimizer, device, loss_fn
        )

        # 测量内存使用
        memory = measure_memory_usage(model, scaled_loader, device)

        results.append({
            'scale_factor': scale,
            'dataset_size': len(scaled_dataset),
            'throughput': throughput,
            'peak_memory': memory['peak_memory'],
            'param_count': memory['param_count']
        })

    return pd.DataFrame(results)


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 动态权重调整模块
class DynamicWeightModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # 统计特征提取器
        self.stats_extractor = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # 输入维度为input_dim*2（均值+方差）
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # 门控机制计算权重
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim // 2, 2),  # 输出两个权重
            nn.Softmax(dim=1)  # 确保权重和为1
        )

    def forward(self, x):
        # 计算输入数据的统计特征
        mean = torch.mean(x, dim=1, keepdim=True)  # 形状: (batch_size, 1)
        var = torch.var(x, dim=1, keepdim=True)  # 形状: (batch_size, 1)

        # 扩展统计特征以匹配输入维度
        mean_expanded = mean.expand(-1, x.size(1))  # 形状: (batch_size, input_dim)
        var_expanded = var.expand(-1, x.size(1))  # 形状: (batch_size, input_dim)

        # 拼接统计特征
        stats = torch.cat([mean_expanded, var_expanded], dim=1)  # 形状: (batch_size, input_dim*2)

        # 提取特征并计算权重
        features = self.stats_extractor(stats)
        weights = self.gate(features)

        return weights  # 返回两个权重：[vae_weight, transformer_weight]

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, csv_files, max_values=None):
        all_data = []
        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            all_data.append(data)
        all_data = pd.concat(all_data, ignore_index=True)

        # ------------------- 新增标签编码逻辑 -------------------
        # 处理字符串标签，映射为整数（如 'ClassA' -> 0, 'ClassB' -> 1）
        label_column = all_data.iloc[:, 0]
        if label_column.dtype == object:  # 判断是否为字符串类型
            unique_labels = label_column.unique()
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            all_data.iloc[:, 0] = label_column.map(label_to_idx)
        # ------------------------------------------------------

        self.labels = torch.tensor(all_data.iloc[:, 0].values, dtype=torch.long)
        features = all_data.iloc[:, 1:].values

        # 处理 NaN 和 inf 值
        features = np.nan_to_num(features)

        if max_values is None:
            self.max_values = features.max(axis=0)
            # 避免除零问题
            self.max_values[self.max_values == 0] = 1
        else:
            self.max_values = max_values

        features = features / self.max_values
        features = np.clip(features, 0, 1)
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        feature = self.features[idx]
        return feature, label

# VAE编码器部分
class VAELatentEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)

    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# 重构解码器
class ReconstructionDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, z):
        return self.decoder(z)

# VAE引导的注意力模块
class VAEGuidedAttention(nn.Module):
    def __init__(self, input_dim, latent_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads

        # 计算补零后的维度，确保能被num_heads整除
        self.padded_input_dim = ((input_dim + num_heads - 1) // num_heads) * num_heads
        self.zero_pad = nn.ZeroPad2d((0, self.padded_input_dim - input_dim))  # 对最后一维补零

        self.q_linear = nn.Linear(self.padded_input_dim, self.padded_input_dim)
        self.k_linear = nn.Linear(self.padded_input_dim, self.padded_input_dim)
        self.v_linear = nn.Linear(self.padded_input_dim, self.padded_input_dim)
        self.out_proj = nn.Linear(self.padded_input_dim, input_dim)  # 输出时恢复原始维度

        self.vae_encoder = VAELatentEncoder(input_dim, latent_dim)
        self.latent_fusion = nn.Linear(latent_dim, self.padded_input_dim)  # 融合到补零后的维度

    def forward(self, x, return_attention=False):
        batch_size, seq_len, dim = x.size()

        # 对输入进行补零，确保dim能被num_heads整除
        x_padded = self.zero_pad(x.unsqueeze(1)).squeeze(1)  # 形状: (batch_size, seq_len, padded_input_dim)
        current_dim = x_padded.size(-1)
        self.head_dim = current_dim // self.num_heads  # 现在head_dim必为整数

        # 拆分多头（q, k, v）
        q = self.q_linear(x_padded).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x_padded).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x_padded).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # VAE编码部分（使用原始输入x，无需补零）
        x_flat = x.view(batch_size * seq_len, dim)
        mu, logvar = self.vae_encoder(x_flat)
        z = self.vae_encoder.reparameterize(mu, logvar)

        # 潜在向量融合到补零后的维度，并拆分多头
        latent_proj = self.latent_fusion(z).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        q_guided = q + latent_proj  # 消融引导
        k_guided = k + latent_proj

        # 计算注意力分数
        scores = torch.matmul(q_guided, k_guided.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)  # 形状: (batch_size, num_heads, seq_len, head_dim)

        # 合并多头并恢复补零前的维度
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, current_dim)
        output = self.out_proj(context)  # 投影回原始输入维度

        if return_attention:
            return output, attn_weights, mu, logvar
        else:
            return output, mu, logvar

def fgsm_attack(inputs, labels, model, eps=0.1, loss_fn=nn.CrossEntropyLoss()):
    """FGSM对抗攻击函数"""
    model.eval()  # 切换至评估模式，避免BN层影响
    inputs = inputs.clone().detach().requires_grad_(True)
    labels = labels.clone().detach()

    # 计算损失并反向传播
    outputs = model(inputs)['logits']
    loss = loss_fn(outputs, labels)
    model.zero_grad()
    loss.backward()

    # 计算梯度符号并生成对抗扰动
    grad = torch.sign(inputs.grad.detach())
    adv_inputs = inputs + eps * grad
    adv_inputs = torch.clamp(adv_inputs, 0, 1)  # 确保输入在[0,1]范围内（根据数据预处理情况调整）

    model.train()  # 恢复训练模式
    return adv_inputs

# Transformer 特征提取器（共享层）
class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, latent_dim, num_heads):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.attention = VAEGuidedAttention(input_dim, latent_dim, num_heads)
        self.Linear_1 = nn.Linear(input_dim, input_dim // 4)
        self.leaky = nn.LeakyReLU(0.2)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.Linear_2 = nn.Linear(input_dim // 4, input_dim)

    def forward(self, x):
        identity = x
        if x.size(0) == 1 and self.training:
            x = x
        else:
            x = self.bn(x)  # 归一化

        # 转换为三维张量 (batch_size, seq_len=1, input_dim)
        x = x.unsqueeze(1)  # 形状: (batch_size, 1, input_dim)
        attn_output, mu, logvar = self.attention(x)  # 自动处理补零
        attn_output = attn_output.squeeze(1)  # 恢复为 (batch_size, input_dim)

        x = self.Linear_1(attn_output)
        x = self.leaky(x)
        x = self.conv(x.unsqueeze(1)).squeeze(1)  # 一维卷积处理
        x = self.Linear_2(x)
        x = x + identity
        return x, mu, logvar

# 整合分类和重构的完整模型（添加动态权重融合）
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, latent_dim, num_heads, hidden_dim, num_classes):
        super().__init__()
        self.feature_extractor = TransformerFeatureExtractor(input_dim, latent_dim, num_heads)

        # 新增：独立VAE特征提取器
        self.vae_encoder = VAELatentEncoder(input_dim, latent_dim)
        self.vae_proj = nn.Linear(latent_dim, input_dim)

        # 新增：动态权重模块
        self.weight_module = DynamicWeightModule(input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        self.decoder = ReconstructionDecoder(latent_dim, input_dim)  # 重构解码器

    def forward(self, x):
        # Transformer特征
        transformer_features, mu, logvar = self.feature_extractor(x)

        # VAE特征
        vae_mu, vae_logvar = self.vae_encoder(x)
        vae_z = self.vae_encoder.reparameterize(vae_mu, vae_logvar)
        vae_features = self.vae_proj(vae_z)

        # 计算动态权重
        weights = self.weight_module(x)
        vae_weight = weights[:, 0].unsqueeze(1)  # 第一维为VAE权重
        transformer_weight = weights[:, 1].unsqueeze(1)  # 第二维为Transformer权重

        # 动态融合特征
        fused_features = vae_weight * vae_features + transformer_weight * transformer_features


        logits = self.classifier(fused_features)
        reconstructed = self.decoder(mu)  # 使用transformer分支的mu进行重构

        return {
            'logits': logits,
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'original': x,
            'vae_weight': vae_weight,  # 新增：返回权重用于监控
            'transformer_weight': transformer_weight  # 新增：返回权重用于监控
        }

# VAE注意力模型的损失函数
def vae_attention_loss(outputs, alpha=1.0, beta=1.0):
    """计算包含分类损失、重构损失和KL散度的总损失"""
    # 分类损失
    criterion = nn.CrossEntropyLoss()
    class_loss = criterion(outputs['logits'], outputs['targets'])

    # 重构损失
    recon_loss = F.mse_loss(outputs['reconstructed'], outputs['original'])

    # KL散度
    mu = outputs['mu']
    logvar = outputs['logvar']
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_divergence = kl_divergence / mu.size(0)

    # 总损失
    total_loss = class_loss + beta * kl_divergence + alpha * recon_loss

    return {
        'total_loss': total_loss,
        'class_loss': class_loss,
        'recon_loss': alpha * recon_loss,
        'kl_divergence': beta * kl_divergence
    }

# 训练分类模型（修改后）
def train_classification(model, train_loader, optimizer, device, reg_param=0.001, alpha=0.2, beta=0.1, eps=0.1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    total_class_loss = 0
    total_recon_loss = 0
    total_kl_divergence = 0
    all_preds = []
    all_labels = []
    loss_fn = nn.CrossEntropyLoss()  # 定义损失函数（用于对抗攻击）

    # 新增：跟踪平均权重
    avg_vae_weight = 0
    avg_transformer_weight = 0
    batch_count = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        batch_size = features.size(0)

        # ---------------------- 原始样本训练 ----------------------
        optimizer.zero_grad()
        # 生成对抗样本
        adv_features = fgsm_attack(features, labels, model, eps, loss_fn)
        # 合并原始样本和对抗样本
        combined_features = torch.cat([features, adv_features], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)

        # 前向传播
        outputs = model(combined_features)
        outputs['targets'] = combined_labels

        # 计算总损失（包含对抗样本）
        losses = vae_attention_loss(outputs, alpha, beta)
        classification_loss = losses['total_loss']

        # 加入L2正则化
        l2_reg = torch.tensor(0., requires_grad=True).to(device)
        for name, param in model.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param)
        classification_loss += reg_param * l2_reg

        classification_loss.backward()
        optimizer.step()

        # ---------------------- 统计指标 ----------------------
        total_loss += classification_loss.item()
        total_class_loss += losses['class_loss'].item()
        total_recon_loss += losses['recon_loss'].item()
        total_kl_divergence += losses['kl_divergence'].item()

        # 原始样本和对抗样本的预测结果
        logits = outputs['logits']
        _, predicted = torch.max(logits.data, 1)
        total += combined_labels.size(0)
        correct += (predicted == combined_labels).sum().item()
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(combined_labels.cpu().tolist())

        # 新增：累加权重用于计算平均值
        avg_vae_weight += outputs['vae_weight'].mean().item()
        avg_transformer_weight += outputs['transformer_weight'].mean().item()
        batch_count += 1

    train_acc = correct / total
    train_f1 = f1_score(all_labels, all_preds, average='macro')
    avg_class_loss = total_class_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_kl_divergence = total_kl_divergence / len(train_loader)

    # 新增：计算平均权重
    avg_vae_weight /= batch_count
    avg_transformer_weight /= batch_count

    print(
        f'Epoch Classification Loss: {avg_class_loss:.4f}, Reconstruction Loss: {avg_recon_loss:.4f}, KL Divergence: {avg_kl_divergence:.4f}')
    print(f'Average VAE Weight: {avg_vae_weight:.4f}, Average Transformer Weight: {avg_transformer_weight:.4f}')

    return total_loss / len(train_loader), train_acc, train_f1

def test_classification_model(model, test_loader, device, save_path=None, save_roc_data=False, roc_data_save_path=None):
    model.eval()
    all_labels = []
    all_probs = []  # 保存预测概率（用于ROC曲线）

    try:
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                logits = outputs['logits']
                probs = F.softmax(logits, dim=1)  # 计算概率

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 处理空数据情况
        if len(all_labels) == 0:
            return {
                'test_acc': 0,
                'test_f1': 0,
                'test_recall': 0,
                'test_precision': 0,
                'auc': np.nan,
                'mcc': np.nan,
                'sensitivity': np.nan,
                'specificity': np.nan,
                'fpr': None,
                'tpr': None,
                'class_metrics': {}
            }

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        num_classes = all_probs.shape[1]
        labels_unique = np.unique(all_labels)

        # 计算基础指标
        y_pred = np.argmax(all_probs, axis=1)
        test_acc = np.mean(all_labels == y_pred)
        test_f1 = f1_score(all_labels, y_pred, average='macro')
        test_recall = recall_score(all_labels, y_pred, average='macro')
        test_precision = precision_score(all_labels, y_pred, average='macro')

        # 初始化指标字典
        metrics = {
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_recall': test_recall,
            'test_precision': test_precision,
            'auc': np.nan,
            'mcc': np.nan,
            'sensitivity': np.nan,
            'specificity': np.nan,
            'fpr': None,
            'tpr': None,
            'class_metrics': {}
        }

        # 处理二分类场景
        if num_classes == 2:
            # 计算敏感度（召回率 for 正类）和特异性（真负率）
            # 假设1是正类，0是负类
            cm = confusion_matrix(all_labels, y_pred)

            # 处理可能的单类情况
            if cm.size == 1:
                tn, fp, fn, tp = 0, 0, 0, 0
                if all_labels[0] == 1:
                    tp = cm[0, 0]
                else:
                    tn = cm[0, 0]
            else:
                tn, fp, fn, tp = cm.ravel()

            # 敏感度 = TP / (TP + FN)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            # 特异性 = TN / (TN + FP)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

            # 计算其他二分类指标
            mcc = matthews_corrcoef(all_labels, y_pred)
            fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
            auc = roc_auc_score(all_labels, all_probs[:, 1])

            # 更新指标字典
            metrics.update({
                'auc': auc,
                'mcc': mcc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'fpr': fpr,
                'tpr': tpr
            })

            # 为每个类别计算指标
            precision = precision_score(all_labels, y_pred, average=None, labels=labels_unique)
            recall = recall_score(all_labels, y_pred, average=None, labels=labels_unique)
            f1 = f1_score(all_labels, y_pred, average=None, labels=labels_unique)

            for i, label in enumerate(labels_unique):
                metrics['class_metrics'][label] = {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i]
                }

        # 处理多分类场景
        else:
            print(f"信息：{num_classes}分类问题，计算多分类指标")
            precision = precision_score(all_labels, y_pred, average=None, labels=labels_unique)
            recall = recall_score(all_labels, y_pred, average=None, labels=labels_unique)
            f1 = f1_score(all_labels, y_pred, average=None, labels=labels_unique)

            for i, label in enumerate(labels_unique):
                metrics['class_metrics'][label] = {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i]
                }

        return metrics

    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        # 确保即使出错也返回一个有效的字典
        return {
            'test_acc': 0,
            'test_f1': 0,
            'test_recall': 0,
            'test_precision': 0,
            'auc': np.nan,
            'mcc': np.nan,
            'sensitivity': np.nan,
            'specificity': np.nan,
            'fpr': None,
            'tpr': None,
            'class_metrics': {}
        }

# 测试分类模型（保持不变）
# def test_classification_model(model, test_loader, device, save_path=None, save_roc_data=False, roc_data_save_path=None):
#     model.eval()
#     all_labels = []
#     all_probs = []  # 保存预测概率（用于ROC曲线）
#
#     with torch.no_grad():
#         for features, labels in test_loader:
#             features = features.to(device)
#             labels = labels.to(device)
#
#             outputs = model(features)
#             logits = outputs['logits']
#             probs = F.softmax(logits, dim=1)  # 计算概率
#
#             all_labels.extend(labels.cpu().numpy())
#             all_probs.extend(probs.cpu().numpy())
#
#     all_probs = np.array(all_probs)
#     all_labels = np.array(all_labels)
#     num_classes = all_probs.shape[1]
#
#
#     labels_unique = np.unique(all_labels)
#
#     # 保存预测结果到CSV
#     if save_path:
#         df_pred = pd.DataFrame({
#             'true_label': all_labels,
#             **{f'class_{i}_prob': all_probs[:, i] for i in range(num_classes)}
#         })
#         df_pred.to_csv(save_path, index=False)
#         print(f"预测结果已保存至: {save_path}")
#     metrics = {
#         'test_acc': np.mean(all_labels == np.argmax(all_probs, axis=1)),
#         'test_f1': f1_score(all_labels, np.argmax(all_probs, axis=1), average='macro'),
#         'test_recall': recall_score(all_labels, np.argmax(all_probs, axis=1), average='macro'),
#         'test_precision': precision_score(all_labels, np.argmax(all_probs, axis=1), average='macro'),
#         'auc': np.nan,
#         'mcc': np.nan,
#         'sensitivity': np.nan,
#         'specificity': np.nan,
#         'fpr': None,
#         'tpr': None,
#         'class_metrics': {}  # 存储每个类别的指标
#     }
#
#     # 处理二分类场景
#     if num_classes == 2:
#         # 初始化指标字典
#
#         y_true = all_labels
#         y_score = all_probs[:, 1]  # 假设正类为标签1
#
#         # 裁剪概率值到[0,1]
#         y_score = np.clip(y_score, 0, 1)
#
#         # 计算ROC曲线和AUC
#         fpr, tpr, thresholds = roc_curve(y_true, y_score)
#         thresholds = np.clip(thresholds, 0, 1)  # 确保阈值有效
#         auc = roc_auc_score(y_true, y_score)
#
#         # 计算敏感性和特异性（使用默认阈值0.5）
#         y_pred = np.argmax(all_probs, axis=1)
#         cm = confusion_matrix(y_true, y_pred)
#         if cm.shape == (2, 2):
#             tn, fp, fn, tp = cm.ravel()
#             sensitivity = tp / (tp + fn) if (tp + fn) != 0 else np.nan
#             specificity = tn / (tn + fp) if (tn + fp) != 0 else np.nan
#         else:
#             sensitivity = np.nan
#             specificity = np.nan
#         mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
#         mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator != 0 else 0
#         # 更新指标字典
#         metrics.update({
#             'auc': auc,
#             'sensitivity': sensitivity,
#             'specificity': specificity,
#             'mcc': mcc,
#             'fpr': fpr.tolist(),
#             'tpr': tpr.tolist()
#         })
#
#         # 保存ROC数据
#         if save_roc_data and roc_data_save_path:
#             df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
#             df_roc.to_csv(roc_data_save_path, index=False)
#             print(f"ROC数据已保存至: {roc_data_save_path}")
#
#     # 多分类场景提示
#     else:
#         print(f"警告：{num_classes}分类问题，不支持ROC曲线")
#         y_pred = np.argmax(all_probs, axis=1)
#         precision = precision_score(all_labels, y_pred, average=None, labels=labels_unique)
#         recall = recall_score(all_labels, y_pred, average=None, labels=labels_unique)
#         f1 = f1_score(all_labels, y_pred, average=None, labels=labels_unique)
#
#         for i, label in enumerate(labels_unique):
#             metrics['class_metrics'][label] = {
#                 'precision': precision[i],
#                 'recall': recall[i],
#                 'f1': f1[i]
#             }
#         print(f"警告：{num_classes}分类任务，已计算每个类别的精确率、召回率、F1值")
#
#     return metrics


def feature_ablation_analysis(model, data_loader, device, input_dim):
    model.eval()
    # 计算基线性能
    baseline_metrics = test_classification_model(model, data_loader, device)
    baseline_f1 = baseline_metrics['test_f1']
    baseline_sensitivity = baseline_metrics['sensitivity']
    baseline_specificity = baseline_metrics['specificity']
    print(f"基线 F1 分数: {baseline_f1:.4f}")

    feature_importance = []

    # 对每个特征进行消融测试
    # for feat_idx in range(input_dim):
    #     # 创建消融后的数据加载器
    #     ablation_loader = []
    #     for features, labels in data_loader:
    #         features = features.clone().to(device)
    #         features[:, feat_idx] = 0  # 将该特征设为0
    #         ablation_loader.append((features, labels))
    ablation_loader = []
    # 为当前特征生成随机打乱的索引
    shuffle_indices = torch.randperm(len(data_loader.dataset)).to(device)

    for feat_idx in range(input_dim):
        # 创建打乱特征后的数据加载器
        ablation_loader = []
        # 为当前特征生成随机打乱的索引（保持在CPU上，因为数据集通常在CPU上）
        torch.manual_seed(19)
        shuffle_indices = torch.randperm(len(data_loader.dataset))  # 不使用.to(device)

        for batch_idx, (features, labels) in enumerate(data_loader):
            features = features.clone().to(device)
            batch_size = features.size(0)

            # 获取当前批次在整个数据集中的索引范围
            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + batch_size

            # 获取当前批次对应的打乱后索引（保持在CPU上）
            batch_shuffle_indices = shuffle_indices[start_idx:end_idx]

            # 打乱当前特征列
            # 从数据集中获取打乱后的特征值（在CPU上操作）
            shuffled_features = data_loader.dataset[batch_shuffle_indices][0][:, feat_idx]
            # 将打乱后的特征值移动到与features相同的设备
            features[:, feat_idx] = shuffled_features.to(device)

            ablation_loader.append((features, labels))

        # 评估消融后的模型性能
        with torch.no_grad():
            all_labels = []
            all_preds = []
            for feats, labs in ablation_loader:
                outputs = model(feats)
                _, preds = torch.max(outputs['logits'], 1)
                all_labels.extend(labs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            # 计算消融后的 F1 分数
            f1 = f1_score(all_labels, all_preds, average='macro')
            cm = confusion_matrix(all_labels, all_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) != 0 else np.nan
                specificity = tn / (tn + fp) if (tn + fp) != 0 else np.nan
            else:
                sensitivity = np.nan
                specificity = np.nan
            # importance = baseline_sensitivity - sensitivity
            # importance = baseline_specificity - specificity
            importance = baseline_f1 - f1  # 重要性 = 基线性能 - 消融后性能
            feature_importance.append(importance)

        # 打印进度
        if (feat_idx + 1) % 50 == 0:
            print(f"已完成 {feat_idx + 1}/{input_dim} 个特征的消融测试")

    # 归一化重要性分数
    feature_importance = np.array(feature_importance)
    print(feature_importance.sum())
    if feature_importance.sum() > 0:
        feature_importance = feature_importance / feature_importance.sum()
    else:
        # 处理总和为0的情况，例如返回均匀分布或保持原值
        feature_importance = feature_importance
    # feature_importance = feature_importance / feature_importance.sum()

    return feature_importance


def feature_perturbation_analysis(model, data_loader, device, input_dim, sigma=0.1):
    model.eval()
    feature_importance = np.zeros(input_dim)

    for feat_idx in range(input_dim):
        disturb_f1 = []
        for features, labels in data_loader:
            features = features.clone().to(device)
            # 添加高斯噪声（标准差sigma）
            noise = torch.randn_like(features[:, feat_idx]) * sigma
            features[:, feat_idx] += noise
            features = torch.clamp(features, 0, 1)  # 确保数据在归一化范围内

            with torch.no_grad():
                outputs = model(features)
                probs = F.softmax(outputs['logits'], dim=1)
                preds = np.argmax(probs.cpu().numpy(), axis=1)
                f1 = f1_score(labels.cpu().numpy(), preds, average='binary')
            disturb_f1.append(f1)

            # 计算重要性得分（基线F1 - 扰动后平均F1）
            baseline_f1 = test_classification_model(model, data_loader, device)['test_f1']
            importance = baseline_f1 - np.mean(disturb_f1)
            feature_importance[feat_idx] = importance

    return feature_importance / feature_importance.sum()  # 标准化


def rank_features(feature_importance, feature_names=None, top_n=20):
    """对特征按重要性进行排序，并返回排名结果"""
    indices = np.argsort(feature_importance)[::-1]  # 降序排列
    sorted_importance = feature_importance[indices]

    # 创建特征名称（如果未提供，则使用索引）
    if feature_names is None:
        feature_names = [f"ID {i + 1}" for i in range(len(feature_importance))]

    sorted_names = [feature_names[i] for i in indices]

    # 返回排名结果
    rank_results = pd.DataFrame({
        'features': sorted_names,
        'importance': sorted_importance
    })

    return rank_results.head(top_n)


# 可视化分析函数
def visualize_feature_importance(rank_results, title="feature importance analysis", save_path=None):
    # 设置中文字体支持，增加宋体以支持中文显示
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
    # 统一设置字体大小
    plt.rcParams["font.size"] = 30  # 基础字体大小
    plt.rcParams["axes.titlesize"] = 30  # 标题字体大小
    plt.rcParams["axes.labelsize"] = 25  # 坐标轴标签字体大小
    plt.rcParams["xtick.labelsize"] = 25  # x轴刻度字体大小
    plt.rcParams["ytick.labelsize"] = 25  # y轴刻度字体大小
    plt.rcParams["legend.fontsize"] = 25  # 图例字体大小

    # 显式创建figure和axes对象，便于colorbar关联
    fig, ax = plt.subplots(figsize=(10, 12))

    # 创建颜色映射 - 根据重要性值生成渐变色
    norm = plt.Normalize(rank_results['importance'].min(), rank_results['importance'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    # 准备调色板并转换为列表，解决numpy数组警告
    palette = list(plt.cm.viridis(norm(rank_results['importance'].values)))

    # 为解决hue警告，添加一个占位列
    rank_results = rank_results.copy()
    rank_results['hue_placeholder'] = 0  # 所有值相同的占位列

    # 绘制柱状图并设置颜色，修复palette和hue相关警告
    sns.barplot(
        x="importance",
        y="features",
        # hue="hue_placeholder",  # 使用占位列作为hue
        data=rank_results,
        palette=palette,
        legend=False,  # 不显示图例，保持原效果
        ax=ax  # 指定axes对象
    )

    # 添加颜色条，并明确关联到axes对象，解决colorbar错误
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.tick_params(labelsize=30)  # 颜色条刻度字体大小
    cbar.set_label('importance', fontsize=25)  # 添加颜色条标签

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")

    return fig, ax


def visualize_feature_heatmap(feature_importance, dataset_name, feature_names=None, top_n=30, save_path=None):
    """可视化特征重要性热图"""
    plt.rcParams['figure.dpi'] = 300

    indices = np.argsort(feature_importance)[::-1][:top_n]  # 获取前top_n个重要特征
    important_features = feature_importance[indices]

    if feature_names is None:
        feature_names = [f"{i + 1}" for i in range(len(feature_importance))]

    important_names = [feature_names[i] for i in indices]

    # 创建热图数据
    heatmap_data = np.zeros((top_n, top_n))
    for i in range(top_n):
        for j in range(top_n):
            if i == j:
                heatmap_data[i, j] = important_features[i]
            else:
                heatmap_data[i, j] = (important_features[i] + important_features[j]) / 2

    # 绘制热图
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(heatmap_data*100, annot=False, #annot_kws={"size": 15},# cmap="YlGnBu",
                xticklabels=important_names, yticklabels=important_names,
                linewidths=.5)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28)
    ax.set_xticklabels(important_names, fontsize=28, rotation=45)
    ax.set_yticklabels(important_names, fontsize=28, rotation=30)
    plt.title(f"VAE-TDW importance heat map of {dataset_name} (×10²)", fontsize=30)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热图已保存至: {save_path}")

    # plt.show()


class CVCustomDataset(Dataset):
    def __init__(self, all_data, max_values=None, indices=None):
        # 筛选当前折的数据集
        self.data = all_data.iloc[indices].reset_index(drop=True) if indices is not None else all_data

        # 标签编码（复用原逻辑）
        label_column = self.data.iloc[:, 0]
        if label_column.dtype == object:
            unique_labels = label_column.unique()
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.data.iloc[:, 0] = label_column.map(self.label_to_idx)
        else:
            self.label_to_idx = None

        # 特征处理（复用原逻辑）
        self.labels = torch.tensor(self.data.iloc[:, 0].values, dtype=torch.long)
        features = self.data.iloc[:, 1:].values
        features = np.nan_to_num(features)

        if max_values is None:
            self.max_values = features.max(axis=0)
            self.max_values[self.max_values == 0] = 1
        else:
            self.max_values = max_values

        features = features / self.max_values
        features = np.clip(features, 0, 1)
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---------------------- 2. 新增：交叉验证核心训练函数 ----------------------
def kfold_cross_validation(
        all_csv_files, k=5, input_dim=None, model_type='vae_transformer', latent_dim=128, num_heads=8,
        hidden_dim=512, batch_size=128, epochs=100, learning_rate=1e-4,
        reg_param=5e-5, final_lr_ratio=0.9, alpha=1.0, beta=0.1, eps=0.5
):
    # 1. 数据加载部分
    all_data = []
    for csv_file in all_csv_files:
        data = pd.read_csv(csv_file)
        all_data.append(data)
    all_data = pd.concat(all_data, ignore_index=True)
    num_classes = len(all_data.iloc[:, 0].unique())
    if input_dim is None:
        input_dim = len(all_data.columns) - 1

    # 获取标签列用于分层抽样（假设第一列是标签）
    labels = all_data.iloc[:, 0].values

    # 2. 初始化分层K折划分器（核心修改：使用StratifiedKFold）
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=19)
    cv_results = {'fold_metrics': [], 'avg_acc': 0, 'avg_f1': 0, 'avg_recall': 0,
                  'avg_precision': 0, 'avg_sensitivity': 0, 'avg_specificity': 0, 'avg_auc': 0}

    # 3. 逐折训练与评估
    # 核心修改：使用标签进行分层划分
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_data, labels)):
        print(f"\n=== 开始 {k} 折交叉验证 - 第 {fold + 1} 折 ===")
        print(f"训练集样本数: {len(train_idx)}, 测试集样本数: {len(test_idx)}")

        # 3.1 数据集准备
        train_data = all_data.iloc[train_idx]
        train_features = np.nan_to_num(train_data.iloc[:, 1:].values)
        train_max_values = train_features.max(axis=0)
        train_max_values[train_max_values == 0] = 1

        # 创建训练集和测试集
        train_dataset = CVCustomDataset(all_data, max_values=train_max_values, indices=train_idx)
        test_dataset = CVCustomDataset(all_data, max_values=train_max_values, indices=test_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 3.2 数据集检查
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"警告：第 {fold + 1} 折训练集/测试集为空，跳过此折")
            continue

        def has_enough_classes(dataloader):
            classes = set()
            for _, labels in dataloader:
                classes.update(labels.numpy())
                if len(classes) == num_classes:
                    return True
            return len(classes) > 0

        if not has_enough_classes(train_loader) or not has_enough_classes(test_loader):
            print(f"警告：第 {fold + 1} 折类别不完整，跳过此折")
            continue

        # 3.3 模型初始化
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == 'vae_transformer':
            model = TransformerClassifier(input_dim, latent_dim, num_heads, hidden_dim, num_classes).to(device)
        elif model_type == 'mlp':
            model = mlp(input_dim, hidden_dim, num_classes).to(device)
        elif model_type == 'rnn':
            model = rnn(input_dim, hidden_dim, num_classes).to(device)
        elif model_type == 'mamba':
            model = mamba(input_dim, hidden_dim, num_classes).to(device)
        elif model_type == 'transformer':
            model = transformer(input_dim, num_heads, hidden_dim, num_classes).to(device)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        gamma = final_lr_ratio ** (1 / epochs)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        # 3.4 单折训练
        best_train_f1 = 0.0
        best_model_state = None

        for epoch in range(epochs):
            train_loss, train_acc, train_f1 = train_classification(
                model, train_loader, optimizer, device, reg_param, alpha, beta, eps
            )

            # 基于训练指标保存最优模型
            if train_f1 >= best_train_f1:
                best_train_f1 = train_f1
                best_model_state = model.state_dict().copy()

            print(
                f"第 {fold + 1} 折 - 第 {epoch + 1}/{epochs} 轮: "
                f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, 训练F1: {train_f1:.4f} | "
                f"最优训练F1: {best_train_f1:.4f}"
            )
            scheduler.step()

        # 3.5 测试集评估
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            test_metrics = test_classification_model(
                model, test_loader, device,
                save_path=f'cv_fold_{fold + 1}_test_preds.csv',
                save_roc_data=True,
                roc_data_save_path=f'cv_fold_{fold + 1}_test_roc.csv'
            )

            # 保存测试集指标
            final_metrics = {
                'fold': fold + 1,
                'acc': test_metrics['test_acc'],
                'f1': test_metrics['test_f1'],
                'recall': test_metrics['test_recall'],
                'precision': test_metrics['test_precision'],
                'sensitivity': test_metrics.get('sensitivity', np.nan),
                'specificity': test_metrics.get('specificity', np.nan),
                'auc': test_metrics.get('auc', np.nan)
            }
            cv_results['fold_metrics'].append(final_metrics)

            print(f"\n第 {fold + 1} 折测试结果:")
            print(f"准确率: {final_metrics['acc']:.4f}, F1值: {final_metrics['f1']:.4f}")

        else:
            print(f"警告：第 {fold + 1} 折未获得有效模型状态")

    # 4. 结果计算
    fold_metrics = cv_results['fold_metrics']
    if not fold_metrics:
        print("警告：没有有效的交叉验证结果")
        return cv_results

    cv_results['avg_acc'] = np.nanmean([m['acc'] for m in fold_metrics])
    cv_results['avg_f1'] = np.nanmean([m['f1'] for m in fold_metrics])
    cv_results['avg_recall'] = np.nanmean([m['recall'] for m in fold_metrics])
    cv_results['avg_precision'] = np.nanmean([m['precision'] for m in fold_metrics])
    cv_results['avg_sensitivity'] = np.nanmean([m['sensitivity'] for m in fold_metrics])
    cv_results['avg_specificity'] = np.nanmean([m['specificity'] for m in fold_metrics])
    cv_results['avg_auc'] = np.nanmean([m['auc'] for m in fold_metrics])

    # 5. 打印总结（保持不变）
    print("\n=== K折交叉验证总结 ===")
    print(f"有效折数: {len(fold_metrics)}/{k}")
    print(f"平均准确率: {cv_results['avg_acc']:.4f} (±{np.nanstd([m['acc'] for m in fold_metrics]):.4f})")
    print(f"平均F1值: {cv_results['avg_f1']:.4f} (±{np.nanstd([m['f1'] for m in fold_metrics]):.4f})")
    # 其他指标打印...

    cv_results_df = pd.DataFrame(fold_metrics)
    cv_results_df.to_csv('cross_validation_results.csv', index=False)
    print(f"\n交叉验证详细结果已保存至: cross_validation_results.csv")

    return cv_results


def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 数组转列表
    elif isinstance(obj, np.generic):
        return obj.item()    # 单个NumPy数值转Python原生数值
    elif isinstance(obj, dict):
        # 重点：先将字典的键转换为Python类型，再递归处理值
        new_dict = {}
        for k, v in obj.items():
            # 转换键：如果是NumPy整数/浮点数，转为Python原生类型
            if isinstance(k, np.integer):
                k = int(k)
            elif isinstance(k, np.floating):
                k = float(k)
            # 递归处理值
            new_dict[k] = convert_numpy_types(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]  # 递归处理列表
    return obj

# 主函数（保持不变）
def main():
    # 设置随机种子
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认值：42）')
    parser.add_argument('--model', type=str, default='transformer',
                        choices=['vae_transformer', 'mlp', 'rnn', 'mamba', 'transformer'],
                        help='模型类型选择')
    args = parser.parse_args()  # 解析参数

    # 修改：使用外部传入的seed，而非固定值42
    seed = args.seed  # 从命令行参数获取seed
    set_seed(seed)
    model_type = args.model
    dataset_name = 'cat'
    roc_data_save_path = f'roc_data_{model_type}_{dataset_name}.csv'

    all_data_files = [  # 所有数据文件（交叉验证需合并所有数据）
        f"./{dataset_name}/train.csv",
        f"./{dataset_name}/test.csv"
    ]

    use_cross_validation = True  # 是否启用交叉验证（True/False）
    k_folds = 5  # 交叉验证折数

    latent_dim = 128  # VAE潜在空间维度
    num_heads = 8
    hidden_dim = 512

    batch_size = 128
    epochs = 100
    learning_rate = 5e-5
    reg_param = 1e-5
    final_lr_ratio = 0.9  # 最终学习率与初始学习率的比例
    alpha = 1.0
    beta = 0.1  # VAE损失权重
    eps = 0 # .5

    if use_cross_validation:
        print("=== 启用 K 折交叉验证模式 ===")
        cv_results = kfold_cross_validation(
            all_csv_files=all_data_files,
            k=k_folds,
            model_type=model_type,  # 传递模型类型
            latent_dim=latent_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            reg_param=reg_param,
            final_lr_ratio=final_lr_ratio,
            alpha=alpha,
            beta=beta,
            eps=eps
        )
    else:

        file_1 = f"./{dataset_name}/train.csv"
        file_2 = f"./{dataset_name}/test.csv"
        train_csv_files = [file_1]
        test_csv = file_2
        input_dim = len(pd.read_csv(train_csv_files[0]).columns) - 1
        num_classes = len(pd.read_csv(train_csv_files[0])['label'].unique())

        # 计算指数衰减因子
        gamma = final_lr_ratio ** (1 / epochs)

        train_dataset = CustomDataset(train_csv_files)
        test_dataset = CustomDataset([test_csv], max_values=train_dataset.max_values)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        feature_num = train_dataset.features.size(-1)  # 等价于 train_dataset.features.shape[-1]
        print(f"train_dataset 的特征数（每个样本的特征维度）: {feature_num}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        classification_model = TransformerClassifier(input_dim, latent_dim, num_heads, hidden_dim, num_classes).to(device)

        optimizer = optim.Adam(classification_model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        print("开始分类模型训练...")
        best_train_f1 = 0
        best_test_metrics = None
        best_model_state = None
        for epoch in range(epochs):
            train_loss, train_acc, train_f1 = train_classification(classification_model, train_loader, optimizer, device,
                                                                   reg_param, alpha, beta, eps)
            test_metrics = test_classification_model(
                classification_model, test_loader, device, save_path=None, save_roc_data=False, roc_data_save_path=None)

            if train_f1 > best_train_f1:
                best_train_f1 = train_f1
                best_test_metrics = test_metrics
                best_model_state = classification_model.state_dict().copy()

            print(
                f'Epoch {epoch + 1}/{epochs}, '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, '
                f'Best Test Acc: {best_test_metrics["test_acc"]:.4f}, Best Test F1: {best_test_metrics["test_f1"]:.4f}, '
                f'Best Sensitivity: {best_test_metrics["sensitivity"]:.4f}, Best Specificity: {best_test_metrics["specificity"]:.4f}, '
                f'Best Recall: {best_test_metrics["test_recall"]:.4f}, Best Precision: {best_test_metrics["test_precision"]:.4f}, '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
            )
            scheduler.step()

        # 加载最佳模型
        if best_model_state is not None:
            classification_model.load_state_dict(best_model_state)
            test_metrics = test_classification_model(
                classification_model, test_loader, device, save_path=None, save_roc_data=True,
                roc_data_save_path=roc_data_save_path)
            # 打印最终指标（包含AUC）
            print("\n最终测试指标:")
            print(f"准确率: {test_metrics['test_acc']:.4f}")
            print(f"F1值: {test_metrics['test_f1']:.4f}")
            print(f"召回率: {test_metrics['test_recall']:.4f}")
            print(f"精确率: {test_metrics['test_precision']:.4f}")
            print(f"敏感度: {test_metrics['sensitivity']:.4f}")
            print(f"特异性: {test_metrics['specificity']:.4f}")
            print(f"MCC: {test_metrics['mcc']:.4f}")

            if test_metrics.get('class_metrics'):
                print("\n各分类指标:")
                for label, metrics in test_metrics['class_metrics'].items():
                    print(f"类别 {label}:")
                    print(f"  精确率: {metrics['precision']:.4f}")
                    print(f"  召回率: {metrics['recall']:.4f}")
                    print(f"  F1值: {metrics['f1']:.4f}")

            if test_metrics['auc'] != np.nan:
                print(f"AUC值: {test_metrics['auc']:.4f}")
                print(f"已保存FPR/TPR数据至: {roc_data_save_path}")
            else:
                print("多分类问题，不支持AUC和ROC曲线。")

            # 转换后再序列化
            test_metrics_converted = convert_numpy_types(test_metrics)
            print(f"METRICS: {json.dumps(test_metrics_converted)}")  # 输出标准格式的指标

            benchmark_results = {
                'inference_time_ms_per_sample': inference_time,
                'training_throughput_samples_per_sec': train_throughput,
                'train_peak_memory_gb': train_memory['peak_memory'],
                'infer_peak_memory_gb': infer_memory['peak_memory'],
                'param_count_million': train_memory['param_count'],
                'scalability': convert_numpy_types(scalability_results.to_dict('records'))
            }

            with open(f'{model.__class__.__name__}_benchmark_{dataset_name}.json', 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            print(f"基准测试结果已保存至: {model.__class__.__name__}_benchmark_{dataset_name}.json")
        #
        # print("\n开始特征重要性分析...")
        # feature_importance = feature_ablation_analysis(classification_model, train_loader, device, input_dim)
        #
        # # 特征排名
        # rank_results = rank_features(feature_importance, top_n=int(feature_num*0.05))
        # print("\n前20个最重要的特征:")
        # print(rank_results)
        #
        # # 保存排名结果
        # rank_results.to_csv(f'feature_ranking_{dataset_name}.csv', index=False)
        # print(f"特征排名结果已保存至: feature_ranking_{dataset_name}.csv")
        #
        # # 可视化分析
        # visualize_feature_importance(
        #     rank_results,
        #     title=f"{dataset_name} analysis",
        #     save_path=f'feature_importance_{dataset_name}.png'
        # )
        #
        # # 可视化热图（可选）
        # visualize_feature_heatmap(
        #     feature_importance,
        #     dataset_name,
        #     top_n=15,
        #     save_path=f'feature_heatmap_{dataset_name}.png'
        # )
        #
        # # 保存完整的特征重要性分数
        # pd.DataFrame({
        #     'feature_index': range(1, input_dim + 1),
        #     'importance_score': feature_importance
        # }).to_csv(f'full_feature_importance_{dataset_name}.csv', index=False)
        # print(f"完整特征重要性分数已保存至: full_feature_importance_{dataset_name}.csv")



if __name__ == "__main__":
    main()