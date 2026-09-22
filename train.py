#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import argparse
from datetime import datetime
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from utils import confusion_matrix_gpu, kappa_coefficient_gpu, tpr_gpu
from network.Pureformer import Pureformer, PureformerTrainer
try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# 导入自定义模块
from utils_HSI import sample_gt, seed_worker
from datasets import get_dataset, HyperX
import shutil as shutil_module


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Pureformer for Hyperspectral Image Domain Generalization')

    # 数据相关参数
    parser.add_argument('--data_path', type=str, default='./dataset/',
                        help='数据集路径')
    parser.add_argument('--source_name', type=str, default='Houston13',
                        help='源域数据集名称')
    parser.add_argument('--target_name', type=str, default='Houston18',
                        help='目标域数据集名称')
    parser.add_argument('--save_path', type=str, default='./results/',
                        help='结果保存路径')

    # 模型相关参数
    parser.add_argument('--patch_size', type=int, default=13,
                        help='图像块大小')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='嵌入维度')
    parser.add_argument('--depth', type=int, default=3,
                        help='Transformer层数')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='注意力头数')

    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='学习率')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='自蒸馏损失权重')
    parser.add_argument('--alpha_kl', type=float, default=2.0,
                        help='KL散度温度参数')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='重建损失权重（对抗性特征纯化：让特征尽可能少包含原始图像信息）')
    parser.add_argument('--use_reconstruction', action='store_true', default=True,
                        help='是否使用重建损失进行对抗性特征纯化')

    # 数据采样相关参数
    parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                        help='训练样本比例')
    parser.add_argument('--re_ratio', type=int, default=5,
                        help='重采样比例')

    # 数据增强相关参数
    parser.add_argument('--flip_augmentation', action='store_true', default=True,
                        help='是否启用翻转数据增强')
    parser.add_argument('--radiation_augmentation', action='store_true', default=True,
                        help='是否启用辐射噪声数据增强')
    parser.add_argument('--mixture_augmentation', action='store_true', default=True,
                        help='是否启用MixUp数据增强')
    parser.add_argument('--center_pixel', action='store_true', default=True,
                        help='是否使用中心像素')

    # 其他参数
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备号')
    parser.add_argument('--seed', type=int, default=344,
                        help='随机种子')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志打印间隔')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    seed_worker(args.seed)

    # 设置设备
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 设置cudnn
    torch.backends.cudnn.benchmark = True

    # 创建保存目录
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    save_dir = os.path.join(
        args.save_path, f'{args.source_name}_to_{args.target_name}_{time_str}')
    os.makedirs(save_dir, exist_ok=True)

    # 创建评估结果文件
    evaluate_file = os.path.join(save_dir, 'evaluate_result.txt')
    with open(evaluate_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Pureformer高光谱图像领域泛化评估结果\n")
        f.write(f"源域: {args.source_name} -> 目标域: {args.target_name}\n")
        f.write(f"开始时间: {time_str}\n")
        f.write("="*80 + "\n\n")

    # 设置TensorBoard (可选)
    if TENSORBOARD_AVAILABLE:
        try:
            writer = SummaryWriter(save_dir)
            use_tensorboard = True
        except:
            writer = None
            use_tensorboard = False
            print("TensorBoard初始化失败，将跳过日志记录")
    else:
        writer = None
        use_tensorboard = False
        print("TensorBoard不可用，将跳过日志记录")

    # 加载数据集
    print("加载数据集...")

    # 根据数据集名称确定正确的子文件夹路径
    def get_dataset_path(dataset_name, base_path):
        """根据数据集名称返回正确的路径"""
        if dataset_name in ['Houston13', 'Houston18']:
            return os.path.join(base_path, 'Houston/')
        elif dataset_name in ['paviaU', 'paviaC']:
            return os.path.join(base_path, 'Pavia/')
        elif dataset_name in ['Dioni', 'Loukia']:
            return os.path.join(base_path, 'HyRANK/')
        else:
            return base_path

    # 获取源域和目标域的数据路径
    src_path = get_dataset_path(args.source_name, args.data_path)
    tar_path = get_dataset_path(args.target_name, args.data_path)

    print(f"源域路径: {src_path}")
    print(f"目标域路径: {tar_path}")

    img_src, gt_src, label_values_src, ignored_labels, rgb_bands, palette = get_dataset(
        args.source_name, src_path)
    img_tar, gt_tar, label_values_tar, _, _, _ = get_dataset(
        args.target_name, tar_path)

    # 数据预处理
    r = args.patch_size // 2
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)),
                    'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)),
                    'constant', constant_values=(0, 0))

    # 数据采样
    train_gt_src, val_gt_src, _, _ = sample_gt(
        gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1.0, mode='random')

    # 计算样本比例参数tmp，用于控制源域和目标域的样本比例
    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])
    tmp = args.training_sample_ratio * args.re_ratio * sample_num_src / sample_num_tar

    # 获取数据集信息
    num_classes = int(gt_src.max())
    n_bands = img_src.shape[-1]

    print(f"源域: {args.source_name}, 目标域: {args.target_name}")
    print(f"类别数: {num_classes}, 波段数: {n_bands}")
    print(f"源域样本数: {sample_num_src}, 目标域样本数: {sample_num_tar}")
    print(f"样本比例参数tmp: {tmp:.4f}")
    print(f"源域训练样本: {len(np.nonzero(train_gt_src)[0])}")
    print(f"源域验证样本: {len(np.nonzero(val_gt_src)[0])}")
    print(f"目标域测试样本: {len(np.nonzero(test_gt_tar)[0])}")

    # 根据tmp参数进行数据扩充（仿照train_hsi.py）
    img_src_con, train_gt_src_con = img_src, train_gt_src
    val_gt_src_con = val_gt_src

    if tmp < 1:  # 训练数据不满足比例要求，进行扩充
        print(f"tmp < 1, 进行数据扩充, 重采样比例: {args.re_ratio}")
        for i in range(args.re_ratio - 1):
            img_src_con = np.concatenate((img_src_con, img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
            val_gt_src_con = np.concatenate((val_gt_src_con, val_gt_src))

        print(f"扩充后源域训练样本: {len(np.nonzero(train_gt_src_con)[0])}")
        print(f"扩充后源域验证样本: {len(np.nonzero(val_gt_src_con)[0])}")

        # 使用扩充后的数据
        img_src = img_src_con
        train_gt_src = train_gt_src_con
        val_gt_src = val_gt_src_con

    # 创建hyperparams字典
    hyperparams = vars(args).copy()
    hyperparams.update({
        'ignored_labels': ignored_labels,
        'n_classes': num_classes,
        'n_bands': n_bands,
        'device': args.gpu,
        'supervision': 'full'
    })

    train_dataset = HyperX(img_src, train_gt_src, **hyperparams)
    val_dataset = HyperX(img_src, val_gt_src, **hyperparams)
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=16)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=16)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True,
                             persistent_workers=True,
                             prefetch_factor=16)
    # 创建模型
    model = Pureformer(
        img_size=args.patch_size,
        patch_size=1,
        in_channels=n_bands,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=0.1,  # 增加dropout以防止过拟合
        use_reconstruction=args.use_reconstruction
    ).to(device)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建训练器 - 优化自蒸馏参数和对抗性特征纯化参数
    trainer = PureformerTrainer(
        model, device,
        alpha=args.alpha,
        alpha_kl=args.alpha_kl,
        beta=args.beta,
        num_epochs=args.num_epochs
    )

    # 训练循环
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_test_kappa = 0.0
    best_test_tpr = []
    best_test_cm = None

    print("开始训练...")

    # 记录训练时间
    training_times = []

    for epoch in range(1, args.num_epochs + 1):
        epoch_start_time = time.time()

        # 训练阶段开始
        model.train()
        train_losses = []
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            y = y - 1
            losses = trainer.train_step(x, y)
            train_losses.append(losses)

        # 计算平均损失
        avg_losses = {k: np.mean([l[k] for l in train_losses])
                      for k in train_losses[0].keys()}

        # 训练集评估开始
        train_eval_start_time = time.time()
        train_acc, _, _ = trainer.evaluate(train_loader)
        train_eval_time = time.time() - train_eval_start_time

        # 验证集评估开始
        val_start_time = time.time()
        val_acc, _, _ = trainer.evaluate(val_loader)
        val_time = time.time() - val_start_time

        # 使用验证集OA保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(
                save_dir, 'best_model.pth'))
            print(f"🎯 新的最佳验证准确率: {val_acc:.4f} (Epoch {epoch})")

        # 测试阶段（如果需要）
        test_time = 0.0
        test_acc = 0.0
        if epoch % args.log_interval == 0:
            test_start_time = time.time()
            # 加载保存的最优模型权重进行测试
            if os.path.exists(os.path.join(save_dir, 'best_model.pth')):
                # 保存当前模型状态
                current_state = deepcopy(model.state_dict())
                # 加载最优模型权重
                model.load_state_dict(torch.load(
                    os.path.join(save_dir, 'best_model.pth')))
                # 使用最优模型进行测试
                test_acc, test_preds, test_labels = trainer.evaluate(
                    test_loader)
                # 恢复当前模型状态
                model.load_state_dict(current_state)

            test_time = time.time() - test_start_time

        # 统计到测试结束的轮次耗时，包含训练、训练集评估、验证和测试
        epoch_time = time.time() - epoch_start_time
        training_times.append(epoch_time)
        avg_epoch_time = np.mean(training_times)

        if epoch % args.log_interval == 0:
            # 将 numpy 结果转为 GPU tensor
            test_preds_tensor = torch.from_numpy(test_preds).to(device)
            test_labels_tensor = torch.from_numpy(test_labels).to(device)

            # 在GPU上计算详细评估指标
            cm_gpu = confusion_matrix_gpu(
                test_preds_tensor, test_labels_tensor, num_classes)
            kappa = kappa_coefficient_gpu(cm_gpu)
            class_tpr = tpr_gpu(cm_gpu)

            # 保存和打印时再转回CPU numpy
            cm = cm_gpu.cpu().numpy()

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_kappa = kappa
                best_test_tpr = class_tpr.copy()
                best_test_cm = cm.copy()
                test_acc_rounded = round(test_acc * 100, 2)
                target_best_path = os.path.join(
                    save_dir, f'best_target_model_{test_acc_rounded:.2f}.pth')
                shutil_module.copy(os.path.join(
                    save_dir, 'best_model.pth'), target_best_path)
                print(f"保存新的最佳测试目标域模型: {target_best_path}")
            # 计算平均训练时间
            avg_training_time = np.mean(training_times)

            # 保存评估结果到文件
            with open(evaluate_file, 'a', encoding='utf-8') as f:
                f.write(f"Epoch {epoch:3d} 评估结果:\n")
                f.write(f"  平均训练时间: {avg_training_time:.4f}秒\n")
                f.write(f"  目标域OA: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
                f.write(f"  目标域Kappa: {kappa:.4f} ({kappa*100:.2f}%)\n")
                f.write(f"  各类TPR: {[f'{tpr:.4f}' for tpr in class_tpr]}\n")
                f.write(f"  混淆矩阵:\n")
                for i, row in enumerate(cm):
                    f.write(f"    类别{i}: {row}\n")
                f.write(f"  最佳测试准确率: {best_test_acc:.4f}\n")
                f.write("-" * 60 + "\n\n")

            print(
                f"Epoch {epoch}: 目标域OA={test_acc:.4f}, Kappa={kappa:.4f}, 平均训练时间={avg_training_time:.2f}s")

        # 记录日志
        if use_tensorboard:
            writer.add_scalar('Loss/Total', avg_losses['total_loss'], epoch)
            writer.add_scalar('Loss/Base', avg_losses['base_loss'], epoch)
            writer.add_scalar('Loss/Distillation',
                              avg_losses['distillation_loss'], epoch)
            if 'reconstruction_loss' in avg_losses:
                writer.add_scalar('Loss/Reconstruction',
                                  avg_losses['reconstruction_loss'], epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            if test_acc > 0:
                writer.add_scalar('Accuracy/Test', test_acc, epoch)

        # 打印训练信息
        recon_loss_str = f', Recon_Loss={avg_losses.get("reconstruction_loss", 0):.4f}' if avg_losses.get(
            'reconstruction_loss', 0) > 0 else ''
        print(f'Epoch {epoch:3d}/{args.num_epochs}: '
              f'Loss={avg_losses["total_loss"]:.4f}, '
              f'Base={avg_losses["base_loss"]:.4f}, '
              f'Distill={avg_losses["distillation_loss"]:.4f}'
              f'{recon_loss_str}, '
              f'Train_Acc={train_acc:.4f}, '
              f'Val_Acc={val_acc:.4f}, '
              f'Best_Val={best_val_acc:.4f}, '
              f'LR={trainer.optimizer.param_groups[0]["lr"]:.6f}, '
              f'Time={epoch_time:.2f}s')

        # 打印详细进度
        if epoch % args.log_interval == 0:
            test_acc_display = test_acc if test_acc > 0 else 0.0
            print(f'Test_Acc={test_acc_display:.4f}, '
                  f'Best_Test={best_test_acc:.4f}, '
                  f'Avg_Time={avg_epoch_time:.2f}s')

        # 打印耗时信息
        print(f'Epoch {epoch}: Train eval time: {train_eval_time:.2f}s, Val time: {val_time:.2f}s, Test time (if any): {test_time:.2f}s, Total epoch time: {epoch_time:.2f}s')

    # 计算总训练时间
    total_training_time = sum(training_times)
    avg_training_time = np.mean(training_times)

    print(f"\n最佳测试准确率: {best_test_acc:.4f}")
    print(f"总训练时间: {total_training_time:.2f}秒 ({total_training_time/60:.2f}分钟)")
    print(f"平均每轮训练时间: {avg_training_time:.2f}秒")

    # 保存最佳结果汇总到评估文件
    with open(evaluate_file, 'a', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("训练完成 - 最佳结果汇总\n")
        f.write("="*80 + "\n")
        f.write(f"目标域OA: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)\n")
        f.write(
            f"目标域Kappa: {best_test_kappa:.4f} ({best_test_kappa*100:.2f}%)\n")
        f.write(f"各类TPR: {[f'{tpr:.4f}' for tpr in best_test_tpr]}\n")
        f.write(f"平均每轮训练时间: {avg_training_time:.4f}秒\n")
        f.write(
            f"总训练时间: {total_training_time:.2f}秒 ({total_training_time/60:.2f}分钟)\n")
        f.write(f"训练轮数: {args.num_epochs}\n")
        if best_test_cm is not None:
            f.write(f"最佳混淆矩阵:\n")
            for i, row in enumerate(best_test_cm):
                f.write(f"  类别{i}: {row}\n")
        f.write("="*80 + "\n")

    print(f"最佳结果汇总已保存到: {evaluate_file}")

    # 保存结果
    results = {
        'best_test_accuracy': best_test_acc,
        'total_training_time': total_training_time,
        'avg_training_time': avg_training_time,
        'training_times': training_times,
        'args': vars(args)
    }

    import json
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if use_tensorboard:
        writer.close()
    print(f"\n训练完成！结果保存在: {save_dir}")


if __name__ == '__main__':
    main()
