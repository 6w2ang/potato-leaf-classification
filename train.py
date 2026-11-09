import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(kind, classes, pretrained):
    weights = models.VGG16_BN_Weights.DEFAULT if pretrained else None
    net = models.vgg16_bn(weights=weights)
    layers = list(net.features.children())
    if kind != 'baseline':
        conv_indices = [i for i, layer in enumerate(layers) if isinstance(layer, nn.Conv2d)]
        # 仅替换最后两个卷积阶段，首层和前期特征提取保持不变。
        for i in conv_indices[7:]:
            old = layers[i]
            layers[i] = nn.Sequential(
                nn.Conv2d(old.in_channels, old.in_channels, 3, padding=1, groups=old.in_channels, bias=False),
                nn.Conv2d(old.in_channels, old.out_channels, 1, bias=True),
            )
    if kind == 'light_gelu':
        layers = [nn.GELU() if isinstance(layer, nn.ReLU) else layer for layer in layers]
    net.features = nn.Sequential(*layers)
    net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    net.classifier = nn.Linear(512, classes)
    return net


def loaders(args, output):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_tf = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    eval_tf = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor(), normalize])
    if args.smoke:
        train = datasets.FakeData(size=16, image_size=(3, args.size, args.size), num_classes=3, transform=train_tf, random_offset=0)
        val = datasets.FakeData(size=8, image_size=(3, args.size, args.size), num_classes=3, transform=eval_tf, random_offset=1000)
        test = datasets.FakeData(size=8, image_size=(3, args.size, args.size), num_classes=3, transform=eval_tf, random_offset=2000)
        names = ['类别一', '类别二', '类别三']
        counts = [len(train), len(val), len(test)]
    else:
        root = Path(args.data)
        raw = datasets.ImageFolder(root)
        augmented = datasets.ImageFolder(root, transform=train_tf)
        evaluated = datasets.ImageFolder(root, transform=eval_tf)
        selected = sorted(args.classes)
        missing = set(selected) - set(raw.classes)
        if missing or len(selected) < 2 or len(selected) != len(set(selected)):
            raise ValueError(f'类别至少两个且不可重复，未找到的类别：{sorted(missing)}')
        for dataset in (raw, augmented, evaluated):
            samples = [(path, selected.index(dataset.classes[label])) for path, label in dataset.samples if dataset.classes[label] in selected]
            dataset.samples = samples
            dataset.imgs = samples
            dataset.targets = [label for _, label in samples]
            dataset.classes = selected
            dataset.class_to_idx = {name: i for i, name in enumerate(selected)}
        names = raw.classes
        mapping_path = Path(args.leaf_map)
        mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
        grouped = {i: {} for i in range(len(names))}
        excluded = {name: 0 for name in names}
        for i, (path, label) in enumerate(raw.samples):
            identifier = Path(path).stem.replace('_final_masked', '').split('___')[-1].split('copy')[0].strip().lower()
            matches = [x for x in mapping.get(identifier, []) if x.split(':::')[0] == names[label]]
            if len(matches) != 1:
                excluded[names[label]] += 1
                continue
            grouped[label].setdefault(matches[0], []).append(i)
        generator = random.Random(args.seed)
        indices = [[], [], []]
        assignments = []
        details = {}
        for label, groups in grouped.items():
            keys = sorted(groups)
            if len(keys) < 3:
                raise ValueError(f'类别 {names[label]} 可确认的叶片组不足三个，无法划分：{len(keys)}')
            generator.shuffle(keys)
            n_test = max(1, round(len(keys) * 0.15))
            n_val = max(1, round(len(keys) * 0.15))
            partitions = [keys[n_test + n_val:], keys[n_test:n_test + n_val], keys[:n_test]]
            class_counts = []
            for part, keys_part in enumerate(partitions):
                class_counts.append(sum(len(groups[key]) for key in keys_part))
                for key in keys_part:
                    for i in groups[key]:
                        indices[part].append(i)
                        assignments.append((Path(raw.samples[i][0]).relative_to(root).as_posix(), names[label], key, ['训练', '验证', '测试'][part]))
            details[names[label]] = {'叶片组数': len(keys), '训练验证测试图片数': class_counts, '未匹配排除图片数': excluded[names[label]]}
        with (output / '数据划分.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['相对路径', '类别', '叶片组', '数据集'])
            writer.writerows(assignments)
        split_info = {'方式': '按类别对官方映射的叶片组随机划分，未明确匹配的图片排除', '随机种子': args.seed, '映射文件摘要': hashlib.sha256(mapping_path.read_bytes()).hexdigest(), '类别统计': details, '排除图片总数': sum(excluded.values())}
        (output / '数据划分统计.json').write_text(json.dumps(split_info, ensure_ascii=False, indent=2), encoding='utf-8')
        train, val, test = (Subset(augmented, indices[0]), Subset(evaluated, indices[1]), Subset(evaluated, indices[2]))
        counts = [len(x) for x in indices]
    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    drop_last = len(train) % args.batch_size == 1
    return DataLoader(train, shuffle=True, drop_last=drop_last, **kwargs), DataLoader(val, shuffle=False, **kwargs), DataLoader(test, shuffle=False, **kwargs), names, counts


def macs(model, size, device):
    total = [0]
    hooks = []
    def hook(layer, inputs, output):
        if isinstance(layer, nn.Conv2d):
            total[0] += output.numel() * (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
        elif isinstance(layer, nn.Linear):
            total[0] += output.numel() * layer.in_features
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size, device=device))
    for h in hooks:
        h.remove()
    return total[0]


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_fn = nn.CrossEntropyLoss()
    loss_sum, correct, count = 0.0, 0, 0
    truth, predicted = [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            pred = logits.argmax(1)
            loss_sum += loss.item() * y.size(0)
            correct += (pred == y).sum().item()
            count += y.size(0)
            truth.extend(y.cpu().tolist())
            predicted.extend(pred.cpu().tolist())
    return loss_sum / count, correct / count, truth, predicted


def main():
    p = argparse.ArgumentParser(description='植物病害分类实验')
    p.add_argument('--data', default='data/plantvillage')
    p.add_argument('--leaf-map', default='data/leaf-map.json')
    p.add_argument('--classes', nargs='+', default=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'])
    p.add_argument('--model', choices=['baseline', 'light', 'light_gelu'], default='baseline')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--output', default='results/baseline')
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--check-data', action='store_true')
    args = p.parse_args()
    if args.size < 32 or args.epochs < 1 or args.batch_size < 2:
        p.error('图像尺寸至少为32，训练轮数至少为1，批次大小至少为2')
    seed_all(args.seed)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'输出目录非空，请指定新目录以避免覆盖：{out}')
    out.mkdir(parents=True, exist_ok=True)
    train, val, test, names, counts = loaders(args, out)
    print(f'训练、验证、测试图片数量：{counts}', flush=True)
    if args.check_data:
        print('数据划分检查完成。')
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.model, len(names), args.pretrained).to(device)
    params = sum(p.numel() for p in model.parameters())
    operations = macs(model, args.size, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = -1.0
    start = time.perf_counter()
    with (out / '训练记录.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['轮次', '训练损失', '训练准确率', '验证损失', '验证准确率'])
        for epoch in range(1, args.epochs + 1):
            tl, ta, _, _ = run_epoch(model, train, device, opt)
            vl, va, _, _ = run_epoch(model, val, device)
            writer.writerow([epoch, tl, ta, vl, va])
            f.flush()
            print(f'第{epoch}轮 训练损失={tl:.4f} 训练准确率={ta:.4f} 验证损失={vl:.4f} 验证准确率={va:.4f}', flush=True)
            if va > best:
                best = va
                torch.save({'model': model.state_dict(), 'classes': names, 'args': vars(args), 'epoch': epoch}, out / '最佳模型.pth')
    elapsed = time.perf_counter() - start
    checkpoint = torch.load(out / '最佳模型.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    loss, accuracy, truth, pred = run_epoch(model, test, device)
    matrix = np.zeros((len(names), len(names)), dtype=int)
    for a, b in zip(truth, pred):
        matrix[a, b] += 1
    per_class = []
    for i, name in enumerate(names):
        tp = int(matrix[i, i])
        precision = tp / int(matrix[:, i].sum()) if matrix[:, i].sum() else 0.0
        recall = tp / int(matrix[i, :].sum()) if matrix[i, :].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({'类别': name, '精确率': precision, '召回率': recall, 'F1': f1})
    result = {'配置': vars(args), '设备': str(device), '数据量_训练验证测试': counts, '类别': names, '参数量': params, '乘加次数': operations, '最佳轮次': checkpoint['epoch'], '最佳验证准确率': best, '测试损失': loss, '测试准确率': accuracy, '宏平均F1': sum(x['F1'] for x in per_class) / len(per_class), '训练秒数': elapsed, '各类别指标': per_class, '混淆矩阵': matrix.tolist()}
    (out / '实验结果.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
