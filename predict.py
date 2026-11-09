import argparse
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from train import build_model


def main():
    parser = argparse.ArgumentParser(description='马铃薯叶片单张图片分类演示')
    parser.add_argument('--image', required=True, help='待识别图片路径')
    parser.add_argument('--checkpoint', default='results/正式_轻量化与激活/最佳模型.pth', help='训练得到的模型文件')
    args = parser.parse_args()
    image_path = Path(args.image)
    model_path = Path(args.checkpoint)
    if not image_path.is_file():
        parser.error(f'图片不存在：{image_path}')
    if not model_path.is_file():
        parser.error(f'模型文件不存在：{model_path}')

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    config = checkpoint['args']
    names = checkpoint['classes']
    net = build_model(config['model'], len(names), pretrained=False)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    transform = transforms.Compose([
        transforms.Resize((config['size'], config['size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    try:
        with Image.open(image_path) as image:
            tensor = transform(image.convert('RGB')).unsqueeze(0)
    except (UnidentifiedImageError, OSError) as exc:
        parser.error(f'无法读取图片：{exc}')
    with torch.inference_mode():
        scores = torch.softmax(net(tensor), dim=1)[0]
    translated = {
        'Potato___Early_blight': '马铃薯早疫病',
        'Potato___Late_blight': '马铃薯晚疫病',
        'Potato___healthy': '马铃薯健康叶片',
    }
    best = int(scores.argmax())
    print(f'图片：{image_path}')
    print(f'预测类别：{translated.get(names[best], names[best])}')
    print('各类别模型分数：')
    for name, score in zip(names, scores.tolist()):
        print(f'  {translated.get(name, name)}：{score * 100:.2f}%')
    print('提示：模型只能在已训练类别中选择；分数不代表诊断可靠性。')


if __name__ == '__main__':
    main()
