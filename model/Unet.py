import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class VGGUNet(nn.Module):
    def __init__(self, num_classes):
        super(VGGUNet, self).__init__()

        # 使用预训练的 VGG16 作为编码器
        vgg = vgg16(pretrained=True)
        features = list(vgg.features.children())

        # 定义下采样路径
        self.enc1 = nn.Sequential(*features[:5])  # 第一阶段（Conv1）
        self.enc2 = nn.Sequential(*features[5:10])  # 第二阶段（Conv2）
        self.enc3 = nn.Sequential(*features[10:17])  # 第三阶段（Conv3）
        self.enc4 = nn.Sequential(*features[17:24])  # 第四阶段（Conv4）

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 定义上采样路径
        self.up4 = self._up_block(1024 + 512, 512)
        self.up3 = self._up_block(512 + 256, 256)
        self.up2 = self._up_block(256 + 128, 128)
        self.up1 = self._up_block(128 + 64, 64)

        # 最终输出层
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # 编码路径
        enc1 = self.enc1(x)  # 第一层
        enc2 = self.enc2(F.max_pool2d(enc1, 2))  # 第二层
        enc3 = self.enc3(F.max_pool2d(enc2, 2))  # 第三层
        enc4 = self.enc4(F.max_pool2d(enc3, 2))  # 第四层

        # 瓶颈层
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # 解码路径
        dec4 = self.up4(torch.cat((self._upsample(bottleneck, enc4), enc4), dim=1))
        dec3 = self.up3(torch.cat((self._upsample(dec4, enc3), enc3), dim=1))
        dec2 = self.up2(torch.cat((self._upsample(dec3, enc2), enc2), dim=1))
        dec1 = self.up1(torch.cat((self._upsample(dec2, enc1), enc1), dim=1))

        # 最终输出
        final_output = self.final_conv(dec1)

        return final_output

    def _upsample(self, x, target):
        # 上采样到目标张量的空间尺寸
        return F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)


# 测试模型
if __name__ == "__main__":
    num_classes = 5  # 假设分割任务有 5 个类别
    model = VGGUNet(num_classes=num_classes)

    # 模拟输入
    inputs = torch.randn(1, 3, 640, 640)  # 1 张 640x640 的 RGB 图像
    outputs = model(inputs)
    print("Output shape:", outputs.shape)  # 输出形状应为 [1, num_classes, 640, 640]
