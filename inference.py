import argparse
import torch.backends.cudnn as cudnn
import torch
from data.get_data_set import get_data
from PIL import Image
from data.loader_tools import get_standard_transformations
import utils.misc as misc
from configs import get_args_parser
from model.get_model import model_generation
from utils.engine import inference_sliding
from data.facade import PolygonTrans
import matplotlib.pyplot as plt
import numpy as np
import os


def show_single(image, original_size, location=None, save=False, name=None):
    # show single image
    image = np.array(image, dtype=np.uint8)

    # Rescale the image back to the original size
    image = Image.fromarray(image)
    image = image.resize(original_size, Image.BILINEAR)

    fig = plt.figure()
    plt.imshow(image)
    plt.gca().set_aspect('auto')  # 或者 'equal'，确保比例是自动的或一致的
    fig.set_size_inches(original_size[0] / 100.0, original_size[1] / 100.0)  # 输出 width*height 像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    if save:
        plt.savefig(name, dpi=100, bbox_inches='tight', pad_inches=0)
        print(fig.get_dpi())  # 查看当前 DPI 设置
    plt.show()


def main():
    # distribution
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    cudnn.benchmark = True
    train_set, val_set, ignore_index = get_data(args)
    model = model_generation(args)
    model.to(device)

    if args.model_name == "Segmenter":
        save_name = args.model_name + "_" + args.encoder
    else:
        save_name = args.model_name

    checkpoint = torch.load(args.output_dir + args.dataset + "_" + save_name + ".pt", map_location="cuda:0")
    print(args.output_dir + args.dataset + "_" + save_name + ".pt")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    standard_transformations = get_standard_transformations()

    # Read image and record the original size
    img = Image.open(img_path).convert('RGB')
    original_size = img.size  # (width, height)

    # Resize image to the model's required size
    img = img.resize((args.setting_size[1], args.setting_size[0]), Image.BILINEAR)
    img = standard_transformations(img).to(device, dtype=torch.float32)

    # Perform inference
    pred, full_pred = inference_sliding(args, model, img.unsqueeze(0))

    # Convert prediction to color image and show
    color_img = PolygonTrans().id2trainId(torch.squeeze(pred, dim=0).cpu().detach().numpy(), select=None)

    show_single(color_img, original_size, save=True, name="single_inf/color_mask.png")



if __name__ == '__main__':
    os.makedirs('./single_inf', exist_ok=True)
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    img_path = r'./c_0211.png'
    main()
