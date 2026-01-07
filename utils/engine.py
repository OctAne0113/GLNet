from math import ceil
import torch
import numpy as np
import math
import sys
import utils.lr_sched as lr_sched
from utils.cal_tools import IouCal, AverageMeter, ProgressMeter
import torch.nn.functional as F
# from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from model.PolygonLoss import polygon_loss_with_gt
import cv2
from matplotlib import pyplot as plt


def train_model(args, epoch, model, train_loader, criterion, optimizer, loss_scaler, device):
    model.train()
    train_main_loss = AverageMeter('Train Main Loss', ':.5')
    lr = AverageMeter('lr', ':.5')
    L = len(train_loader)
    curr_iter = epoch * L
    record = [lr, train_main_loss]
    if args.model_name == "PSPNet":
        train_aux_loss = AverageMeter('Train Aux Loss', ':.5')
        record.append(train_aux_loss)
    progress = ProgressMeter(L, record, prefix="Epoch: [{}]".format(epoch))
    accum_iter = args.accum_iter

    for data_iter_step, data in enumerate(train_loader):
        optimizer.param_groups[0]['lr'] = args.lr * (1 - float(curr_iter) / (args.num_epoch * L)) ** args.lr_decay
        # if data_iter_step % accum_iter == 0:
        #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, args)

        inputs = data["images"].to(device, dtype=torch.float32)
        mask = data["masks"].to(device, dtype=torch.int64)

        with torch.cuda.amp.autocast():
            if args.model_name == "PSPNet":
                outputs, aux = model(inputs)
                main_loss = criterion(outputs, mask)
                aux_loss = criterion(aux, mask)
                loss = main_loss + 0.4 * aux_loss  # ori0.4
            else:
                outputs = model(inputs)  # Segmenter and PSPNet
                # outputs, Unetmask, Refmask = model(inputs)  # Ours
                # # 初始化损失函数
                # dice_loss = DiceLoss(mode='multiclass',ignore_index=None,)  # 多分类模式
                # focal_loss = FocalLoss(mode='multiclass',alpha=0.25,gamma=2.0,reduction='mean',)  # 多分类模式
                # DICE_loss = dice_loss(outputs, mask)
                # Focal_loss = focal_loss(outputs, mask)
                #
                main_loss = criterion(outputs, mask)
                # Unetloss = criterion(Unetmask, mask)  # Ours
                # refloss = criterion(Refmask, mask)  # Ours

                # class_ids = [1,2,3,4,5,6]  # 窗户和门
                #
                # #初始化 IoU 计算模块
                # iou_loss = IOUForClasses(class_ids=class_ids)
                # iou_value_ori = iou_loss(outputs, mask)
                # iou_value_unet = iou_loss(Unetmask, mask)
                # iou_value_ref = iou_loss(Refmask, mask)
                #
                # preds = torch.argmax(outputs, dim=1)
                # shape_loss = polygon_loss_with_gt(preds, mask, target_class=6, device=device)
                # scaled_shape_loss = shape_loss / 1000

                loss = main_loss  # Segmenter and PSPNet
                # loss = main_loss * 0.5 + Unetloss * 0.3 + refloss * 0.3  # Ours

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(2)

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        train_main_loss.update(main_loss.item())
        if args.model_name == "PSPNet":
            train_aux_loss.update(aux_loss.item())
        lr.update(optimizer.param_groups[0]['lr'])

        curr_iter += 1

        if data_iter_step % args.print_freq == 0:
            progress.display(data_iter_step)


def evaluation(args, best_record, epoch, model, model_without_ddp, val_loader, criterion, device):
    model.eval()
    val_loss = AverageMeter('Val Main Loss', ':.4')
    progress = ProgressMeter(len(val_loader), [val_loss], prefix="Epoch: [{}]".format(epoch))
    iou = IouCal(args)
    for i_batch, data in enumerate(val_loader):
        inputs = data["images"].to(device, dtype=torch.float32)
        mask = data["masks"].to(device, dtype=torch.int64)

        pred, full_pred = inference_sliding(args, model, inputs)
        iou.evaluate(pred, mask)
        val_loss.update(criterion(full_pred, mask).item())

        if i_batch % args.print_freq == 0:
            progress.display(i_batch)

    acc, acc_cls, mean_iou = iou.iou_demo()

    if mean_iou > best_record['mean_iou']:
        best_record['val_loss'] = val_loss.avg
        best_record['epoch'] = epoch
        best_record['acc'] = acc
        best_record['acc_cls'] = acc_cls
        best_record['mean_iou'] = mean_iou
        if args.output_dir:
            if args.model_name == "Segmenter":
                save_name = args.model_name + "_" + args.encoder
            else:
                save_name = args.model_name
            torch.save(model_without_ddp.state_dict(), args.output_dir + args.dataset + "_" + save_name + ".pt")

    print('-----------------------------------------------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iou %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iou))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iou %.5f], ---- [epoch %d], '
          % (best_record['val_loss'], best_record['acc'],
             best_record['acc_cls'], best_record['mean_iou'], best_record['epoch']))

    print('-----------------------------------------------------------------------------------------------------------')


@torch.no_grad()
def tta_inference(args, model, image):
    """
    测试时增强推理，包括原图和水平翻转图。
    """
    # 原图推理
    preds_normal, full_probs_normal = inference_sliding(args, model, image)

    # 水平翻转推理
    flipped_image = torch.flip(image, dims=[3])  # 水平翻转
    preds_flipped, full_probs_flipped = inference_sliding(args, model, flipped_image)
    full_probs_flipped = torch.flip(full_probs_flipped, dims=[3])  # 翻转回原始方向

    # 融合结果
    full_probs = (full_probs_normal + full_probs_flipped) / 2
    _, preds = torch.max(full_probs, dim=1)

    return preds, full_probs


import os

from data.facade import PolygonTrans
import os.path as path


def save_single(image, save_path):
    """
    保存单张图像到指定路径
    """
    image = np.array(image, dtype=np.uint8)
    plt.imsave(save_path, image)  # 使用 plt.imsave 直接保存图像


@torch.no_grad()
def evaluation_none_training_ours(args, model, val_loader, device):
    model.eval()

    image_folder = r"./data/test/images"  # 传入你的测试图像文件夹路径
    os.makedirs('demo/', exist_ok=True)  # 确保有保存目录

    # 获取文件夹中的所有图片文件名
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]  # 根据需要添加其他格式

    image_index = 0  # 用于从 image_files 中按顺序提取图像

    for i_batch, data in enumerate(val_loader):
        if i_batch % 5 == 0:
            print(f"{i_batch}/{len(val_loader)}")

        inputs = data["images"].to(device, dtype=torch.float32)

        # 获取预测结果
        pred1, pred2, pred3 = inference_sliding_pro(args, model, inputs)

        # 获取当前批次中的每张图像文件名，并保存对应的分割图
        batch_size = len(pred1)  # 确保批次大小是4

        for i in range(batch_size):  # 遍历批次中的每张图像
            img_name = image_files[image_index]  # 获取当前批次对应的图像文件名

            # # 保存分割图1
            # color_img1 = PolygonTrans().id2trainId(torch.squeeze(pred1[i], dim=0).cpu().numpy(), select=None)
            # save_name = f"demo/{img_name.replace('.jpg', '_segmentation1.png').replace('.png', '_segmentation1.png')}"
            # save_single(color_img1, save_name)
            #
            # # 保存分割图2
            # color_img2 = PolygonTrans().id2trainId(torch.squeeze(pred2[i], dim=0).cpu().numpy(), select=None)
            # save_name = f"demo/{img_name.replace('.jpg', '_segmentation2.png').replace('.png', '_segmentation2.png')}"
            # save_single(color_img2, save_name)

            # 保存分割图3
            color_img3 = PolygonTrans().id2trainId(torch.squeeze(pred3[i], dim=0).cpu().numpy(), select=None)
            save_name = f"demo/{img_name.replace('.jpg', '_segmentation3.png').replace('.png', '_segmentation3.png')}"
            save_single(color_img3, save_name)

            image_index += 1  # 更新图像索引

    print("Evaluation complete.")


def evaluation_none_training(args, model, val_loader, device):
    model.eval()

    image_folder = r"./data/test_cfp/images"  # 传入你的测试图像文件夹路径

    # 获取文件夹中的所有图片文件名
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]  # 根据需要添加其他格式

    image_index = 0  # 用于从 image_files 中按顺序提取图像

    for i_batch, data in enumerate(val_loader):
        if i_batch % 5 == 0:
            print(f"{i_batch}/{len(val_loader)}")

        inputs = data["images"].to(device, dtype=torch.float32)

        # 获取预测结果
        pred1, _ = inference_sliding(args, model, inputs)

        # 获取当前批次中的每张图像文件名，并保存对应的分割图
        batch_size = len(pred1)  # 确保批次大小是4

        for i in range(batch_size):  # 遍历批次中的每张图像
            img_name = image_files[image_index]  # 获取当前批次对应的图像文件名

            # 保存分割图1
            color_img1 = PolygonTrans().id2trainId(torch.squeeze(pred1[i], dim=0).cpu().numpy(), select=None)
            save_name = f"demo/{img_name.replace('.jpg', '_segmentation1.png').replace('.png', '_segmentation1.png')}"
            save_single(color_img1, save_name)

            image_index += 1  # 更新图像索引

    print("Evaluation complete.")


@torch.no_grad()
def evaluation_none_training_ori(args, model, val_loader, device):
    model.eval()
    iou = IouCal(args)
    for i_batch, data in enumerate(val_loader):
        if i_batch % 5 == 0:
            print(str(i_batch) + "/" + str(len(val_loader)))
        inputs = data["images"].to(device, dtype=torch.float32)
        mask = data["masks"].to(device, dtype=torch.int64)

        # image_folder = "./data/classall//translated_data_test_hznu/images"  # 传入你的测试图像文件夹路径
        # os.makedirs('demo/', exist_ok=True)  # 确保有保存目录
        #
        # # 获取文件夹中的所有图片文件名
        # image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]  # 根据需要添加其他格式
        #
        # image_index = 0  # 用于从 image_files 中按顺序提取图像
        #
        # for i_batch, data in enumerate(val_loader):
        #     if i_batch % 5 == 0:
        #         print(f"{i_batch}/{len(val_loader)}")
        #
        #     inputs = data["images"].to(device, dtype=torch.float32)
        #
        #     # 获取预测结果
        #     preds_tta, _ = inference_sliding(args, model, inputs)
        #
        #     # 获取当前批次中的每张图像文件名，并保存对应的分割图
        #     batch_size = len(preds_tta)  # 确保批次大小是4
        #
        #     for i in range(batch_size):  # 遍历批次中的每张图像
        #         img_name = image_files[image_index]  # 获取当前批次对应的图像文件名
        #         # 保存分割图3
        #         color_img3 = PolygonTrans().id2trainId(torch.squeeze(preds_tta[i], dim=0).cpu().numpy(), select=None)
        #         save_name = f"demo/{img_name.replace('.jpg', '_segmentation3.png').replace('.png', '_segmentation3.png')}"
        #         save_single(color_img3, save_name)
        #
        #         image_index += 1  # 更新图像索引

        # 测试时增强推理
        # preds_tta, full_probs_tta = tta_inference(args, model, inputs)

        # iou.evaluate(preds_tta, mask)

        pred, full_pred = inference_sliding(args, model, inputs)
        iou.evaluate(pred, mask)

    acc, acc_cls, mean_iou = iou.iou_demo()
    print("acc:", acc)
    print("acc_cls", acc_cls)
    print("mean_iou", mean_iou)


def fit_and_fill_polygons(preds, window_class, epsilon_factor=0.02):
    """
    对检测为窗户的区域进行多边形拟合，并将拟合区域设置为窗户类。

    Args:
        preds (torch.Tensor): 模型预测的类别结果，[batch_size, height, width]。
        window_class (int): 窗户类别的索引。
        epsilon_factor (float): 多边形拟合的精度因子，越小拟合越精确。

    Returns:
        torch.Tensor: 更新后的预测结果。
    """
    batch_size, height, width = preds.shape
    updated_preds = preds.clone()  # 创建副本以更新
    saved = False
    save_path = r"./evaluation/show.jpg"
    for b in range(batch_size):
        # 获取窗户的二值掩码
        window_mask = (preds[b] == window_class).cpu().numpy().astype(np.uint8)

        # # 形态学闭操作，平整边界
        # kernel = np.ones((3, 3), np.uint8)
        # Bi_mask = cv2.morphologyEx(window_mask, cv2.MORPH_CLOSE, kernel)

        # 提取轮廓 —— OpenCV 2.x返回：修改后的图像、轮廓列表和轮廓层级信息；OpenCV 3.x及更高返回：轮廓列表和轮廓层级信息
        # _, contours, _ = cv2.findContours(window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个空白图像，用于存放拟合后的多边形
        refined_mask = np.zeros_like(window_mask)

        for contour in contours:
            # 多边形拟合
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 填充拟合的多边形
            cv2.fillPoly(refined_mask, [approx], 1)
        if not saved:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(window_mask, cmap='gray')
            axes[0].set_title("Before Update")
            axes[0].axis("off")

            axes[1].imshow(refined_mask, cmap='gray')
            axes[1].set_title("After Update")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            saved = True
        # 更新预测结果，将拟合的区域标记为窗户类
        updated_preds[b][torch.tensor(refined_mask == 1)] = window_class

    return updated_preds


@torch.no_grad()
def inference_sliding(args, model, image):
    image_size = image.size()
    stride = int(math.ceil(args.crop_size[0] * args.stride_rate))
    tile_rows = ceil((image_size[2] - args.crop_size[0]) / stride) + 1
    tile_cols = ceil((image_size[3] - args.crop_size[1]) / stride) + 1
    b = image_size[0]

    full_probs = torch.from_numpy(np.zeros((b, args.num_classes, image_size[2], image_size[3]))).to(args.device)
    count_predictions = torch.from_numpy(np.zeros((b, args.num_classes, image_size[2], image_size[3]))).to(args.device)

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = x1 + args.crop_size[0]
            y2 = y1 + args.crop_size[1]
            if row == tile_rows - 1:
                y2 = image_size[2]
                y1 = image_size[2] - args.crop_size[1]
            if col == tile_cols - 1:
                x2 = image_size[3]
                x1 = image_size[3] - args.crop_size[0]

            img = image[:, :, y1:y2, x1:x2]

            with torch.set_grad_enabled(False):
                # _, _, padded_prediction = model(img)  # Ours
                padded_prediction = model(img)  # Segmenter and PSPNet
            count_predictions[:, :, y1:y2, x1:x2] += 1
            full_probs[:, :, y1:y2, x1:x2] += padded_prediction  # accumulate the predictions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    _, preds = torch.max(full_probs, 1)
    # window_class = 6  # RTFP
    # preds = fit_and_fill_polygons(preds, window_class, epsilon_factor=0.02)  # RTFP
    return preds, full_probs


def inference_sliding_pro(args, model, image):
    image_size = image.size()
    stride = int(math.ceil(args.crop_size[0] * args.stride_rate))
    tile_rows = ceil((image_size[2] - args.crop_size[0]) / stride) + 1
    tile_cols = ceil((image_size[3] - args.crop_size[1]) / stride) + 1
    b = image_size[0]

    full_probs = torch.from_numpy(np.zeros((b, args.num_classes, image_size[2], image_size[3]))).to(args.device)
    count_predictions = torch.from_numpy(np.zeros((b, args.num_classes, image_size[2], image_size[3]))).to(args.device)

    full_probs1 = full_probs
    full_probs2 = full_probs
    full_probs3 = full_probs

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = x1 + args.crop_size[0]
            y2 = y1 + args.crop_size[1]
            if row == tile_rows - 1:
                y2 = image_size[2]
                y1 = image_size[2] - args.crop_size[1]
            if col == tile_cols - 1:
                x2 = image_size[3]
                x1 = image_size[3] - args.crop_size[0]

            img = image[:, :, y1:y2, x1:x2]

            with torch.set_grad_enabled(False):
                padded_prediction1, padded_prediction2, padded_prediction3 = model(img)
                # padded_prediction = model(img)
            count_predictions[:, :, y1:y2, x1:x2] += 1

            full_probs1[:, :, y1:y2, x1:x2] += padded_prediction1  # accumulate the predictions
            full_probs2[:, :, y1:y2, x1:x2] += padded_prediction2  # accumulate the predictions
            full_probs3[:, :, y1:y2, x1:x2] += padded_prediction3  # accumulate the predictions

    # average the predictions in the overlapping regions
    full_probs1 /= count_predictions
    _, preds1 = torch.max(full_probs1, 1)
    full_probs2 /= count_predictions
    _, preds2 = torch.max(full_probs2, 1)
    full_probs3 /= count_predictions
    _, preds3 = torch.max(full_probs3, 1)

    # window_class = 6  # RTFP
    # preds3 = fit_and_fill_polygons(preds3, window_class, epsilon_factor=0.02)  # RTFP
    return preds1, preds2, preds3
