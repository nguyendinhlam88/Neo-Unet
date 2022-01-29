from tqdm import tqdm
from loss import *
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import math

def reverse_transform(inp):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp

def show_result(input, y_true, y_pred):
    input = reverse_transform(input.numpy().transpose(1, 2, 0))
    y_true = y_true.numpy().squeeze(0)
    y_pred = torch.argmax(y_pred.permute(1, 2, 0).contiguous(), dim=-1).numpy()
    plt.figure(figsize=(15, 8))
    plt.axis('off')
    plt.subplot(1,3,1)
    plt.imshow(input)
    plt.title("Input")
    plt.subplot(1, 3, 2)
    plt.imshow(input)
    plt.imshow(y_true, alpha=0.3, cmap='nipy_spectral')
    plt.title("target")
    plt.subplot(1, 3, 3)
    plt.imshow(input)
    plt.imshow(y_pred, alpha=0.3, cmap='nipy_spectral')
    plt.title("prediction")
    plt.show()

def train_fn(model, dataloader_dict, optimizer, nEpoch, BATCH_SIZE):
    # all_train_loss = checkpoint['train_loss']
    # all_val_loss = checkpoint['val_loss']
    # all_train_dice = checkpoint['train_dice']
    # all_val_dice = checkpoint['val_dice']
    all_train_loss, all_val_loss, all_train_dice, all_train_iou, all_val_dice, all_val_iou = [], [], [], [], [], []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(0, nEpoch):
        print("Epoch {}/{}".format(epoch+1, nEpoch))
        train_dice_score, train_iou_score = [], []
        val_dice_score, val_iou_score = [], []
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        for phase in ["train", "val"]:
            n_batch = 0
            if phase == "train":
                model.train()
                print("\n###################################")
                print("[INFO] Training network")
                print("Train on {} samples".format(len(dataloader_dict["train"]) * BATCH_SIZE))
            else:
                model.eval()
                print("\n###################################")
                print("[INFO] Evaluation network")
                print("Eval on {} samples".format(len(dataloader_dict["val"]) * BATCH_SIZE))

            for inputs, targets in tqdm(dataloader_dict[phase]): 
                inputs = inputs.to(device).float()
                targets = targets.to(device)

                with torch.set_grad_enabled(phase=="train"):
                    y_pred = model(inputs)
                    loss = multiclass_loss(y_pred, targets)
                    score = iou_dice_score(y_pred.data.cpu(), targets.data.cpu())

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epoch_train_loss += loss.item()

                        # if n_batch == 99:
                        with torch.no_grad():
                            for i in range(2):
                                show_result(inputs[i].cpu(), targets[i].cpu(), y_pred[i].cpu())

                        #####################
                        train_iou_score.append(score[0])
                        train_dice_score.append(score[1])
                        #####################
                    
                    else:
                        epoch_val_loss += loss.item()

                        #####################
                        val_iou_score.append(score[0])
                        val_dice_score.append(score[1])
                        #####################
                n_batch += 1

            if phase == "train":
                print("Training - loss: {:.4f}".format(epoch_train_loss / n_batch))
                print("Average Dice Score: {:.4f}".format(sum(train_dice_score)/ n_batch))
                print("Average IOU Score: {:.4f}".format(sum(train_iou_score) / n_batch))
                print("###################################")

                #####################
                all_train_loss.append(epoch_train_loss / n_batch)
                all_train_dice.append(sum(train_dice_score).item() / n_batch)
                all_train_iou.append(sum(train_iou_score).item() / n_batch)
                #####################
            else:
                print("Evaluation - loss: {:.4f}".format(epoch_val_loss / n_batch))
                print("Average Dice Score: {:.4f}".format(sum(val_dice_score) / n_batch))
                print("Average IOU Score: {:.4f}".format(sum(val_iou_score) / n_batch))
                print("###################################")

                #####################
                all_val_loss.append(epoch_val_loss / n_batch)
                all_val_dice.append(sum(val_dice_score).item() / n_batch)
                all_val_iou.append(sum(val_iou_score).item() / n_batch)
                #####################

        if ((epoch + 1) >= 10) and (((epoch + 1) % 5) == 0):
            torch.save({
                'epoch': epoch+1,
                'train_loss': all_train_loss,
                'val_loss': all_val_loss,
                'train_dice': all_train_dice,
                'val_dice': all_val_dice,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "/content/drive/MyDrive/Neo-Unet/out/weights/neounet_weights_{}.pth.tar".format(epoch+1))
            ##################### 
        if ((epoch+1) >= 10) and (((epoch+1) % 5) == 0):
            range_x = np.arange(epoch+1)
            plt.figure()
            plt.plot(range_x, all_train_loss, label="train_loss")
            plt.plot(range_x, all_val_loss, label="val_loss")
            plt.title("LOSS")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.grid()
            plt.legend()
            plt.savefig("/content/drive/MyDrive/Neo-Unet/out/visualize/loss_{}.png".format(epoch+1))

            plt.figure()
            plt.plot(range_x, all_train_dice, label="train_dice")
            plt.plot(range_x, all_val_dice, label="val_dice")
            plt.title("DICE")
            plt.xlabel("Epoch #")
            plt.ylabel("Dice score")
            plt.grid()
            plt.legend()
            plt.savefig("/content/drive/MyDrive/Neo-Unet/out/visualize/dice_score_{}.png".format(epoch+1))

            plt.figure()
            plt.plot(range_x, all_train_iou, label="train_iou")
            plt.plot(range_x, all_val_iou, label="val_iou")
            plt.title("IOU")
            plt.xlabel("Epoch #")
            plt.ylabel("Iou score")
            plt.grid()
            plt.legend()
            plt.savefig("/content/drive/MyDrive/Neo-Unet/out/visualize/iou_score_{}.png".format(epoch+1))
        #####################