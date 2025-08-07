import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import time
from model import YOLOv1
from loss import YoloLoss
from dataset import YOLODataset
from utils import *
import random
device = torch.device('cuda')


class ComposeTransform:
    def __init__(self, resize=448):
        self.transforms = T.Compose([
            T.Resize((resize, resize)),
            T.ToTensor(),
        ])

    def __call__(self, image, bboxes):
        image = self.transforms(image)
        return image, bboxes


def main():
    transform = ComposeTransform(resize=448)

    # with open("train.csv", "r") as f:
    #     all_lines = f.readlines()
    # subset_size = len(all_lines) // 2
    # sampled_lines = random.sample(all_lines, subset_size)

    # temp_csv_path = "train_half_temp.csv"
    # with open(temp_csv_path, "w") as f:
    #     f.writelines(sampled_lines)

    train_dataset = YOLODataset("train.csv", "images", "labels", transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=8, persistent_workers=True
    )

    learning_rate = 2e-5
    epochs = 100
    best_loss = float("inf")
    patience = 5
    counter = 0

    model = YOLOv1().to(device)
    criterion = YoloLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_training = time.time()
    print("\nStarting training...\n")

    for epoch in range(epochs):
        model.train()
        losses = []
        epoch_start = time.time()

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
        for batch_idx, (X, y) in enumerate(loop):
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            loop.set_postfix(loss=loss.item())

        mean_loss = sum(losses) / len(losses)
        if mean_loss < best_loss:
            best_loss = mean_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            counter += 1
            if counter >= patience:
                print(f"\nEarly stopping triggered. No improvement for {patience} epochs.")
                break

        epoch_duration = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{epochs}] | Mean Loss: {mean_loss:.4f} | Time: {epoch_duration:.2f} sec")

        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")
            print(f"Saved checkpoint at epoch {epoch+1}")

    total_duration = time.time() - start_training
    print(f"\nTraining completed in {total_duration / 60:.2f} minutes.")
    model.load_state_dict(torch.load("best_model.pth"))
    torch.save(model.state_dict(), "yolov1_trained.pth")



if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
