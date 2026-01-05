import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm
import os
from models.wideresnet import *
from models.resnet import *
from trades import trades_loss
from PIL import Image
import torchvision.transforms.functional as TF

from utils import get_model  # Your custom WRN loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 1. Load Your Classifier
# -------------------------
num_classes = 10
classifier = get_model("wrn-28-10", num_classes=num_classes, normalize_input=False)
checkpoint = torch.load('/home/c01sogh/CISPA-home/trades/TRADES-master/cifar10_20percent_dataratio0.3generated_DDPM_beta60percentrandomsamples/model-wideres-epoch200.pt')
classifier = nn.DataParallel(classifier).to(device)
classifier.load_state_dict(checkpoint)
classifier.eval()
for p in classifier.parameters():
    p.requires_grad = False  # Freeze classifier

# -------------------------
# 2. Load DDPM UNet Model
# -------------------------
model = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device)
model.train()

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")

# -------------------------
# 3. CIFAR-10 DataLoader
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # [-1, 1] for DDPM
])
trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

# -------------------------
# 4. Confidence Minimization Loss
# -------------------------
def confidence_minimization_loss(logits):
    probs = F.softmax(logits, dim=1)
    max_conf = probs.max(dim=1)[0]
    return -max_conf.mean()

# -------------------------
# 5. Fine-Tuning Loop
# -------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
lambda_adv = 0.5  # weight for adversarial loss
n_epochs = 7

for epoch in range(n_epochs):
    pbar = tqdm(trainloader)
    for batch in pbar:
        x_start, _ = batch
        x_start = x_start.to(device)

        # 1. Sample random timestep
        bsz = x_start.size(0)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()

        # 2. Add noise
        noise = torch.randn_like(x_start)
        x_noisy = scheduler.add_noise(x_start, noise, t)

        # 3. Predict noise
        noise_pred = model(x_noisy, t).sample

        # 4. Standard DDPM loss
        ddpm_loss = F.mse_loss(noise_pred, noise)

        # 5. Reconstruct x_start from predicted noise
        with torch.no_grad():
            x_recon_list = []
            for i in range(x_noisy.size(0)):
                x_i = x_noisy[i].unsqueeze(0)
                noise_pred_i = noise_pred[i].unsqueeze(0)
                t_i = t[i].item()  # convert to scalar
                out = scheduler.step(noise_pred_i, t_i, x_i)
                x_recon_list.append(out["prev_sample"])

            x_recon = torch.cat(x_recon_list, dim=0)


        # 6. Adversarial loss (confidence minimization)
        logits = classifier(x_recon)
        adv_loss = confidence_minimization_loss(logits)

        # 7. Total loss
        loss = ddpm_loss + lambda_adv * adv_loss

        # 8. Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | DDPM: {ddpm_loss.item():.4f} | Adv: {adv_loss.item():.4f}")

# -------------------------
# 6. Save Fine-Tuned Model
# -------------------------
os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "/home/c01sogh/CISPA-home/trades/TRADES-master/fine_tuned_ddpm_confmin.pth")
print("✅ Fine-tuned DDPM saved.")

# -------------------------
# 7. Generate Samples

# -------------------------
# 7. Generate 30,000 Samples in Batches
# -------------------------
print("🎨 Generating 30,000 risky samples in batches...")
model.eval()

total_images = 100000
batch_size = 100   # adjust depending on your GPU memory
start_index = 0  # adjust starting index if continuing
output_dir = "/home/c01sogh/CISPA-az6/dropattack-2024/YOLO_project_2/trades/newdiff_5_riskynotminus"
os.makedirs(output_dir, exist_ok=True)

num_batches = 1000
scheduler.set_timesteps(1000)

for batch_num in range(num_batches):
    # 1. Sample initial Gaussian noise
    x = torch.randn((batch_size, 3, 32, 32), device=device)

    # 2. Reverse DDPM process
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(x, t).sample
        x = scheduler.step(noise_pred, t, x)["prev_sample"]

    # 3. Unnormalize to [0,1]
    samples = (x.clamp(-1, 1) + 1) / 2

    # 4. Save each image
    for idx, img_tensor in enumerate(samples):
        img = TF.to_pil_image(img_tensor.cpu())
        img.save(os.path.join(output_dir, f"sample_{start_index + batch_num * batch_size + idx:05d}.png"))

    print(f" Batch {batch_num+1}/{num_batches} saved.")

print(f" All {total_images} images saved to: {output_dir}")
