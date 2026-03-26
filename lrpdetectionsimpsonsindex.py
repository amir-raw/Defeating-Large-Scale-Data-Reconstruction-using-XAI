#simple working, Needs lengthy experiment.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import copy
import warnings

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Scaled Settings ---
batch_size = 64        
num_clients = 15
num_bins = batch_size * 4
local_epochs = 50        
learning_rate = 1e-3
noise_scale = 0.15      #  Gaussian Defense strength

# Data Prep
transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
dst = datasets.CIFAR10(root="~/.torch", train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dst, batch_size=batch_size*num_clients, shuffle=True)
images, labels = next(iter(loader))
images, labels = images.to(device), labels.to(device)


class imprintLayer(nn.Module):
    def __init__(self, number_bins, data_ch=3, im_dim=[32,32]):
        super().__init__()
        self.conv1 = nn.Conv2d(data_ch, data_ch, 3, padding=1)
        self.act = nn.Hardtanh(0, 1)
        self.FC1 = nn.Linear(data_ch * im_dim[0] * im_dim[1], number_bins)
        self.FC2 = nn.Linear(number_bins, data_ch * im_dim[0] * im_dim[1])
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.act(self.FC1(x))
        return F.relu(self.FC2(x))

def detect_and_defend(model, data, threshold=0.15):
    """LRP Detection + Gaussian Defense"""
    model.eval()
    # 1. Forward pass to get activations
    x_conv = torch.flatten(F.relu(model[0].conv1(data)), 1)
    x_bins = model[0].act(model[0].FC1(x_conv))
    output = model[0].FC2(x_bins)

    # 2. LRP (Relevance Calculation)
    # We trace relevance from output back to the Bins
    R_out = output.detach()
    w2 = model[0].FC2.weight
    z = F.linear(x_bins, w2) + 1e-9
    s = R_out / z
    R_bins = x_bins * F.linear(s, w2.t())

    # 3. Anomaly Scoring (Simpson's Index)
    relevance_dist = R_bins.sum(dim=0).abs()
    relevance_dist /= (relevance_dist.sum() + 1e-8)
    concentration = torch.sum(relevance_dist**2).item()

    is_attack = concentration > threshold
    return is_attack, concentration

def apply_gaussian_defense(model, scale):
    """Inject noise into gradients to neutralize the imprint"""
    for param in model[0].parameters():
        if param.grad is not None:
            param.grad.add_(torch.randn_like(param.grad) * scale)

# --- Main Logic ---
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
psnr_arr_all, ssim_arr_all, lpip_arr_all = [], [], []

print(f"Starting Federated Experiment with {num_clients} Clients...")

for c in range(num_clients):
    client_x = images[c*batch_size : (c+1)*batch_size]
    client_y = labels[c*batch_size : (c+1)*batch_size]

    # Setup Model
    imp = imprintLayer(num_bins).to(device)
    # Attack Init: We force high bias to 'imprint' data
    nn.init.constant_(imp.FC1.bias, -0.5)

    net = nn.Sequential(imp, nn.Identity()).to(device) # Simplified for observation
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    w_init = net[0].FC1.weight.clone().detach()

    for epoch in range(local_epochs):
        optimizer.zero_grad()
        out = net(client_x)
        loss = F.mse_loss(out, client_x.view(batch_size, -1)) # Auto-encoder style loss triggers leak
        loss.backward()

        # DETECTION
        is_attack, conc = detect_and_defend(net, client_x)

        # DEFENSE
        if is_attack:
            apply_gaussian_defense(net, noise_scale)

        optimizer.step()

    # --- Metrics Computation ---
    w_final = net[0].FC1.weight.detach()
    grad_leak = (w_final - w_init)[0].view(3, 32, 32) # Attempt to recover first image

    recon = (grad_leak - grad_leak.min()) / (grad_leak.max() - grad_leak.min() + 1e-8)
    target = client_x[0]

    # Calculate scores
    mse = F.mse_loss(recon, target).item()
    p = 10 * np.log10(1 / mse) if mse > 0 else 100
    s = torchvision.ops.misc.SqueezeExcitation 

    # Correct Metric Appends
    psnr_arr_all.append(p)
    from torchmetrics.functional import structural_similarity_index_measure as ssim_func
    ssim_arr_all.append(ssim_func(recon.unsqueeze(0), target.unsqueeze(0)).item())
    lpip_arr_all.append(lpips(recon.unsqueeze(0), target.unsqueeze(0)).item())

    status = "DEFENDED" if is_attack else "CLEAN"
    print(f"Client {c} | Status: {status} | Conc: {conc:.4f} | PSNR: {p:.2f}dB")

# --- Final Global Metrics ---
print("\n" + "="*30)
print("FINAL RECONSTRUCTION METRICS (GLOBAL)")
print("="*30)
print(f"Mean PSNR: {np.mean(psnr_arr_all):.2f} dB")
print(f"Mean SSIM: {np.mean(ssim_arr_all):.4f}")
print(f"Mean LPIPS: {np.mean(lpip_arr_all):.4f}")
