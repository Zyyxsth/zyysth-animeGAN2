from PIL import Image
import torch
import matplotlib.pyplot as plt
# model = torch.hub.load("bryandlee/animegan2-pytorch", "generator").eval()
# out = model(img_tensor)  # BCHW tensor
model = torch.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
face2paint = torch.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)
img = Image.open('E:\\zhangkaihan\\plwy.jpg').convert("RGB")
out = face2paint(model, img)
plt.show()