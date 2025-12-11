import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. DNN 모델 정의
# -----------------------------
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# 2. 모델 불러오기
# -----------------------------
@st.cache_resource
def load_model():
    model = DNN()
    model.load_state_dict(torch.load("mnist_dnn.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
st.title("MNIST Digit Recognition with DNN")


# -----------------------------
# 3. MNIST 테스트 이미지 선택
# -----------------------------
st.subheader("Select MNIST Test Image")

# MNIST 테스트 세트 불러오기
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)

index = st.slider("Select image index", 0, len(test_dataset)-1, 0)
img, label = test_dataset[index]

# 이미지 표시
# st.image(img.squeeze().numpy(), caption=f"True Label : {label}", width=150)
st.image(img.squeeze().numpy().clip(0,1), caption=f"True Label : {label}", width=150)


# -----------------------------
# 4. 추론 버튼
# -----------------------------
if st.button("Predict"):
    img_flat = img.view(1, 28*28)
    with torch.no_grad():
        output = model(img_flat)
        pred = torch.argmax(output, dim=1).item()

    st.success(f"Predicted Label : **{pred}**")


# -----------------------------
# 5. 사용자 업로드 이미지 테스트
# -----------------------------
st.subheader("Upload Your Own Digit Image (PNG/JPG)")

uploaded = st.file_uploader("Upload 28x28 MNIST style image", type=["png", "jpg", "jpeg"])

if uploaded:
    user_img = Image.open(uploaded).convert("L").resize((28, 28))
    st.image(user_img, caption="Uploaded Image", width=150)

    # 전처리 동일하게 적용
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    input_tensor = transform(user_img).view(1, 28*28)

    with torch.no_grad():
        out = model(input_tensor)
        pred2 = torch.argmax(out, 1).item()

    st.success(f"Prediction for Uploaded Image: **{pred2}**")
