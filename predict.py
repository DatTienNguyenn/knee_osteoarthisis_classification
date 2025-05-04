import os
import torch
import timm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# Thiết lập tham số dòng lệnh
parser = argparse.ArgumentParser(description="Predict knee osteoarthritis using a trained model.")
parser.add_argument("image_path", type=str, help="Path to the image for prediction ")
args = parser.parse_args()

# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Định nghĩa lớp (classes)
classes = ['normal', 'osteopenia', 'osteoporosis']

# Đường dẫn đến file mô hình và ảnh
model_name = "rexnet_150"
save_dir = "saved_models"  # Thay bằng đường dẫn thực tế nếu cần
save_prefix = "knee"
image_path = args.image_path  # Lấy đường dẫn ảnh từ tham số

# Kiểm tra thư mục và file
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir())
if not os.path.exists(save_dir):
    raise FileNotFoundError(f"Directory {save_dir} not found. Please move 'saved_models' to {os.getcwd()} or update save_dir.")

best_model_path = os.path.join(save_dir, f"{save_prefix}_best_model.pth")
if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"File {best_model_path} not found. Please ensure 'knee_best_model.pth' is in {save_dir}.")

if not os.path.exists(os.path.dirname(image_path) or "."):
    raise FileNotFoundError(f"Directory for {image_path} not found. Please move 'data/test' to {os.getcwd()} or update image_path.")
if not os.path.exists(image_path):
    print(f"Warning: File {image_path} not found. Available files in {os.path.dirname(image_path) or '.'}:")
    print(os.listdir(os.path.dirname(image_path) or '.'))
    raise FileNotFoundError(f"Please update image_path to an existing file.")

# Load mô hình đã lưu
model = timm.create_model(model_name, pretrained=False, num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
print(f"Loaded best model from {best_model_path}")

# Hàm tiền xử lý ảnh
def preprocess_xray(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, contrast_factor=2.0))
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Tiền xử lý và hiển thị ảnh với nhãn
input_image = preprocess_xray(image_path)

with torch.no_grad():
    output = model(input_image)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    predicted_prob = probabilities[0][predicted_class].item()

# Ánh xạ kết quả sang nhãn
class_names = {0: 'normal', 1: 'osteopenia', 2: 'osteoporosis'}
predicted_label = class_names[predicted_class]

# Hiển thị ảnh với nhãn
plt.figure(figsize=(6, 6))
plt.imshow(Image.open(image_path).convert('RGB'))
plt.title(f"Predicted: {predicted_label} (Confidence: {predicted_prob:.2f})", fontsize=12, pad=10)
plt.axis('off')

# Thêm nhãn dán lên ảnh
plt.text(10, 20, f"Class: {predicted_label}\nConfidence: {predicted_prob:.2f}", 
         color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.7))

plt.show()

# In kết quả bổ sung
print(f"Predicted class: {predicted_label}")
print(f"Confidence score: {predicted_prob:.4f}")
print(f"Class probabilities: {probabilities[0].tolist()}")