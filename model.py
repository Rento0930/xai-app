import torch
from torchvision import models, transforms
from PIL import Image

# 現在選択中のモデルとカテゴリを保持する変数
current_model = None
current_categories = None
current_model_name = "resnet50"  # デフォルトのモデル名

def load_model(model_name):
    """指定されたモデルを読み込んでグローバル変数に保存する"""
    global current_model, current_categories, current_model_name

    if model_name == "resnet50":
        current_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        current_categories = models.ResNet50_Weights.DEFAULT.meta["categories"]
    elif model_name == "efficientnet":
        current_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        current_categories = models.EfficientNet_B0_Weights.DEFAULT.meta["categories"]

    current_model.eval()
    current_model_name = model_name  
    return current_model, current_categories

def get_target_layer():
    """Grad-CAMで見る対象の層を、モデルの種類に応じて返す"""
    if current_model_name == "resnet50":
        return current_model.layer4[-1]
    elif current_model_name == "efficientnet":
        return current_model.features[-1]

# 起動時にデフォルトでResNet50を読み込んでおく
load_model("resnet50")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image_path):
    """画像を分類してTOP5の結果を返す"""
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = current_model(input_tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_idx = torch.topk(probabilities, 5)

    results = []
    for i in range(5):
        results.append({
            "label": current_categories[top5_idx[i]],
            "probability": round(top5_prob[i].item() * 100, 2)
        })
    return results, input_tensor