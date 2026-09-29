import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap_raw = torch.mean(activations, dim=0).detach().numpy()
        heatmap_raw = np.maximum(heatmap_raw, 0)

        # 表示用（正規化済み、0〜1）と、統計用（正規化前の生の値）の両方を返す
        raw_max = np.max(heatmap_raw) if np.max(heatmap_raw) > 0 else 1e-8
        heatmap_normalized = heatmap_raw / raw_max

        return heatmap_normalized, heatmap_raw

def apply_heatmap(image_path, heatmap, output_path, heatmap_only_path):
    """元画像にヒートマップを重ねた画像と、ヒートマップ単体の画像を保存する"""
    img = cv2.imread(image_path)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # ヒートマップ単体を保存
    cv2.imwrite(heatmap_only_path, heatmap_colored)

    # 元画像に重ね合わせたものを保存（型をuint8に揃える）
    superimposed = heatmap_colored.astype(np.float32) * 0.4 + img.astype(np.float32)
    superimposed = np.uint8(np.clip(superimposed, 0, 255))
    cv2.imwrite(output_path, superimposed)
    
def calculate_heatmap_stats(heatmap_raw):
    """ヒートマップ（正規化前）から注目度の統計情報を計算する"""
    max_activation_raw = float(np.max(heatmap_raw))
    mean_activation_raw = float(np.mean(heatmap_raw))

    # 実データの分布(50件評価: 概ね0.0002〜0.0008)を基準に0〜100へスケーリング
    # 0.001を上限の目安とし、それ以上は100に丸める
    SCALE_UPPER_BOUND = 0.001
    max_activation_scaled = min((max_activation_raw / SCALE_UPPER_BOUND) * 100, 100)

    raw_max = max_activation_raw if max_activation_raw > 0 else 1e-8
    heatmap_normalized = heatmap_raw / raw_max
    high_activation_ratio = float(np.sum(heatmap_normalized > 0.5) / heatmap_normalized.size)
    concentration_score = 1 - high_activation_ratio

    return {
        "max_activation_raw": round(max_activation_raw, 4),
        "max_activation_scaled": round(max_activation_scaled, 2),
        "mean_activation_raw": round(mean_activation_raw, 4),
        "concentration_score": round(concentration_score * 100, 2)
    }