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

        # 勾配の重要度を計算
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

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
    
def calculate_heatmap_stats(heatmap):
    """ヒートマップから注目度の統計情報を計算する"""
    # 最大注目度（0〜1の範囲。1に近いほど強く注目した場所がある）
    max_activation = float(np.max(heatmap))

    # 平均注目度（画像全体でどれくらい満遍なく注目したか）
    mean_activation = float(np.mean(heatmap))

    # 集中度：閾値0.5以上の範囲がどれくらい狭いか
    # 値が高いほど「一部分に集中して注目した」ことを意味する
    high_activation_ratio = float(np.sum(heatmap > 0.5) / heatmap.size)
    concentration_score = 1 - high_activation_ratio  # 狭いほど高スコア

    return {
        "max_activation": round(max_activation * 100, 2),
        "mean_activation": round(mean_activation * 100, 2),
        "concentration_score": round(concentration_score * 100, 2)
    }