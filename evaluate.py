import os
import csv
import torch
import random
from PIL import Image
import model as model_module
from model import predict, preprocess
from gradcam import GradCAM, calculate_heatmap_stats
from fuzzy_logic import calculate_confidence
from sklearn.metrics import roc_auc_score

CSV_FILENAME = "evaluation_results.csv"

# 評価用データセットのパス
DATASET_DIR = r"C:\Users\morir\Downloads\Small-ImageNet-Validation-Dataset-1000-Classes-main\Small-ImageNet-Validation-Dataset-1000-Classes-main\ILSVRC2012_img_val_subset"

# ランダムに50クラスを選ぶ(再現性のためseedを固定)
random.seed(42)
TARGET_CLASS_IDS = random.sample(range(1000), 50)


def save_to_csv(results_list):
    """評価結果を既存のCSVに追記する。ファイルが無ければ新規作成してヘッダーを書く"""
    file_exists = os.path.exists(CSV_FILENAME)

    with open(CSV_FILENAME, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results_list)

    print(f"\n{len(results_list)}件の評価結果を{CSV_FILENAME}に追記しました。")


def evaluate():
    results_list = []

    for class_id in TARGET_CLASS_IDS:
        folder_path = os.path.join(DATASET_DIR, str(class_id))
        if not os.path.exists(folder_path):
            print(f"フォルダが見つかりません: {folder_path}")
            continue

        # 各フォルダの最初の1枚だけを使う
        image_files = os.listdir(folder_path)
        if not image_files:
            continue
        image_path = os.path.join(folder_path, image_files[0])

        # 正解ラベル（フォルダ番号がそのままImageNetのクラスIDという前提）
        true_label = model_module.current_categories[class_id]

        # 分類実行
        pred_results, input_tensor = predict(image_path)
        top1_label = pred_results[0]["label"]
        top1_probability = pred_results[0]["probability"]

        # 正解かどうか判定
        is_correct = (top1_label == true_label)

        # Grad-CAM計算
        target_layer = model_module.get_target_layer()
        cam = GradCAM(model_module.current_model, target_layer)
        with torch.no_grad():
            output = model_module.current_model(input_tensor)
        top_class_idx = output.argmax(dim=1).item()
        heatmap_normalized, heatmap_raw = cam.generate(input_tensor, top_class_idx)
        heatmap_stats = calculate_heatmap_stats(heatmap_raw)

        # ファジィ信頼度計算
        fuzzy_result = calculate_confidence(
            prob_value=top1_probability,
            max_act_value=heatmap_stats["max_activation_scaled"],
            concentration_value=heatmap_stats["concentration_score"]
        )

        print(f"[{class_id}] 正解:{true_label} / 予測:{top1_label} / "
              f"{'OK' if is_correct else 'NG'} / 信頼度:{fuzzy_result['score']}%")

        results_list.append({
            "class_id": class_id,
            "true_label": true_label,
            "predicted_label": top1_label,
            "is_correct": is_correct,
            "top1_probability": top1_probability,
            "max_activation_raw": heatmap_stats["max_activation_raw"],
            "mean_activation_raw": heatmap_stats["mean_activation_raw"],
            "concentration_score": heatmap_stats["concentration_score"],
            "fuzzy_confidence_score": fuzzy_result["score"]
        })

    # 全件処理が終わってから、まとめて1回だけCSVに追記する
    if results_list:
        save_to_csv(results_list)
        
    # 定量評価：信頼度スコアで正解/不正解をどれだけ判別できるか
    y_true = [1 if r["is_correct"] else 0 for r in results_list]  # 正解=1, 不正解=0
    fuzzy_scores = [r["fuzzy_confidence_score"] for r in results_list]
    softmax_scores = [r["top1_probability"] for r in results_list]

    correct_fuzzy = [r["fuzzy_confidence_score"] for r in results_list if r["is_correct"]]
    incorrect_fuzzy = [r["fuzzy_confidence_score"] for r in results_list if not r["is_correct"]]

    print("\n===== 評価サマリー =====")
    print(f"正解率: {sum(y_true) / len(y_true) * 100:.2f}%")
    print(f"正解時の平均ファジィ信頼度: {sum(correct_fuzzy) / len(correct_fuzzy):.2f}")
    print(f"不正解時の平均ファジィ信頼度: {sum(incorrect_fuzzy) / len(incorrect_fuzzy):.2f}")

    auroc_fuzzy = roc_auc_score(y_true, fuzzy_scores)
    auroc_softmax = roc_auc_score(y_true, softmax_scores)
    print(f"\nAUROC(ファジィ信頼度で誤分類検出): {auroc_fuzzy:.4f}")
    print(f"AUROC(softmax確率のみで誤分類検出): {auroc_softmax:.4f}")

    # 各指標単体でのAUROCも計算して比較
    auroc_max_act = roc_auc_score(y_true, [r["max_activation_raw"] for r in results_list])
    auroc_concentration = roc_auc_score(y_true, [r["concentration_score"] for r in results_list])

    print(f"\nAUROC(max_activationのみ): {auroc_max_act:.4f}")
    print(f"AUROC(concentration_scoreのみ): {auroc_concentration:.4f}")

if __name__ == "__main__":
    evaluate()