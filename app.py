from flask import Flask, render_template, request, redirect, url_for
import os
import uuid
import torch
from tabulate import tabulate
import model as model_module
from model import predict, load_model
from gradcam import GradCAM, apply_heatmap, calculate_heatmap_stats
from fuzzy_logic import calculate_confidence
from explanation import generate_explanation
from history import init_db, save_result, get_all_history, get_history_by_id, delete_history

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

init_db()

@app.route("/")
def index():
    return render_template("index.html", current_model=model_module.current_model_name)

@app.route("/predict", methods=["POST"])
def predict_route():
    file = request.files["image"]

    ext = os.path.splitext(file.filename)[1]
    safe_filename = str(uuid.uuid4()) + ext
    image_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(image_path)

    results, input_tensor = predict(image_path)

    table_data = [[r["label"], f"{r['probability']}%"] for r in results]
    print(f"\nModel / 使用モデル: {model_module.current_model_name}")
    print(tabulate(table_data, headers=["Class / クラス", "Probability / 確率"], tablefmt="grid"))

    target_layer = model_module.get_target_layer()
    cam = GradCAM(model_module.current_model, target_layer)

    with torch.no_grad():
        output = model_module.current_model(input_tensor)
    top_class_idx = output.argmax(dim=1).item()

    heatmap = cam.generate(input_tensor, top_class_idx)

    heatmap_stats = calculate_heatmap_stats(heatmap)
    print(f"\nGrad-CAM Statistics / 統計: "
          f"Max Activation/最大注目度={heatmap_stats['max_activation']}%, "
          f"Mean Activation/平均注目度={heatmap_stats['mean_activation']}%, "
          f"Concentration/集中度={heatmap_stats['concentration_score']}%")

    top_probability = results[0]["probability"]
    fuzzy_result = calculate_confidence(
        prob_value=top_probability,
        max_act_value=heatmap_stats["max_activation"],
        concentration_value=heatmap_stats["concentration_score"]
    )
    print(f"\nFuzzy Confidence / ファジィ信頼度: "
          f"{fuzzy_result['score']}% ({fuzzy_result['label']})")

    show_warning = fuzzy_result["score"] < 50
    alternative_labels = [r["label"] for r in results[1:3]]

    technical_explanation = generate_explanation(
        results[0]["label"], top_probability, heatmap_stats, fuzzy_result, audience="technical"
    )
    general_explanation = generate_explanation(
        results[0]["label"], top_probability, heatmap_stats, fuzzy_result, audience="general"
    )

    heatmap_filename = "heatmap_" + safe_filename
    heatmap_only_filename = "heatmaponly_" + safe_filename
    heatmap_path = os.path.join(UPLOAD_FOLDER, heatmap_filename)
    heatmap_only_path = os.path.join(UPLOAD_FOLDER, heatmap_only_filename)
    apply_heatmap(image_path, heatmap, heatmap_path, heatmap_only_path)

    save_result(
        image_filename=safe_filename,
        heatmap_filename=heatmap_filename,
        heatmap_only_filename=heatmap_only_filename,
        model_name=model_module.current_model_name,
        top1_label=results[0]["label"],
        top1_probability=results[0]["probability"],
        top5_predictions=results,
        gradcam_stats=heatmap_stats,
        fuzzy_confidence_score=fuzzy_result["score"],
        fuzzy_confidence_label=fuzzy_result["label"],
        show_warning=show_warning,
        alternative_labels=alternative_labels
    )

    return render_template("index.html",
                           results=results,
                           image_path=image_path,
                           heatmap_path=heatmap_path,
                           heatmap_only_path=heatmap_only_path,
                           heatmap_stats=heatmap_stats,
                           fuzzy_result=fuzzy_result,
                           technical_explanation=technical_explanation,
                           general_explanation=general_explanation,
                           show_warning=show_warning,
                           alternative_labels=alternative_labels,
                           current_model=model_module.current_model_name)

@app.route("/switch_model", methods=["POST"])
def switch_model():
    model_name = request.form.get("model_name")
    load_model(model_name)
    return render_template("index.html", current_model=model_module.current_model_name)

@app.route("/history")
def history():
    records = get_all_history()
    return render_template("history.html", records=records)

@app.route("/history/<int:history_id>")
def history_detail(history_id):
    record = get_history_by_id(history_id)
    if record is None:
        return "指定された履歴が見つかりません / History record not found", 404
    return render_template("history_detail.html", record=record)

@app.route("/history/<int:history_id>/delete", methods=["POST"])
def history_delete(history_id):
    delete_history(history_id)
    return redirect(url_for("history"))

if __name__ == "__main__":
    app.run(debug=True)