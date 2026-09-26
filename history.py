import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.db")


def init_db():
    """
    データベースとテーブルを初期化する。
    アプリ起動時に一度呼び出せば、既に存在する場合は何もしない。
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            heatmap_filename TEXT,
            heatmap_only_filename TEXT,
            model_name TEXT NOT NULL,
            top1_label TEXT NOT NULL,
            top1_probability REAL NOT NULL,
            top5_predictions TEXT NOT NULL,
            gradcam_max_activation REAL,
            gradcam_mean_activation REAL,
            gradcam_concentration_score REAL,
            fuzzy_confidence_score REAL,
            fuzzy_confidence_label TEXT,
            show_warning INTEGER DEFAULT 0,
            alternative_labels TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_result(
    image_filename: str,
    heatmap_filename: str,
    heatmap_only_filename: str,
    model_name: str,
    top1_label: str,
    top1_probability: float,
    top5_predictions: list,
    gradcam_stats: dict,
    fuzzy_confidence_score: float,
    fuzzy_confidence_label: str,
    show_warning: bool = False,
    alternative_labels: list = None
) -> int:
    """
    1件の解析結果をデータベースに保存する。
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analysis_history (
            timestamp, image_filename, heatmap_filename, heatmap_only_filename,
            model_name, top1_label, top1_probability, top5_predictions,
            gradcam_max_activation, gradcam_mean_activation, gradcam_concentration_score,
            fuzzy_confidence_score, fuzzy_confidence_label,
            show_warning, alternative_labels
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        image_filename,
        heatmap_filename,
        heatmap_only_filename,
        model_name,
        top1_label,
        top1_probability,
        json.dumps(top5_predictions, ensure_ascii=False),
        gradcam_stats.get("max_activation"),
        gradcam_stats.get("mean_activation"),
        gradcam_stats.get("concentration_score"),
        fuzzy_confidence_score,
        fuzzy_confidence_label,
        1 if show_warning else 0,
        json.dumps(alternative_labels or [], ensure_ascii=False)
    ))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def _row_to_dict(row) -> dict:
    """内部用: sqlite3.Rowを辞書に変換し、JSON列を復元する"""
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "image_filename": row["image_filename"],
        "heatmap_filename": row["heatmap_filename"],
        "heatmap_only_filename": row["heatmap_only_filename"],
        "model_name": row["model_name"],
        "top1_label": row["top1_label"],
        "top1_probability": row["top1_probability"],
        "top5_predictions": json.loads(row["top5_predictions"]),
        "gradcam_max_activation": row["gradcam_max_activation"],
        "gradcam_mean_activation": row["gradcam_mean_activation"],
        "gradcam_concentration_score": row["gradcam_concentration_score"],
        "fuzzy_confidence_score": row["fuzzy_confidence_score"],
        "fuzzy_confidence_label": row["fuzzy_confidence_label"],
        "show_warning": bool(row["show_warning"]),
        "alternative_labels": json.loads(row["alternative_labels"]) if row["alternative_labels"] else [],
    }


def get_all_history() -> list:
    """全履歴を新しい順に取得する(一覧ページ用)。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analysis_history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_dict(row) for row in rows]


def get_history_by_id(history_id: int):
    """特定の解析結果を1件取得する(詳細ページ用)。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analysis_history WHERE id = ?", (history_id,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return _row_to_dict(row)


def delete_history(history_id: int) -> bool:
    """特定の履歴を削除する。"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analysis_history WHERE id = ?", (history_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted