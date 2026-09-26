from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_explanation(label, probability, heatmap_stats, fuzzy_result, audience="technical"):
    """AIの判断結果から説明文を生成する。audienceで専門的/一般向けを切り替える"""

    if audience == "technical":
        role = "あなたはAI画像分類システムの解析結果を説明する担当です。技術的な内容を専門家向けに、正確かつ簡潔に説明してください。"
        instruction = "専門用語を使って構いません。判断根拠となった数値も具体的に含めてください。"
    else:
        role = "あなたはAI画像分類システムの解析結果を説明する担当です。一般の方向けに、分かりやすい言葉で説明してください。"
        instruction = "専門用語は避け、平易な言葉で何が判定されたのかを伝えてください。"

    prompt = f"""
以下はAI画像分類システムの解析結果です。この内容を説明文にしてください。

判定結果：{label}（確率 {probability}%）
Grad-CAM最大注目度：{heatmap_stats['max_activation']}%
注目集中度：{heatmap_stats['concentration_score']}%
ファジィ論理による信頼度：{fuzzy_result['score']}%（{fuzzy_result['label']}）

{instruction}
3〜4文程度でまとめてください。
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content