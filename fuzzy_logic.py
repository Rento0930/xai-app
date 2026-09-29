import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ===== 入力変数の定義（予測確率のみ、0〜100の範囲） =====
probability = ctrl.Antecedent(np.arange(0, 101, 1), 'probability')

# ===== 出力変数の定義 =====
confidence = ctrl.Consequent(np.arange(0, 101, 1), 'confidence')

# ===== メンバーシップ関数 =====
probability['low'] = fuzz.trimf(probability.universe, [0, 0, 50])
probability['medium'] = fuzz.trimf(probability.universe, [20, 50, 80])
probability['high'] = fuzz.trimf(probability.universe, [50, 100, 100])

confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 50])
confidence['medium'] = fuzz.trimf(confidence.universe, [20, 50, 80])
confidence['high'] = fuzz.trimf(confidence.universe, [50, 100, 100])

# ===== ファジィルール =====
rule1 = ctrl.Rule(probability['high'], confidence['high'])
rule2 = ctrl.Rule(probability['medium'], confidence['medium'])
rule3 = ctrl.Rule(probability['low'], confidence['low'])

confidence_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

def calculate_confidence(prob_value, max_act_value=None, concentration_value=None):
    """予測確率から判断信頼度を計算する（Grad-CAM統計は評価の結果、使用しない設計に変更）"""
    confidence_sim = ctrl.ControlSystemSimulation(confidence_ctrl)
    confidence_sim.input['probability'] = prob_value
    confidence_sim.compute()
    score = confidence_sim.output['confidence']

    if score >= 65:
        label = "高い / High"
    elif score >= 35:
        label = "中程度 / Medium"
    else:
        label = "低い / Low"

    return {
        "score": round(score, 2),
        "label": label
    }