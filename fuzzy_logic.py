import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ===== 入力変数の定義（0〜100の範囲） =====
probability = ctrl.Antecedent(np.arange(0, 101, 1), 'probability')
max_activation = ctrl.Antecedent(np.arange(0, 101, 1), 'max_activation')
concentration = ctrl.Antecedent(np.arange(0, 101, 1), 'concentration')

# ===== 出力変数の定義 =====
confidence = ctrl.Consequent(np.arange(0, 101, 1), 'confidence')

# ===== メンバーシップ関数（「低い」「中くらい」「高い」の定義） =====
# 三角形の形で「低い・中・高い」の度合いを表現する
probability['low'] = fuzz.trimf(probability.universe, [0, 0, 50])
probability['medium'] = fuzz.trimf(probability.universe, [20, 50, 80])
probability['high'] = fuzz.trimf(probability.universe, [50, 100, 100])

max_activation['low'] = fuzz.trimf(max_activation.universe, [0, 0, 50])
max_activation['medium'] = fuzz.trimf(max_activation.universe, [20, 50, 80])
max_activation['high'] = fuzz.trimf(max_activation.universe, [50, 100, 100])

concentration['low'] = fuzz.trimf(concentration.universe, [0, 0, 50])
concentration['medium'] = fuzz.trimf(concentration.universe, [20, 50, 80])
concentration['high'] = fuzz.trimf(concentration.universe, [50, 100, 100])

confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 50])
confidence['medium'] = fuzz.trimf(confidence.universe, [20, 50, 80])
confidence['high'] = fuzz.trimf(confidence.universe, [50, 100, 100])

# ===== ファジィルールの定義 =====
# 「もし〇〇なら、信頼度は〇〇」というルールを人間の言葉に近い形で書く
rule1 = ctrl.Rule(probability['high'] & max_activation['high'] & concentration['high'], confidence['high'])
rule2 = ctrl.Rule(probability['high'] & max_activation['high'] & concentration['medium'], confidence['high'])
rule3 = ctrl.Rule(probability['medium'] & max_activation['high'] & concentration['high'], confidence['medium'])
rule4 = ctrl.Rule(probability['low'] | max_activation['low'], confidence['low'])
rule5 = ctrl.Rule(probability['medium'] & max_activation['medium'], confidence['medium'])
rule6 = ctrl.Rule(concentration['low'], confidence['low'])
rule7 = ctrl.Rule(probability['high'] & concentration['low'], confidence['medium'])

# ===== 制御システムの構築 =====
confidence_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])

def calculate_confidence(prob_value, max_act_value, concentration_value):
    """3つの入力から判断信頼度を計算する"""
    confidence_sim = ctrl.ControlSystemSimulation(confidence_ctrl)

    confidence_sim.input['probability'] = prob_value
    confidence_sim.input['max_activation'] = max_act_value
    confidence_sim.input['concentration'] = concentration_value

    confidence_sim.compute()
    score = confidence_sim.output['confidence']

    # 数値をラベルに変換
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