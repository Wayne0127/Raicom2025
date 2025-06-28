 #  **睿抗 2025 智海人工智能算法应用赛**

# 🎧 SER-Baseline  

基于 **Wav2Vec 2.0 / HuBERT / EfficientNet-Mel** 的语音情感识别

| 自监督             | 频谱 CNN           | 池化                    | K-Fold | 推理集成 |
| ------------------ | ------------------ | ----------------------- | ------ | -------- |
| Wav2Vec 2.0/Hubert | EfficientNet (Mel) | Attentive / Mean / Stat | ✔️      | ✔️        |

> 本仓库提供三条基线：  
> ① **Wav2Vec2-SER**　② **HuBERT-SER**　③ **Mel-EfficientNet-SER**  
> 支持 **5 折交叉验证** + **多折概率融合**，开箱即用。

---

## 📦 环境依赖

```bash
pip install torch torchaudio transformers efficientnet_pytorch tqdm scikit-learn numpy
```

* Python ≥ 3.9
* 若使用 GPU，需安装对应 CUDA 版 PyTorch
* EfficientNet 依赖 `efficientnet_pytorch`

---

## 📂 目录结构

```
├── notebooks/
│   ├── main_wav2vec2.ipynb      # Wav2Vec2 五折训练
│   ├── main_hubert.ipynb        # HuBERT 五折训练
│   ├── main_mel.ipynb           # EfficientNet-Mel 五折训练
│   └── inference.ipynb          # 多折集成推理测试代码
├── datasets/
├── results/                     # 保存各折权重 *.pt
└── README.md
```

---

## 📊 数据准备

```
datasets/
└── train/
    ├── anger/*.wav
    ├── happy/*.wav
    ├── neutral/*.wav
    └── sad/*.wav
```

* 原始采样率**44.1 kHz**，脚本会自动重采样到 **16 kHz**
* 训练时统一裁剪 / 右填零至 **3 s**（`max_sec` 可自行修改）

---

## 🧩 模型说明

| 模型                   | Backbone                 | 输入          | 隐藏维 | 池化 (可选)            | 分类头                                            |
| ---------------------- | ------------------------ | ------------- | ------ | ---------------------- | ------------------------------------------------- |
| **HF\_Wav2Vec2SER**    | `facebook/wav2vec2-base` | 波形          | 768    | Mean / Stat / **Attn** | LayerNorm → Linear(128) → GELU → Dropout → Linear |
| **HF\_HuBERT\_SER**    | `hubert-base-ls960`      | 波形          | 768    | Mean / Attn            | LayerNorm → Linear(256) → ReLU → Dropout → Linear |
| **MelEfficientNetSER** | **EfficientNet-B0**      | 128 × 300 Mel | 1280   | GAP                    | Linear → ReLU → Dropout → Linear                  |

> **Attn 池化**：`softmax(Linear(tanh(Linear)))`，对高情感帧赋更大权重。

---

## 🚀 训练示例（HuBERT）

```bash
jupyter nbconvert --to script notebooks/main_hubert.ipynb
python notebooks/main_hubert.py  # 自动 5-Fold 训练
```

* 优化器：`AdamW(lr=3e-5)`
* 调度器：`OneCycleLR(max_lr=5e-5, pct_start=0.3)`
* 早停：`patience=10`

权重将保存在 `results/hubert_fold{1..5}.pt`。

---

## 🔍 推理与多折融合

```python
from src.models import HF_HuBERT_SER
from notebooks.inference import load_kfold_models, predict
from transformers import AutoFeatureExtractor
import torchaudio

feature_extractor = AutoFeatureExtractor.from_pretrained("results/hubert-base-ls960")
models = load_kfold_models(HF_HuBERT_SER, "results/hubert_fold{}.pt", folds=3)

wav, sr = torchaudio.load("demo.wav")
emotion = predict(wav, sr, models, feature_extractor)
print("情绪预测 →", emotion)
```

> `predict()` 会对每个模型做 `softmax`，然后 **平均概率** 再取 argmax。
> 如需改为加权平均，对 `prob_sum += weight_i * probs` 调整权重即可。

---

## ⏱️ 性能与速度

| 模型      | 数据集   | OOF - F1 (5-Fold AVG 本地) | 单条推理 (CPU) |
| --------- | -------- | -------------------------- | -------------- |
| Wav2Vec2  | datasets | 0.8635                     | 80 ms          |
| HuBERT    | datasets | **0.8653**                 | 82 ms          |
| Mel-EffB0 | datasets | 0.79                       | **45 ms**      |

---

## FAQ

加载模型时出现多余/缺失权重警告？
这是因为加载的 `.pt` 权重不包含分类头或池化改动后的参数。
* 若只想加载主干，可设 `strict=False`；
* 或重新训练并保存匹配当前结构的权重。

---

## License

如果本项目对你有帮助，欢迎 ⭐Star + Fork！感谢各位！
以后有比赛也可以一起学习交流
