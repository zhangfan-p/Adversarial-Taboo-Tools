# Adversarial-Taboo-Tools

## 说明

句子流利度与相关性检测工具

### 1. SentencePPL类

用于检测句子的流利度，输入句子，返回该句子的流利度分数，越低越好

参考 `examples/run_ppl.py`

### 2. SentenceMatch类

用于检测两个句子是否关联，输入两个句子，返回 matched/unmatched（关联/不关联）

参考 `examples/predict_match.py`
