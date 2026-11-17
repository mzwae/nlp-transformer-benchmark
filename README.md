# Arabic NLP Transformer & Generative LLM Benchmark

A rigorous empirical reproduction and architectural comparison of transformer encoders (`MARBERTv2`) and generative large language models (`Qwen2.5-Instruct`) on Arabic sentiment analysis and sarcasm detection using the `ArSarcasm-v2` dataset.

---

## Key Results & Performance Comparison

| Model Architecture | Approach | Target Task | Evaluation Metric | Score | Paper Benchmark |
| --- | --- | --- | --- | --- | --- |
| **MARBERTv2** *(UBC-NLP)* | Fine-Tuned (3 Epochs) | Sentiment Analysis | True $F_{PN}$ Score | **0.7470** | ~0.7240 |
| **MARBERTv2** *(UBC-NLP)* | Fine-Tuned (5 Epochs) | Sarcasm Detection | Sarcasm F1 ($F1_{\text{sarc}}$) | **0.5326** | ~0.5840 |
| **Qwen2.5-1.5B-Instruct** | Zero-Shot Prompting | Sentiment Analysis | True $F_{PN}$ Score | **0.5462** | N/A (Baseline) |

---

## Methodology & Pipeline Architecture

The project implements a robust three-stage benchmarking pipeline designed for dialectal Arabic social media text:

1. **Preprocessing & Schema Alignment:** Handled schema mapping for abbreviated multi-class labels (`neg`, `neu`, `pos`) and recasted Arrow table features to resolve multi-class label bounds.
2. **Task 1 - Sarcasm Detection:** Fine-tuned `MARBERTv2` on binary sarcasm labels, evaluating generalization limits and managing class imbalance.
3. **Task 2 - Sentiment Analysis:** Fine-tuned `MARBERTv2` for 3-class polarity classification, optimizing epochs (2–3) to prevent validation loss divergence and overfitting.
4. **Task 3 - Generative LLM Baseline:** Implemented a zero-shot prompting evaluation harness using `Qwen2.5-1.5B-Instruct` in FP16 precision, utilizing precise token slicing to isolate model completions from prompt text.

