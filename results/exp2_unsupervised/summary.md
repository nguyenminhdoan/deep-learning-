Test split, mean ± std over 3 seeds

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| BiLSTM, Random init (baseline) | 0.809 ± 0.011 | 0.860 ± 0.018 | 0.663 ± 0.024 | 0.748 ± 0.016 |
| BiLSTM, W2V skip-gram, frozen | 0.786 ± 0.011 | 0.844 ± 0.004 | 0.615 ± 0.029 | 0.711 ± 0.021 |
| BiLSTM, W2V skip-gram, fine-tuned | 0.813 ± 0.007 | 0.855 ± 0.015 | 0.681 ± 0.035 | 0.757 ± 0.016 |
| BiLSTM, W2V CBOW, frozen | 0.794 ± 0.005 | 0.837 ± 0.011 | 0.647 ± 0.008 | 0.730 ± 0.007 |
| BiLSTM, W2V CBOW, fine-tuned | 0.815 ± 0.004 | 0.856 ± 0.008 | 0.685 ± 0.007 | 0.761 ± 0.006 |
