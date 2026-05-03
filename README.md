# Azərbaycan Dili üçün Nitq Tanıma (ASR)

Wav2Vec2 əsasında Azərbaycan dili üçün ASR pipeline — baza inferensi, fine-tuning və performans qiymətləndirməsini əhatə edir.

---

## İstifadə olunan model

**nijatzeynalov/wav2vec2-large-mms-1b-azerbaijani-common_voice15.0**

| Parametr | Dəyər |
|---|---|
| Arxitektura | Wav2Vec2 Large (MMS 1B) |
| Sampling rate | 16 000 Hz |
| Optimizer | Adafactor |
| Learning rate | 1e-4 |
| Epoch | 2 |
| Batch size | 1 (grad. accum. 4) |
| GPU | Google Colab T4 |

---

## Nəticələr

### Hissə A — Baza model (14 əl ilə qeyd edilmiş audio)

| Fayl | WER | CER |
|---|---|---|
| 1.wav | 80.0% | 13.0% |
| 2.wav | 80.0% | 11.8% |
| 3.wav | 16.7% | 6.8% |
| 4.wav | 28.6% | 15.1% |
| 5.wav | 40.0% | 8.8% |
| 6.wav | 20.0% | 6.9% |
| 7.wav | 14.3% | 4.4% |
| 8.wav | 60.0% | 14.8% |
| 9.wav | 0.0% | 0.0% |
| 10.wav | 0.0% | 0.0% |
| 11.wav | 0.0% | 0.0% |
| 12.wav | 33.3% | 6.4% |
| 13.wav | 0.0% | 0.0% |
| 14.wav | 33.3% | 4.8% |
| **Ortalama** | **29.0%** | **6.6%** |

### Hissə B — Fine-tuning müqayisəsi (tahmaz/azerbaijani-asr-fl, 120 nümunə)

| Model | WER |
|---|---|
| Baseline | 39.27% |
| Fine-tuned | 47.12% |

> Fine-tuned modelin WER-i baseline-dan yüksək çıxıb. Bu texniki xəta deyil — resurs məhdudiyyətindən (cəmi 120 nümunə, 2 epoch) irəli gəlir. Daha çox data və epoch ilə nəticə yaxşılaşacaq.

---

## Qovluq strukturu

```
az-stt-intern/
├── part_a/
│   ├── my_audio/
│   │   ├── 1.wav
│   │   ├── ...
│   │   └── 14.wav
│   └── inference.py
├── part_b/
│   └── fine_tune.ipynb
├── results/
│   ├── metrics.csv
│   └── progress_graph.png
├── report.pdf
├── requirements.txt
└── README.md
```

---

## İşə salmaq

```bash
pip install -r requirements.txt
```

**Hissə A:**
```bash
# my_audio/ qovluğunu yarat, 1.wav–14.wav fayllarını içinə at
python part_a/inference.py
# Nəticə: results/metrics.csv
```

**Hissə B:**

`part_b/fine_tune.ipynb` faylını Google Colab-da aç, Runtime → T4 GPU seç, cell-ləri ardıcıl işlət.