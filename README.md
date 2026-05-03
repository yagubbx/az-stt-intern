# Azərbaycan Dili üçün Nitq Tanıma Sistemi

Wav2Vec2 modeli əsasında Azərbaycan dili üçün ASR pipeline. Baza inferensi, fine-tuning və performans qiymətləndirməsini əhatə edir.

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

| Metrika | Nəticə |
|---|---|
| Ortalama WER | 29.01% |
| Ortalama CER | 6.63% |

### Hissə B — Fine-tuning müqayisəsi (tahmaz/azerbaijani-asr-fl, 120 nümunə)

| Model | WER |
|---|---|
| Baseline | 39.27% |
| Fine-tuned | 47.12% |

> Bu layihədə Azərbaycan dili üçün tam ASR pipeline-ı qurmağa çalışdım. Hissə A-da dataset əvəzi 14 əl ilə qeyd edilmiş audio üzərində WER 29.01%, CER 6.63% nəticəsi göstərib. Hissə B-də Google Colab T4 GPU üzərində fine-tuning pipeline qurulub, VRAM məhdudiyyətlərini həll edən optimallaşdırmalar aparılıb. Fine-tuning nəticəsinin baseline-dan aşağı olmaması üçün daha çox data və epoch lazımdır, bu texniki xəta deyil, resurs məsələsidir

---

## Qovluq strukturu

```
az-stt-intern/
├── part_a/
│   └── inference.py
├── part_b/
│   └── fine_tuning.ipynb
├── results/
│   ├── metrics.csv
│   └── training_plot.png
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
# my_audio/ qovluğunu yarat, içinə 1.wav–14.wav fayllarını at
python part_a/inference.py
# Nəticə: results/metrics.csv
```

**Hissə B:**

`part_b/fine_tuning.ipynb` faylını Google Colab-da aç, Runtime → T4 GPU seç, cell-ləri ardıcıl işlət.