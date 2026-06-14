# Ödev-2: Eğitilen Word2Vec Modelleri ile Metin Benzerliği Hesaplama ve Değerlendirme

**Proje Konusu:** Mobil Uygulama (Seyahat Uygulaması) Yorum Analizi
**Proje Sahipleri:** Miraç BİRBEN (2202131003), Ayhan TUĞALAY (2202131012), Muhammet Ali ÇEBİ (2202131019)

Bu çalışma, Ödev-1'de oluşturulan iki temiz veri seti (`lemmatized_veri_seti.csv`, `stemmed_veri_seti.csv`) üzerinde **16 Word2Vec modeli** eğitir, bir örnek giriş metnine en benzer 5 yorumu her model için bulur ve modelleri **Cosine**, **Anlamsal (1–5)** ve **Jaccard** olmak üzere üç yöntemle karşılaştırmalı olarak değerlendirir.

## Klasör Yapısı
```
odev2/
├── train_and_evaluate.py     # Görev-1 + Görev-2 + 3 değerlendirme (ana script)
├── build_report.py           # outputs/ çıktılarından Word/PDF rapor üretir
├── requirements.txt
├── README.md
├── model/                    # Eğitilen 16 Word2Vec modeli (.model)
├── outputs/                  # Tüm sonuç tabloları (.csv) + jaccard_heatmap.png
└── rapor/                    # Odev2_Raporu.docx ve Odev2_Raporu.pdf
```

## Kurulum
Python 3.9+ önerilir.
```bash
pip install -r requirements.txt
```

## Çalıştırma

**1) Modelleri eğit ve tüm değerlendirmeleri üret:**
```bash
python train_and_evaluate.py
```
Bu komut:
- `lemmatized` ve `stemmed` setleri için 8'er = **16 Word2Vec modeli** eğitir ve `model/` altına `.model` olarak kaydeder.
- Her model için anahtar kelimeye (`premium`) en yakın 5 kelimeyi çıkarır → `outputs/similar_words.csv`
- Bir örnek giriş metni için her modelde en benzer 5 yorumu bulur → `outputs/top5_per_model.csv`
- **Cosine** değerlendirme tablosu → `outputs/cosine_eval.csv`
- **Anlamsal** değerlendirme tablosu (1–5) → `outputs/semantic_eval.csv`
- **16×16 Jaccard matrisi** → `outputs/jaccard_matrix.csv` ve **heatmap** → `outputs/jaccard_heatmap.png`

**2) PDF/Word raporu üret (opsiyonel — hazır rapor `rapor/` içinde mevcuttur):**
```bash
python build_report.py
```
> PDF dönüşümü `docx2pdf` üzerinden MS Word ile yapılır (Windows). Word yoksa oluşan `.docx`'i elle PDF'e çevirebilirsiniz.

## Yöntem Özeti
- **Vektörleştirme:** Gensim `Word2Vec`, `min_count=2`, `epochs=60`, `workers=1`, `seed=42`. Değişen parametreler: tür (CBOW/SkipGram), `window` (2/4), `vector_size` (100/300).
- **Doküman vektörü:** Metindeki kelime vektörlerinin **aritmetik ortalaması**. Modelde olmayan (OOV) kelimeler atlanır; hiçbir kelime yoksa **sıfır vektörü** atanır (NaN/ZeroDivision koruması).
- **Benzerlik:** Giriş metni vektörü ile her doküman vektörü arasında **cosine similarity**; en yüksek 5 sonuç seçilir.
- **Jaccard:** İki modelin ilk 5 sonuç kümesinin kesişim/birleşim oranı.

## Modeller
Boyut nedeniyle GitHub'a yüklenemezse modeller Google Drive'a yüklenir ve link buraya eklenir.
Drive linki: _(gerekirse buraya eklenecek — herkese açık erişim sağlanmalıdır)_

Eğitilen model isimleri örneği: `word2vec_lemmatized_cbow_win2_dim100.model`, `word2vec_stemmed_skipgram_win4_dim300.model` …
