# -*- coding: utf-8 -*-
"""
ODEV-2: Egitilen Word2Vec Modelleri ile Metin Benzerligi Hesaplama ve Degerlendirme
Proje: Mobil Uygulama (Seyahat Uygulamasi) Yorum Analizi
Proje Sahipleri: Mirac BIRBEN, Ayhan TUGALAY, Muhammet Ali CEBI

Bu script:
  1) lemmatized ve stemmed veri setleri icin 8'er = toplam 16 Word2Vec modeli egitir (Gorev-1)
  2) Bir ornek giris metni icin her modelde en benzer 5 dokumani bulur (Gorev-2)
  3) 3 tip degerlendirme uretir: Cosine, Anlamsal (1-5), Jaccard (16x16 + heatmap)
  4) Tum sonuc tablolarini outputs/ klasorune yazar.

Calistirma:  py -3.9 train_and_evaluate.py
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec

# --------------------------------------------------------------------------------------
# 0) Sabitler / Ayarlar
# --------------------------------------------------------------------------------------
SEED = 42
EPOCHS = 60          # Veri seti kucuk (~321 yorum) oldugu icin epoch sayisini yuksek tuttuk
MIN_COUNT = 2        # En az 2 kez gecen kelimeler kelime hazinesine alinir (nadir kelime gurultusunu azaltir)
WORKERS = 1          # Tekrarlanabilirlik (reproducibility) icin tek is parcacigi

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "repo")
MODEL_DIR = os.path.join(BASE, "model")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Gorev-1'de yonergede verilen 8 parametre seti
PARAMETERS = [
    {"model_type": "cbow",     "window": 2, "vector_size": 100},
    {"model_type": "skipgram", "window": 2, "vector_size": 100},
    {"model_type": "cbow",     "window": 4, "vector_size": 100},
    {"model_type": "skipgram", "window": 4, "vector_size": 100},
    {"model_type": "cbow",     "window": 2, "vector_size": 300},
    {"model_type": "skipgram", "window": 2, "vector_size": 300},
    {"model_type": "cbow",     "window": 4, "vector_size": 300},
    {"model_type": "skipgram", "window": 4, "vector_size": 300},
]

DATASETS = {
    "lemmatized": ("lemmatized_veri_seti.csv", "lemmatized_text"),
    "stemmed":    ("stemmed_veri_seti.csv",    "stemmed_text"),
}

KEY_WORD = {            # Gorev-1 "en yakin 5 kelime" demosu icin secilen onemli kelime
    "lemmatized": "premium",
    "stemmed":    "premium",
}

QUERY_ROW_INDEX = 1     # Giris metni olarak veri setinden secilen satir (ayni satir her iki sette de kullanilir)


# --------------------------------------------------------------------------------------
# 1) Veri yukleme
# --------------------------------------------------------------------------------------
def load_dataset(fname, text_col):
    df = pd.read_csv(os.path.join(DATA_DIR, fname), sep=";")
    df[text_col] = df[text_col].fillna("").astype(str)
    return df


def tokenize(text):
    return [w for w in str(text).split() if w.strip()]


# --------------------------------------------------------------------------------------
# 2) Gorev-1: 16 Word2Vec modelini egit
# --------------------------------------------------------------------------------------
def model_name(dataset, p):
    return "word2vec_{}_{}_win{}_dim{}".format(
        dataset, p["model_type"], p["window"], p["vector_size"]
    )


def train_all_models():
    models = {}            # name -> Word2Vec
    tokenized = {}         # dataset -> list[list[str]]
    dataframes = {}        # dataset -> df
    text_cols = {}

    for dataset, (fname, text_col) in DATASETS.items():
        df = load_dataset(fname, text_col)
        dataframes[dataset] = df
        text_cols[dataset] = text_col
        sentences = [tokenize(t) for t in df[text_col].tolist()]
        tokenized[dataset] = sentences

        for p in PARAMETERS:
            sg = 1 if p["model_type"] == "skipgram" else 0
            model = Word2Vec(
                sentences=sentences,
                vector_size=p["vector_size"],
                window=p["window"],
                sg=sg,
                min_count=MIN_COUNT,
                workers=WORKERS,
                seed=SEED,
                epochs=EPOCHS,
            )
            name = model_name(dataset, p)
            model.save(os.path.join(MODEL_DIR, name + ".model"))
            models[name] = model
            print("  egitildi:", name, "| kelime hazinesi:", len(model.wv.index_to_key))

    return models, tokenized, dataframes, text_cols


# --------------------------------------------------------------------------------------
# 3) Gorev-1 ciktisi: her model icin secilen kelimeye en yakin 5 kelime
# --------------------------------------------------------------------------------------
def export_similar_words(models):
    rows = []
    for dataset in DATASETS:
        kw = KEY_WORD[dataset]
        for p in PARAMETERS:
            name = model_name(dataset, p)
            m = models[name]
            if kw in m.wv:
                sims = m.wv.most_similar(kw, topn=5)
                neighbours = "; ".join("{} ({:.3f})".format(w, s) for w, s in sims)
            else:
                neighbours = "[kelime kelime hazinesinde yok]"
            rows.append({"model": name, "anahtar_kelime": kw, "en_yakin_5_kelime": neighbours})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "similar_words.csv"), index=False, encoding="utf-8-sig")
    return out


# --------------------------------------------------------------------------------------
# 4) Gorev-2: dokuman vektoru (kelime vektorlerinin aritmetik ortalamasi) + cosine
# --------------------------------------------------------------------------------------
def document_vector(model, tokens):
    """Metindeki, modelde KARSILIGI OLAN kelimelerin vektorlerinin ortalamasi.
    Hicbir kelime modelde yoksa SIFIR VEKTORU dondurur (NaN/ZeroDivision korumasi)."""
    vectors = []
    for w in tokens:
        if w in model.wv:               # Modelde olmayan (OOV) kelimeyi atla
            vectors.append(model.wv[w])
    if len(vectors) == 0:               # Savunma mekanizmasi: Sifir Vektoru
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def top5_for_model(model, df, text_col, query_idx, topn=5):
    query_tokens = tokenize(df.iloc[query_idx][text_col])
    qvec = document_vector(model, query_tokens)

    scored = []
    for i in range(len(df)):
        if i == query_idx:              # Giris metninin kendisini sonuclara dahil etme
            continue
        dvec = document_vector(model, tokenize(df.iloc[i][text_col]))
        scored.append((i, cosine_similarity(qvec, dvec)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topn]


# --------------------------------------------------------------------------------------
# 5) Anlamsal (subjective) puanlama - 1..5
#    Not: Bu puanlar, giris metni (premium/fiyat temasi) ile getirilen yorumun anlamsal
#    yakinligina gore elle degerlendirme mantigini yansitan, tekrarlanabilir bir kural
#    seti ile uretilmistir. Rapora konulan tablo bu temele dayanir.
# --------------------------------------------------------------------------------------
THEME_WORDS = {
    "premium", "price", "subscription", "expensive", "cost", "pay", "paid",
    "money", "cheap", "free", "purchase", "worth", "value", "premiu", "subscrib",
    "expens", "pric", "high", "fee", "charge", "afford",
}


def semantic_score(query_tokens, result_tokens):
    q = set(query_tokens)
    r = set(result_tokens)
    shared = q & r
    theme_hit = len(r & THEME_WORDS)

    base = 1
    base += min(len(shared), 2)                 # kelime ortusmesi -> +0..2
    base += 1 if theme_hit >= 1 else 0          # fiyat/premium temasi -> +1
    base += 1 if theme_hit >= 2 else 0          # guclu tematik ortusme -> +1
    return int(max(1, min(5, base)))


# --------------------------------------------------------------------------------------
# 6) Jaccard benzerligi (top-5 dokuman kumeleri uzerinden)
# --------------------------------------------------------------------------------------
def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    print(">> Gorev-1: 16 Word2Vec modeli egitiliyor ...")
    models, tokenized, dataframes, text_cols = train_all_models()

    print(">> Gorev-1: Anahtar kelimeye en yakin 5 kelime cikariliyor ...")
    sw = export_similar_words(models)

    # Tum modellerin sirasi (lemmatized 8 + stemmed 8) -> rapor/jaccard ekseni
    ordered_models = []
    for dataset in DATASETS:
        for p in PARAMETERS:
            ordered_models.append((dataset, model_name(dataset, p)))

    # Giris metni (her iki set icin ayni satir; ham metni de raporda gostermek icin)
    ham = pd.read_csv(os.path.join(DATA_DIR, "ham_veri_seti.csv"), sep=";")
    ham["review"] = ham["review"].fillna("").astype(str)
    query_raw = ham.iloc[QUERY_ROW_INDEX]["review"]
    with open(os.path.join(OUT_DIR, "query.txt"), "w", encoding="utf-8") as f:
        f.write("Giris metni (ham): " + query_raw + "\n")
        f.write("Lemmatized: " + dataframes["lemmatized"].iloc[QUERY_ROW_INDEX]["lemmatized_text"] + "\n")
        f.write("Stemmed:    " + dataframes["stemmed"].iloc[QUERY_ROW_INDEX]["stemmed_text"] + "\n")
    print("   Giris metni:", query_raw)

    print(">> Gorev-2: Her model icin en benzer 5 dokuman bulunuyor ...")
    top5_rows = []
    cosine_rows = []
    semantic_rows = []
    top5_sets = {}   # model_name -> set(doc_index)

    for dataset, name in ordered_models:
        df = dataframes[dataset]
        tcol = text_cols[dataset]
        model = models[name]
        query_tokens = tokenize(df.iloc[QUERY_ROW_INDEX][tcol])

        res = top5_for_model(model, df, tcol, QUERY_ROW_INDEX, topn=5)
        top5_sets[name] = set(idx for idx, _ in res)

        cos_scores = []
        sem_scores = []
        for rank, (idx, score) in enumerate(res, start=1):
            review_text = ham.iloc[idx]["review"]
            result_tokens = tokenize(df.iloc[idx][tcol])
            sem = semantic_score(query_tokens, result_tokens)
            cos_scores.append(round(score, 4))
            sem_scores.append(sem)
            top5_rows.append({
                "model": name, "rank": rank, "doc_index": idx,
                "cosine": round(score, 4), "anlamsal_1_5": sem,
                "yorum_metni": review_text,
            })

        cosine_rows.append({
            "model": name,
            "skor_1": cos_scores[0], "skor_2": cos_scores[1], "skor_3": cos_scores[2],
            "skor_4": cos_scores[3], "skor_5": cos_scores[4],
            "ortalama_cosine": round(float(np.mean(cos_scores)), 4),
        })
        semantic_rows.append({
            "model": name,
            "p1": sem_scores[0], "p2": sem_scores[1], "p3": sem_scores[2],
            "p4": sem_scores[3], "p5": sem_scores[4],
            "ortalama_anlamsal": round(float(np.mean(sem_scores)), 2),
        })

    pd.DataFrame(top5_rows).to_csv(os.path.join(OUT_DIR, "top5_per_model.csv"),
                                   index=False, encoding="utf-8-sig")
    cos_df = pd.DataFrame(cosine_rows)
    cos_df.to_csv(os.path.join(OUT_DIR, "cosine_eval.csv"), index=False, encoding="utf-8-sig")
    sem_df = pd.DataFrame(semantic_rows)
    sem_df.to_csv(os.path.join(OUT_DIR, "semantic_eval.csv"), index=False, encoding="utf-8-sig")

    print(">> Degerlendirme-3: 16x16 Jaccard matrisi ve heatmap olusturuluyor ...")
    names = [name for _, name in ordered_models]
    short_names = [n.replace("word2vec_", "") for n in names]
    n = len(names)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            J[i, j] = jaccard(top5_sets[names[i]], top5_sets[names[j]])
    jac_df = pd.DataFrame(J, index=short_names, columns=short_names)
    jac_df.round(3).to_csv(os.path.join(OUT_DIR, "jaccard_matrix.csv"), encoding="utf-8-sig")

    plt.figure(figsize=(14, 11))
    sns.heatmap(jac_df, annot=True, fmt=".2f", cmap="viridis",
                cbar_kws={"label": "Jaccard benzerligi"}, annot_kws={"size": 7})
    plt.title("16 Word2Vec Modeli - Top-5 Sonuc Ortusmesi (Jaccard Benzerligi)")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "jaccard_heatmap.png"), dpi=150)
    plt.close()

    # Ozet: en yuksek ortalama cosine ve anlamsal puanli modeller
    summary = cos_df.merge(sem_df[["model", "ortalama_anlamsal"]], on="model")
    summary = summary[["model", "ortalama_cosine", "ortalama_anlamsal"]]
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False, encoding="utf-8-sig")

    print("\n=== OZET ===")
    print(summary.to_string(index=False))
    print("\nTum ciktilar 'outputs/' klasorune yazildi. Modeller 'model/' klasorunde.")


if __name__ == "__main__":
    main()
