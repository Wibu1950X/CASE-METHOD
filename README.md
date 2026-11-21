# CASE-METHOD
# Analisis Sentimen Ulasan MyPertamina dengan IndoBERT

Repositori ini berisi *notebook*, dataset, dan hasil visualisasi untuk tugas **Case Method Data Mining** mengenai *Text Mining* menggunakan *pretrained model* **IndoBERT** (`mdhugol/indonesia-bert-sentiment-classification`) terhadap 500 ulasan terbaru aplikasi **MyPertamina** (`com.dafturn.mypertamina`) di Google Play Store.

## Alur Program (Notebook)

| Tahap | Deskripsi |
|---|---|
| **1. Setup & Penyimpanan** | Instalasi `google-play-scraper` dan `Sastrawi`, pengecekan GPU (Tesla T4), serta fungsi `export_file()` untuk menyimpan/mengunduh file hasil secara opsional (lokal atau Google Drive). |
| **2. Scraping & Baseline Rating** | Pengambilan ulasan terbaru berbahasa Indonesia (`lang="id"`, `country="id"`, `Sort.NEWEST`). Program mengambil 600 ulasan awal, membuang ulasan kosong, lalu menyaring 500 ulasan terbaru ke `dataset_raw.csv`. Rating bintang dipetakan sebagai **proxy label (*baseline rating*)**: 1–2 (*negative*), 3 (*neutral*), dan 4–5 (*positive*). |
| **3. Text Preprocessing** | Pembersihan teks ringan khusus BERT: *lowercase*, menghapus URL, emoji/karakter non-alfanumerik, tanda baca berulang, serta spasi berlebih. **Tidak dilakukan *stopword removal* maupun *stemming*** agar struktur kalimat dan kata negasi (mis. *"tidak bisa login"*) tetap terbaca utuh oleh BERT. Ulasan tanpa huruf setelah pembersihan dibuang (5 ulasan dibuang, tersisa 495 data bersih). |
| **4. Inference IndoBERT** | Menjalankan *inference* tanpa *training* ulang menggunakan Hugging Face `pipeline` (`text-classification`) di GPU T4 (`batch_size=32`, `truncation=True`, `max_length=512`). Label model dipetakan ulang: `LABEL_0` → *positive*, `LABEL_1` → *neutral*, `LABEL_2` → *negative*, lalu disimpan ke `hasil_bert.csv` beserta skor *confidence*. |
| **5. Evaluasi Model** | Membandingkan prediksi IndoBERT terhadap *baseline rating* menggunakan *Accuracy* (*agreement*), *Classification Report*, dan *Confusion Matrix* (disimpan sebagai `confusion_matrix.png`, 300 dpi). Disertakan pula perhitungan *agreement* pada subset *baseline rating* non-netral (bintang 1–2 dan 4–5). |
| **6A. Analisis Anomali** | Menyaring ulasan yang mengalami diskrepansi ekstrem antara rating dan isi teks: rating 4–5 yang diprediksi *negative*, serta rating 1–2 yang diprediksi *positive*. Sampel disimpan ke `contoh_anomali.csv`. |
| **6B. Ekstraksi Bigram Keluhan** | Menghitung pasangan dua kata (*bigram*) pada 200 ulasan bersentimen *negative* menggunakan `CountVectorizer(ngram_range=(2, 2), min_df=2)`. *Stopword removal* Sastrawi baru diterapkan di tahap ini, dengan pengecualian kata negasi/kendala (`KEEP`) agar frasa masalah seperti *"tidak bisa"* atau *"gagal terus"* tidak terhapus. |
| **6C. Visualisasi** | Membuat grafik distribusi sentimen (`grafik_distribusi_sentimen.png`) dan 10 frasa keluhan terbanyak (`grafik_top10_frasa_keluhan.png`) beresolusi 300 dpi. |
| **6D. Pemetaan Akar Masalah** | Pengelompokan otomatis 200 ulasan negatif menggunakan *rule-based keyword matching* (*regex*) bersifat *multi-label* ke tiga kategori utama: *QR/Barcode & Ganti Plat*, *Stabilitas & Performa Aplikasi*, dan *Pembayaran & E-Wallet*. |

> **Catatan Metodologis:** Kolom `label_true` pada kode merupakan *proxy label* hasil pemetaan rating bintang, bukan *ground truth* dari anotasi manual manusia. Oleh karena itu, metrik evaluasi mengukur tingkat kesesuaian (*agreement*) antara prediksi teks IndoBERT dengan rating pengguna.

## Daftar File Output
* `dataset_raw.csv` — 500 ulasan mentah hasil *scraping*.
* `hasil_bert.csv` — 495 ulasan bersih beserta prediksi IndoBERT dan skor *confidence*.
* `contoh_anomali.csv` — 10 sampel kasus perbedaan antara rating bintang dan sentimen teks.
* `confusion_matrix.png` — *Confusion Matrix* prediksi IndoBERT vs *baseline rating*.
* `grafik_distribusi_sentimen.png` — Distribusi sentimen hasil prediksi IndoBERT.
* `grafik_top10_frasa_keluhan.png` — Top 10 frasa keluhan (*bigram*) pada ulasan negatif.

## Cara Menjalankan di Google Colab
1. Buka file `.ipynb` di [Google Colab](https://colab.research.google.com/) (**File → Open notebook → GitHub**).
2. Aktifkan GPU melalui menu **Runtime → Change runtime type → T4 GPU → Save**.
3. Pastikan `DOWNLOAD_FILES = False` pada Cell 1 (ubah ke `True` hanya jika ingin mengunduh ulang seluruh file output secara otomatis).
4. Jalankan sel secara berurutan dari Cell 1 hingga Cell 6D.
   * *Untuk reproduksi angka yang identik dengan laporan:* unggah file `dataset_raw.csv` dari repositori ini ke sesi Colab, lalu **lewati Cell 2** agar tidak melakukan *scraping* ulang data baru.

## Ringkasan Hasil
* **Total Data Bersih:** 495 ulasan (inferensi selesai dalam 4,0 detik di GPU T4).
* **Distribusi Sentimen (IndoBERT):** Positif 47,5% (235), Negatif 40,4% (200), Netral 12,1% (60).
* **Kesesuaian terhadap *Baseline Rating*:** 78,6% untuk 3 kelas (*Weighted F1-Score* 0,808; *Macro F1-Score* 0,620) dan 81,9% pada subset *baseline rating* non-netral (bintang 1–2 dan 4–5).
* **Kasus Diskrepansi:** 28 ulasan (5,7%) rating 4–5 diprediksi negatif; 4 ulasan (0,8%) rating 1–2 diprediksi positif.
* **Pemetaan Akar Masalah (Multi-label, n = 200 ulasan negatif):**
  * QR/Barcode & Ganti Plat: 78 ulasan (39,0%)
  * Stabilitas & Performa Aplikasi: 75 ulasan (37,5%)
  * Pembayaran & E-Wallet: 30 ulasan (15,0%)
  * *Overlap* (>1 kategori sekaligus): 44 ulasan (22,0%)
  * Lain-lain / keluhan umum: 68 ulasan (34,0%)
