# Proyek Prediksi Harga Time Series (Bitcoin) dengan LSTM & Seq2Seq

Proyek ini bertujuan untuk memprediksi pergerakan harga *Close* Bitcoin menggunakan arsitektur Deep Learning berbasis *Long Short-Term Memory* (LSTM) dan pola *Sequence-to-Sequence* (Seq2Seq) yang dipadukan dengan mekanisme *Attention*.

Berikut adalah rincian tahapan *pipeline* yang dikembangkan pada proyek ini:

## 1. Exploratory Data Analysis (EDA)
Tahapan EDA dilakukan untuk memahami karakteristik dan distribusi data runtun waktu:
- **Analisis Deskriptif:** Mengevaluasi distribusi fitur-fitur utama seperti `Close`, `Volume USDT`, `RSI`, `MACD_Hist`, `ATR`, dan `KAMAO`.
- **Visualisasi Tren:** Memplot pergerakan harga *Close* secara keseluruhan.
- **Analisis Autokorelasi:** Menggunakan plot **ACF** (*Autocorrelation Function*) dan **PACF** (*Partial Autocorrelation Function*) untuk menentukan dependensi kelambanan (*lag*). Dari hasil analisis ACF, ditentukan bahwa batas kelambanan yang signifikan adalah 72 jam, sehingga ditetapkan `WINDOW_SIZE = 72` jam untuk prediksi horizon `HORIZON = 24` jam.

## 2. Feature Engineering
Untuk meningkatkan kinerja model prediktif, berbagai fitur turunan diekstraksi dari data historis:
- **Rolling Statistics:** Menambahkan fitur rata-rata pergerakan (*rolling mean*), standar deviasi (*rolling std*), dan nilai maksimum (*rolling max*) dengan periode 24 jam pada fitur `Close`.
- **Time-Based Features:** Melakukan transformasi siklikal (Sine dan Cosine) pada variabel waktu seperti jam (`hour_sin`, `hour_cos`) dan hari dalam seminggu (`dayofweek_sin`, `dayofweek_cos`) untuk menangkap pola musiman.
- **Data Scaling:** Melakukan normalisasi fitur menggunakan `MinMaxScaler` sehingga skala fitur seragam sebelum masuk ke Jaringan Saraf Tiruan.
- **Sliding Window:** Membentuk dataset *supervised learning* berdasarkan dimensi `WINDOW_SIZE` (input 72 step) dan `HORIZON` (output 24 step).

## 3. Arsitektur Modeling
Eksperimen membangun dua jenis model utama, yang seluruhnya memanfaatkan lapisan (*layers*) *custom* (seperti `CustomMultiHeadAttention`, `CustomDropout`, `CustomDense`, dan `ScaleLayer`):

### A. Baseline LSTM
Model *baseline* ringan yang menerima *encoder input*:
- **Lapisan Utama:** Layer LSTM (48 unit) untuk mengekstraksi representasi sekuensial.
- **Attention:** *Custom Multi-Head Attention* (2 heads, key_dim=8) untuk meningkatkan retensi informasi.
- **Feed Forward & Residual:** Gabungan hasil attention melewati layer Dense, LeakyReLU, lalu digabungkan dengan teknik *Residual Connection* dengan mengulangi target terakhir (*RepeatLastTarget*) dan disesuaikan menggunakan *ScaleLayer*.

### B. Sequence-to-Sequence (Seq2Seq) LSTM
Model kompleks dengan pola *Encoder-Decoder*:
- **Encoder:** Menerima sekuens input (72 jam) melalui layer LSTM (48 unit) dan meneruskan `state_h` serta `state_c` ke Decoder.
- **Decoder:** Layer LSTM memproses *decoder input* (sepanjang horizon 24 jam) menggunakan inisialisasi *state* dari Encoder.
- **Attention Mechanism:** *Custom Multi-Head Attention* menghitung *context vector* menggunakan *decoder outputs* sebagai query dan *encoder outputs* sebagai key/value.
- **Output Layer:** Penggabungan fitur melalui konkatensi, *CustomDropout* (10%), *CustomDense* (32 unit), *CustomLeakyReLU*, dan *CustomDense* (1 unit). Terakhir, sebuah *Residual Connection* digunakan untuk menjumlahkan prediksi residu dengan *decoder input*.

*Proses pelatihan (training) menggunakan Custom Training Loop, fungsi kerugian kustom (`custom_mae_loss` & `weighted_horizon_mae`), Custom Early Stopping, dan Reduce Learning Rate On Plateau.*

## 4. Inference (Prediksi & Evaluasi)
Tahap akhir dilakukan untuk mengevaluasi kemampuan prediksi model pada data *test*:
- Model mengeluarkan prediksi untuk 24 step ke depan.
- Prediksi dikembalikan ke skala asalnya menggunakan `inverse_transform` pada *scaler*.
- **Perbandingan Nilai:** Dihitung selisih dan galat absolut (`abs_error`) antara nilai prediksi (`predicted_close`) dengan aktual (`actual_close`).
- Model dievaluasi menggunakan metrik **Mean Absolute Error (MAE)**.
- **Pemilihan Model:** Secara dinamis membandingkan model Baseline dan Seq2Seq berdasarkan MAE, dan menyimpan model terbaik dengan nama `best_model_seq2seq_LSTM.keras`.
