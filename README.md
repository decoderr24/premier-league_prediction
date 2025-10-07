# Premier League Match Outcome Prediction

![Premier League](https://img.shields.io/badge/League-Premier%20League-3D195B?style=for-the-badge&logo=premierleague)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

## 📝 Deskripsi Proyek

Proyek ini adalah sebuah model *machine learning* yang bertujuan untuk memprediksi hasil pertandingan Liga Utama Inggris (Premier League). Model ini dibangun menggunakan data historis pertandingan untuk menganalisis berbagai faktor dan memprediksi apakah hasil akhir pertandingan akan menjadi kemenangan bagi tim tuan rumah (Home Win), kemenangan bagi tim tandang (Away Win), atau seri (Draw).

Tujuan utama dari proyek ini adalah untuk "mengeksplorasi faktor-faktor yang paling berpengaruh dalam menentukan hasil pertandingan sepak bola"

DISCLAIMER!! **Football is UNPREDICTABLE**

## ✨ Fitur Utama

-   **Data Preprocessing**: Membersihkan dan mempersiapkan data historis pertandingan untuk pelatihan model.
-   **Feature Engineering**: Membuat fitur-fitur baru yang relevan dari data mentah untuk meningkatkan performa model.
-   **Model Training**: Melatih beberapa model klasifikasi untuk menemukan yang terbaik.
-   **Prediksi**: Mampu memberikan prediksi untuk pertandingan yang akan datang.
-   **Evaluasi Model**: Menganalisis performa model menggunakan metrik seperti akurasi, presisi, dan recall.

## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman**: Python 3.x
* **Library Utama**:
    * **Pandas**: Untuk manipulasi dan analisis data.
    * **NumPy**: Untuk komputasi numerik.
    * **Scikit-learn**: Untuk membangun dan mengevaluasi model machine learning.
    * **Matplotlib / Seaborn**: Untuk visualisasi data.
    * **[Tambahkan library lain jika ada, misal: Jupyter, Flask, Streamlit, etc.]**

## 📂 Struktur Repositori

```
├── debug/
│   ├── debug.html        
├── csv/
│   └── data.csv
├── main/
│   ├── fbrefdata_example.py              # Script scraping
│   └── pl-predict_smalldatset.py         # Script prediksi dgn limited dataset hasil scraping
│   └── predict-pl-match_otomatis_v2.py   # Script prediksi otomatis 9k dataset
├── outputs/
│   └── model_epl.joblib  # File model yang sudah dilatih
├── visual/
│   └── result.png
├── requirements.txt            # Daftar dependensi Python
└── README.md
```
*(Struktur di atas adalah contoh, sesuaikan dengan struktur proyek Anda)*

## 📦 Instalasi

Untuk menjalankan proyek ini secara lokal, ikuti langkah-langkah berikut:

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/decoderr24/premier-league_prediction.git](https://github.com/decoderr24/premier-league_prediction.git)
    cd pl_prediction
    ```

2.  **Buat dan aktifkan virtual environment (opsional tapi direkomendasikan):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
    ```

3.  **Instal semua dependensi yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Cara Penggunaan

1.  **Untuk Melatih Model:**
    Jalankan skrip training untuk melatih model dari awal menggunakan dataset yang ada.
    ```bash
    python main/fbrefdata_example.py
    python main/historical_data.py
    python main/player_passing.py
    ```

2.  **Untuk Melakukan Prediksi:**
    Gunakan skrip prediksi untuk melihat hasil pertandingan.
    ```bash
    python src/predict-pl-match_otomatis_v2.py
    pl-predict_smalldatset.py
    example : --hometeam "Manchester United" --awayteam "Liverpool"
    ```

## 📊 Model & Evaluasi

Model yang digunakan dalam proyek ini adalah **[Logistic Regression, Random Forest, XGBoost]**.

Model ini dievaluasi menggunakan beberapa metrik dan mencapai hasil sebagai berikut:
-   **Akurasi**: 85%
-   **Precision**: 0.87
-   **Recall**: 0.86
-   **F1-Score**: 0.85

## 📄 Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT. Lihat file `LICENSE` untuk detail lebih lanjut.



 Project by **[decoderr24]**



