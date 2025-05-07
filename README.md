# Sistem Rekomendasi Film Menggunakan MovieLens Dataset

*Oleh: Filza Rahma Muflihah*

## Project Overview

Di era digital saat ini, platform streaming film dan konten daring semakin berkembang pesat. Pengguna dihadapkan pada ribuan pilihan film, namun tidak memiliki waktu untuk menjelajahi semuanya. Hal ini menciptakan fenomena yang disebut sebagai "choice overload" atau "paradox of choice", di mana terlalu banyak pilihan justru membuat pengguna kesulitan mengambil keputusan.

Sistem rekomendasi film hadir sebagai solusi untuk masalah ini, dengan membantu pengguna menemukan konten yang relevan dengan preferensi mereka secara efisien. Sistem ini tidak hanya meningkatkan pengalaman pengguna tetapi juga memberikan manfaat bisnis berupa peningkatan engagement, retensi pengguna, dan pendapatan.

Menurut penelitian McKinsey [1], sistem rekomendasi yang efektif mampu meningkatkan penjualan hingga 35% pada platform e-commerce dan layanan streaming. Netflix sendiri mengklaim bahwa 80% konten yang ditonton pengguna mereka berasal dari rekomendasi, sedangkan YouTube melaporkan bahwa 70% waktu tontonan dihabiskan untuk konten yang direkomendasikan oleh sistem.

Proyek ini akan mengimplementasikan dan membandingkan dua pendekatan utama dalam sistem rekomendasi film: Content-Based Filtering dan Collaborative Filtering, menggunakan dataset MovieLens untuk menghasilkan rekomendasi personalisasi film yang akurat dan relevan.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang di atas, berikut adalah rumusan masalah yang akan diselesaikan dalam proyek ini:

1. Bagaimana mengembangkan sistem rekomendasi film yang dapat menyarankan film relevan kepada pengguna berdasarkan film yang pernah mereka tonton atau sukai sebelumnya?
2. Bagaimana mengembangkan sistem rekomendasi yang dapat menemukan film yang mungkin disukai pengguna berdasarkan preferensi pengguna lain yang memiliki selera serupa?
3. Bagaimana membandingkan performa kedua pendekatan sistem rekomendasi tersebut untuk menentukan pendekatan yang lebih efektif?

### Goals
Tujuan dari proyek ini adalah:

1. Mengembangkan sistem rekomendasi berbasis konten (content-based filtering) yang merekomendasikan film mirip berdasarkan kesamaan fitur (genre) dengan film yang disukai pengguna.
2. Mengembangkan sistem rekomendasi berbasis kolaboratif (collaborative filtering) yang merekomendasikan film berdasarkan pola rating dari pengguna yang memiliki preferensi serupa.
3. Mengevaluasi dan membandingkan performa kedua sistem rekomendasi tersebut menggunakan Mean Squared Error (MSE) sebagai metrik evaluasi. MSE akan mengukur seberapa akurat model dalam memprediksi rating pengguna dengan menghitung rata-rata kuadrat selisih antara rating prediksi dan rating aktual. Semakin kecil nilai MSE, semakin baik performa model.

### Solution Approach

**1. Content-Based Filtering**

Pendekatan ini merekomendasikan film berdasarkan kesamaan fitur (dalam hal ini genre) dengan film yang disukai pengguna di masa lalu. Langkah-langkah implementasinya:

- Mengekstrak fitur genre dari data film
- Menggunakan TF-IDF untuk mengubah data genre menjadi representasi vektor numerik
- Menghitung similaritas kosinus antar item film
- Merekomendasikan film yang memiliki similaritas tertinggi dengan film yang disukai pengguna

Kelebihan pendekatan ini adalah dapat merekomendasikan item baru yang belum di-rating oleh pengguna manapun, sehingga mengatasi masalah "cold start" untuk item.

**2. Collaborative Filtering**

Pendekatan ini merekomendasikan film berdasarkan preferensi pengguna lain yang memiliki pola rating serupa. Implementasinya:

- Menggunakan embedding untuk merepresentasikan pengguna dan film dalam ruang laten
- Melatih model deep learning untuk memprediksi rating pengguna terhadap film
- Merekomendasikan film dengan prediksi rating tertinggi yang belum ditonton pengguna

Kelebihan pendekatan ini adalah dapat menemukan preferensi tersembunyi yang tidak terlihat dari fitur eksplisit film.

## Data Understanding

### Tentang Dataset

Pada proyek ini, saya menggunakan dataset MovieLens 100K yang berisi 100.000 peringkat dari 600 pengguna pada 9.000 film. Dataset ini merupakan versi kecil dari dataset MovieLens lengkap dan sering digunakan untuk pengembangan dan evaluasi sistem rekomendasi.

Dataset dapat diunduh dari: [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/)

Dataset terdiri dari beberapa file, namun dalam proyek ini saya berfokus pada dua file utama:

1. **movies.csv**: Berisi informasi film termasuk judul dan genre
   - `movieId`: ID unik untuk setiap film dalam database
   - `title`: Judul film, biasanya disertai dengan tahun rilis dalam tanda kurung
   - `genres`: Genre film yang dipisahkan dengan karakter '|' (pipe), seperti 'Adventure|Animation|Children'

2. **ratings.csv**: Berisi peringkat pengguna untuk film
   - `userId`: ID unik untuk setiap pengguna dalam database
   - `movieId`: ID film yang diberi rating oleh pengguna
   - `rating`: Nilai rating yang diberikan pengguna (skala 0.5 hingga 5.0 dengan interval 0.5)
   - `timestamp`: Waktu ketika rating diberikan (dalam format UNIX timestamp)

### Data Quality Analysis

Sebelum melakukan penggabungan atau pengolahan data, penting untuk memastikan kualitas dataset yang digunakan:

#### 1. Analisis Missing Values
- **Dataset Film (movies.csv)**:
  - `movieId`: 0 missing values
  - `title`: 0 missing values
  - `genres`: 0 missing values (terdapat 18 film dengan genre "no genres listed")

- **Dataset Rating (ratings.csv)**:
  - `userId`: 0 missing values
  - `movieId`: 0 missing values
  - `rating`: 0 missing values
  - `timestamp`: 0 missing values

#### 2. Analisis Duplikasi Data
- **Dataset Film**: Tidak ditemukan data duplikat berdasarkan `movieId` (primary key)
- **Dataset Rating**: Tidak ditemukan duplikasi data dengan kombinasi `userId` dan `movieId` yang sama (satu pengguna hanya memberikan satu rating untuk satu film)

#### 3. Analisis Outlier
- **Dataset Film**: Tidak terdeteksi outlier yang signifikan pada tahun rilis film
- **Dataset Rating**:
  - Distribusi rating cenderung normal dengan sedikit kemiringan positif
  - Tidak terdapat outlier yang signifikan karena rating dibatasi pada skala 0.5 hingga 5.0
  - Beberapa film memiliki banyaknya rating jauh di atas rata-rata (> 500 rating), yang menunjukkan popularitas film tersebut

Hasil analisis menunjukkan bahwa kedua dataset memiliki kualitas data yang baik tanpa masalah missing value atau duplikasi yang perlu ditangani dalam tahap preprocessing.

### Exploratory Data Analysis

#### 1. Analisis Statistik Deskriptif

**Dataset Film:**
- Jumlah film: 9.742
- Tahun rilis berkisar dari 1902 hingga 2018, dengan rata-rata tahun 1994
- Jumlah genre per film berkisar antara 1 hingga 10, dengan rata-rata 2,27 genre per film

**Dataset Rating:**
- Jumlah rating: 100.836
- Rating berkisar dari 0,5 hingga 5,0 dengan rata-rata 3,5
- Sebagian besar rating berada pada rentang 3,0 hingga 4,0

#### 2. Distribusi Rating

![image](https://github.com/user-attachments/assets/5b9c1522-c080-4256-a320-7ff3cdd223e8)

Distribusi rating film menunjukkan bahwa mayoritas pengguna cenderung memberikan rating positif, dengan puncak tertinggi pada rating 4.0, diikuti oleh 3.0 dan 5.0. Rating rendah seperti 0.5 hingga 2.0 jarang diberikan. Rata-rata rating berada di angka 3.5, menunjukkan adanya kecenderungan pengguna untuk menilai film secara positif.

#### 3. Distribusi Genre

![image](https://github.com/user-attachments/assets/195a0415-7ace-405c-924a-33038e169718)
![image](https://github.com/user-attachments/assets/eab5f5b5-8815-40c5-bc98-30f70869290e)


Genre film yang paling banyak diproduksi adalah Drama dan Comedy, masing-masing dengan lebih dari 4000 dan 3500 film. Genre Thriller, Action, dan Romance juga cukup populer. Sebagian besar film memiliki 1 hingga 3 genre, dengan jumlah film menurun drastis seiring bertambahnya jumlah genre.

#### 4. Distribusi Tahun Rilis

![image](https://github.com/user-attachments/assets/33c5efde-2d91-4391-aaaa-1d09011fd8b6)



Produksi film meningkat secara signifikan sejak tahun 1980-an dan mencapai puncaknya sekitar tahun 2000-an. Grafik ini mencerminkan tren pertumbuhan industri perfilman modern yang pesat dalam beberapa dekade terakhir.

#### 5. Analisis Rating Berdasarkan Genre

![image](https://github.com/user-attachments/assets/a58b8a48-507d-48fd-8a1e-c14261746a59)

Genre Film-Noir menempati posisi teratas dengan rata-rata rating tertinggi, disusul oleh War dan Documentary. Sebaliknya, genre Horror, Comedy, dan Children berada di posisi terbawah. Film dengan genre serius cenderung lebih dihargai dibanding genre yang lebih ringan.

## Data Preparation

Dalam tahap data preparation, saya melakukan beberapa transformasi dan pembersihan data untuk memastikan data siap digunakan dalam pengembangan model. Berikut adalah langkah-langkah yang dilakukan:

### 1. Penggabungan Data

Untuk mendapatkan dataset yang lengkap dan siap dianalisis, saya melakukan penggabungan (merge) antara dataset film dan dataset rating. Hasil penggabungan menghasilkan dataset dengan kolom-kolom: 'userId', 'movieId', 'rating', 'timestamp', 'title', dan 'genres'. Dataset gabungan ini memberikan informasi lengkap tentang setiap rating yang diberikan pengguna, termasuk judul dan genre film yang diberi rating. Penggabungan data ini tidak dimasukkan ke dalam pembuatan model, tetapi berguna untuk analisis lebih lanjut pada EDA sebelumnya.

### 2. Feature Engineering

Tahapan feature engineering dilakukan untuk memperkaya informasi dalam dataset:

- **Ekstraksi Tahun**: Mengekstrak tahun rilis film dari judul menggunakan ekspresi reguler. Fitur ini memungkinkan analisis berdasarkan periode waktu dan melihat tren historis.
  
- **Penambahan Rating Rata-rata**: Menghitung rating rata-rata dan jumlah rating untuk setiap film, yang memberikan informasi tentang popularitas dan penerimaan film.
  
- **Perhitungan Jumlah Genre**: Menambahkan fitur yang menunjukkan berapa banyak genre yang dimiliki oleh setiap film, yang dapat mengindikasikan kompleksitas atau keberagaman film.

### 3. Encoding Fitur Genre dengan TF-IDF

Untuk menggunakan informasi genre dalam content-based filtering, kita perlu mengubah data teks genre menjadi representasi numerik. Pendekatan yang digunakan adalah Term Frequency-Inverse Document Frequency (TF-IDF).

Hasil dari proses ini adalah matriks TF-IDF dengan dimensi (jumlah film × jumlah genre unik), di mana setiap baris merepresentasikan satu film dan setiap kolom merepresentasikan bobot TF-IDF untuk genre tertentu. Matriks ini kemudian akan digunakan untuk menghitung similaritas antar film berdasarkan genre mereka.

### 4. Normalisasi Rating

Untuk model collaborative filtering, rating pengguna dinormalisasi untuk meningkatkan performa model dengan menskala nilai rating ke rentang [0,1].

### 5. Train-Test Split

Data dibagi menjadi set pelatihan dan pengujian dengan proporsi 80:20 untuk evaluasi model.

Transformasi-transformasi ini penting untuk analisis yang lebih mendalam dan meningkatkan kualitas fitur yang tersedia untuk model rekomendasi.

## Modeling and Result

Dalam proyek ini, saya mengembangkan dua model sistem rekomendasi: Content-Based Filtering dan Collaborative Filtering.

### 1. Content-Based Filtering

Model ini merekomendasikan film berdasarkan kesamaan genre dengan film yang sudah disukai pengguna. Melalui konsep tersebut, kita memanfaatkan teknik Cosine Similarity. Cosine Similarity adalah teknik untuk mengukur kesamaan antara dua vektor dengan menghitung kosinus sudut di antara keduanya. Dalam konteks sistem rekomendasi berbasis konten, setiap film direpresentasikan sebagai vektor fitur (dalam hal ini, vektor TF-IDF dari genre). Nilai cosine similarity berkisar antara -1 hingga 1, di mana:
- 1 berarti kedua vektor memiliki arah yang sama (film sangat mirip)
- 0 berarti kedua vektor tegak lurus (film tidak terkait)
- -1 berarti kedua vektor berlawanan arah (film sangat berbeda)

Rumus cosine similarity:
$$\cos(\theta) = \frac{A \cdot B}{||A|| \times ||B||}$$

Dimana:
- A · B adalah dot product dari vektor A dan B
- ||A|| dan ||B|| adalah magnitude (panjang) dari vektor A dan B

Cosine similarity memiliki peran penting dalam CBF karena:
1. Dapat menangkap kesamaan makna semantik meskipun panjang vektor berbeda
2. Fokus pada arah vektor, bukan besarannya, sehingga cocok untuk rekomendasi berbasis kesamaan
3. Efisien untuk komputasi pada data sparse (banyak nilai nol) seperti representasi genre

**Hasil Top-N Rekomendasi untuk "Toy Story (1995)":**

![image](https://github.com/user-attachments/assets/a81618c3-0e11-4681-a2f4-7c2612d58c8c)

Model ini juga dapat digunakan untuk memberikan rekomendasi personalisasi berdasarkan seluruh riwayat rating pengguna.

**Hasil Top-N Rekomendasi untuk userId 255:**

![image](https://github.com/user-attachments/assets/80169726-65b7-43f8-b445-cbb9f4deef92)

### 2. Collaborative Filtering

Model kedua menggunakan pendekatan deep learning untuk mempelajari pola rating dan memberikan rekomendasi berdasarkan preferensi pengguna lain dengan selera serupa. Algoritma yang digunakan pada pendekatan ini adalah RecommenderNet. RecommenderNet adalah model deep learning sederhana yang mengimplementasikan Matrix Factorization melalui embedding. Konsep kerjanya:

1. **Embedding Layer**: Model memiliki dua embedding layer:
   - User Embedding: Memetakan ID pengguna ke vektor latent dimension (ruang tersembunyi)
   - Movie Embedding: Memetakan ID film ke vektor latent dimension yang sama

2. **Latent Space**: Model mempelajari representasi pengguna dan film dalam ruang laten berdimensi rendah (dalam contoh ini 50 dimensi). Dalam ruang laten ini:
   - Pengguna dengan preferensi serupa akan memiliki vektor embedding berdekatan
   - Film dengan karakteristik serupa akan memiliki vektor embedding berdekatan
   - Kedekatan vektor pengguna dan film mengindikasikan potensi kecocokan

3. **Dot Product**: Interaksi pengguna-film dimodelkan sebagai dot product antara vektor embedding pengguna dan film. Nilai dot product yang tinggi menunjukkan kecocokan yang tinggi antara preferensi pengguna dan karakteristik film.

4. **Pembelajaran**: Model belajar dengan meminimalkan selisih antara rating prediksi (hasil dot product) dengan rating aktual yang diberikan pengguna.

Keunggulan model ini adalah kemampuannya untuk menemukan pola tersembunyi dalam data rating tanpa memerlukan fitur konten eksplisit, hanya berdasarkan interaksi pengguna-film sebelumnya.

**Hasil Top-N Rekomendasi untuk User 255:**

![image](https://github.com/user-attachments/assets/b46a5a70-554b-4f14-8f20-98f3e7475ed6)

### Perbandingan Pendekatan

**Content-Based Filtering:**
- **Kelebihan:** 
  - Mampu merekomendasikan film baru yang belum memiliki rating
  - Rekomendasi dapat dijelaskan berdasarkan kesamaan fitur (genre)
  - Tidak memerlukan data dari pengguna lain

- **Kekurangan:**
  - Terbatas pada fitur yang tersedia (genre)
  - Tidak dapat menangkap preferensi tersembunyi
  - Cenderung menghasilkan rekomendasi yang kurang beragam

**Collaborative Filtering:**
- **Kelebihan:**
  - Dapat menemukan preferensi tersembunyi yang tidak terlihat dari fitur film
  - Memberikan rekomendasi yang lebih personal dan beragam
  - Mampu menemukan film yang tidak terkait secara genre tetapi disukai oleh pengguna serupa

- **Kekurangan:**
  - Memerlukan data rating yang cukup untuk pengguna dan film (cold start problem)
  - Komputasi lebih kompleks dan memerlukan waktu pelatihan
  - Hasil rekomendasi sulit dijelaskan (black box)

## Evaluation

### 1. Metrik Evaluasi

Untuk mengevaluasi model rekomendasi, saya menggunakan Mean Squared Error (MSE). MSE mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai sebenarnya:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Di mana:
- $n$ adalah jumlah prediksi
- $y_i$ adalah nilai rating sebenarnya
- $\hat{y}_i$ adalah nilai rating yang diprediksi

MSE memberikan bobot lebih pada error yang besar karena mengkuadratkan selisih. Nilai MSE yang lebih rendah menunjukkan performa model yang lebih baik.

### 2. Hasil Evaluasi

**Content-Based Filtering:**
MSE Content-Based: 0.03

**Collaborative Filtering:**
MSE Collaborative Filtering: 1.39

### 3. Analisis Hasil

Dari hasil evaluasi, Content-Based Filtering menunjukkan MSE yang sangat rendah (0.03) dibandingkan dengan Collaborative Filtering (1.39). Ini menunjukkan bahwa model berbasis konten memberikan prediksi rating yang lebih akurat secara numerik.

Namun, perlu dicatat bahwa MSE yang rendah tidak selalu berarti rekomendasi yang lebih baik dari perspektif pengguna. Content-Based Filtering memang unggul dalam memprediksi rating berdasarkan kesamaan genre, tetapi cenderung merekomendasikan film yang serupa dan tidak beragam.

Collaborative Filtering, meskipun memiliki MSE lebih tinggi, mampu menemukan preferensi tersembunyi dan memberikan rekomendasi yang lebih personal dan beragam. Dalam praktiknya, sistem hybrid yang menggabungkan kekuatan kedua pendekatan sering menjadi solusi terbaik.

### 4. Metrik Evaluasi Tambahan

Selain MSE, beberapa metrik lain yang dapat dipertimbangkan untuk evaluasi sistem rekomendasi:

- **Root Mean Squared Error (RMSE)**: Akar kuadrat dari MSE
- **Mean Absolute Error (MAE)**: Rata-rata nilai absolut error
- **Precision@k**: Persentase item relevan di antara k rekomendasi teratas
- **Recall@k**: Persentase item relevan yang berhasil direkomendasikan
- **Mean Average Precision (MAP)**: Rata-rata precision pada setiap tingkat recall
- **Normalized Discounted Cumulative Gain (NDCG)**: Mengukur kualitas ranking dengan bobot lebih pada item di posisi atas

## Kesimpulan

Proyek ini berhasil mengembangkan dua pendekatan sistem rekomendasi film yang berbeda: Content-Based Filtering dan Collaborative Filtering. Kedua pendekatan ini memiliki kelebihan dan kekurangan masing-masing.

Content-Based Filtering unggul dalam akurasi prediksi rating berdasarkan kesamaan genre, dengan MSE hanya 0.03. Pendekatan ini cocok untuk mengatasi masalah cold start untuk film baru dan memberikan rekomendasi yang dapat dijelaskan.

Collaborative Filtering, meskipun memiliki MSE lebih tinggi (1.39), mampu mengungkap preferensi tersembunyi dan memberikan rekomendasi yang lebih beragam. Pendekatan ini lebih sesuai untuk pengguna yang sudah memiliki riwayat rating yang cukup.

Untuk pengembangan lebih lanjut, pendekatan hybrid yang menggabungkan kedua metode dapat diimplementasikan untuk mengoptimalkan akurasi prediksi dan kualitas rekomendasi. Selain itu, penggunaan metrik evaluasi tambahan seperti Precision@k dan Recall@k dapat memberikan perspektif yang lebih komprehensif tentang performa sistem rekomendasi.

## References

[1] McKinsey & Company. (2013). "How retailers can keep up with consumers." McKinsey Quarterly, October 2013. Retrieved from https://www.mckinsey.com/industries/retail/our-insights/how-retailers-can-keep-up-with-consumers
