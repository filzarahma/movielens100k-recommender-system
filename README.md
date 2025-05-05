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
3. Mengevaluasi dan membandingkan performa kedua sistem rekomendasi tersebut menggunakan metrik evaluasi yang sesuai.

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

![image](https://github.com/user-attachments/assets/ddb0b53e-5f42-4463-b259-7c432b19025b)


Produksi film meningkat secara signifikan sejak tahun 1980-an dan mencapai puncaknya sekitar tahun 2000-an. Grafik ini mencerminkan tren pertumbuhan industri perfilman modern yang pesat dalam beberapa dekade terakhir.

#### 5. Analisis Rating Berdasarkan Genre

![image](https://github.com/user-attachments/assets/a58b8a48-507d-48fd-8a1e-c14261746a59)

Genre Film-Noir menempati posisi teratas dengan rata-rata rating tertinggi, disusul oleh War dan Documentary. Sebaliknya, genre Horror, Comedy, dan Children berada di posisi terbawah. Film dengan genre serius cenderung lebih dihargai dibanding genre yang lebih ringan.

## Data Preparation

Dalam tahap data preparation, saya melakukan beberapa transformasi dan pembersihan data untuk memastikan data siap digunakan dalam pengembangan model. Berikut adalah langkah-langkah yang dilakukan:

### 1. Eksplorasi Dasar dan Pemeriksaan Data

Tahapan ini penting untuk memastikan kualitas data yang akan digunakan. Hasil menunjukkan bahwa kedua dataset (film dan rating) tidak memiliki data duplikat atau nilai yang hilang, sehingga tidak diperlukan penanganan khusus untuk masalah tersebut.

### 2. Feature Engineering

Tahapan feature engineering dilakukan untuk memperkaya informasi dalam dataset:

- **Ekstraksi Tahun**: Mengekstrak tahun rilis film dari judul menggunakan ekspresi reguler. Fitur ini memungkinkan analisis berdasarkan periode waktu dan melihat tren historis.
  
- **Penambahan Rating Rata-rata**: Menghitung rating rata-rata dan jumlah rating untuk setiap film, yang memberikan informasi tentang popularitas dan penerimaan film.
  
- **Perhitungan Jumlah Genre**: Menambahkan fitur yang menunjukkan berapa banyak genre yang dimiliki oleh setiap film, yang dapat mengindikasikan kompleksitas atau keberagaman film.

Transformasi ini penting untuk analisis yang lebih mendalam dan meningkatkan kualitas fitur yang tersedia untuk model rekomendasi.

## Modeling and Result

Dalam proyek ini, saya mengembangkan dua model sistem rekomendasi: Content-Based Filtering dan Collaborative Filtering.

### 1. Content-Based Filtering

Model ini merekomendasikan film berdasarkan kesamaan genre dengan film yang sudah disukai pengguna. Implementasinya sebagai berikut:

```python
# TF-IDF pada data genre film
tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

# Menghitung similaritas kosinus antara semua film
cosine_sim = cosine_similarity(tfidf_matrix)

def recommend_content(title, movies, cosine_sim=cosine_sim, movie_indices=movie_indices):
    # Ambil index film input
    idx = movie_indices[title]
    
    # Hitung kemiripan
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # 10 teratas (selain dirinya sendiri)
    
    # Ambil informasi film rekomendasi
    results = []
    for i, score in sim_scores:
        row = movies.iloc[i]
        results.append({
            "Title": row['title'],
            "Genres": row['genres'],
            "Similarity": round(score, 3)
        })
    
    return pd.DataFrame(results)
```

**Hasil Top-N Rekomendasi untuk "Toy Story (1995)":**

![image](https://github.com/user-attachments/assets/a81618c3-0e11-4681-a2f4-7c2612d58c8c)


Model ini juga dapat digunakan untuk memberikan rekomendasi personalisasi berdasarkan seluruh riwayat rating pengguna:

```python
def recommend_for_user(user_id, movies, ratings, cosine_sim, top_n=10):
    # Gabungkan movies dan ratings
    user_data = ratings[ratings['userId'] == user_id]
    
    # Ambil film dengan rating tinggi
    liked_movies = user_data[user_data['rating'] >= 3.5]
    
    # Hitung kemiripan rata-rata antara film yang disukai user dengan semua film
    sim_scores = cosine_sim[liked_movie_indices]
    sim_scores = sim_scores.mean(axis=0)
    
    # Urutkan skor similarity
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Siapkan list rekomendasi (film yang belum ditonton)
    watched_movie_ids = set(user_data['movieId'])
    recommendations = []
    
    for idx, score in sim_scores:
        movie_id = movies.iloc[idx]['movieId']
        if movie_id not in watched_movie_ids:
            recommendations.append({
                'Title': movies.iloc[idx]['title'],
                'Genres': movies.iloc[idx]['genres'],
                'Similarity': round(score, 3)
            })
        if len(recommendations) >= top_n:
            break
    
    return pd.DataFrame(recommendations)
```
**Hasil Top-N Rekomendasi untuk userId 255:**

![image](https://github.com/user-attachments/assets/80169726-65b7-43f8-b445-cbb9f4deef92)


### 2. Collaborative Filtering

Model kedua menggunakan pendekatan deep learning untuk mempelajari pola rating dan memberikan rekomendasi berdasarkan preferensi pengguna lain dengan selera serupa:

```python
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size=50, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)
        self.movie_embedding = tf.keras.layers.Embedding(num_movies, embedding_size)
        self.dot = tf.keras.layers.Dot(axes=1)
        
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        return self.dot([user_vector, movie_vector])

# Kompilasi dan Pelatihan Model
model = RecommenderNet(num_users, num_movies)
model.compile(optimizer='adam', loss='mae', metrics=['mse'])

history = model.fit(x=x_train, y=y_train, 
                    validation_data=(x_test, y_test),
                    batch_size=64, epochs=10, verbose=1)
```

Fungsi untuk menghasilkan rekomendasi:

```python
def recommend_movies(user_id_original, model, movie_df, top_n=10):
    user_idx = user_to_index[user_id_original]
    
    # Dapatkan semua indeks film
    movie_indices = np.arange(len(movie_ids))
    
    # Buat kombinasi input antara pengguna dengan semua film
    user_input = np.array([[user_idx, movie] for movie in movie_indices])
    
    # Prediksi rating
    predicted_ratings = model.predict(user_input).flatten()
    
    # Urutkan berdasarkan rating
    top_indices = predicted_ratings.argsort()[-top_n:][::-1]
    top_movie_ids = [movie_ids[i] for i in top_indices]
    
    # Dapatkan judul film
    recommended_titles = movie_df[movie_df['movieId'].isin(top_movie_ids)]['title'].tolist()
    return recommended_titles
```

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
```python
mse_content = mean_squared_error(y_true, y_pred)
print("MSE Content-Based:", mse_content)
# MSE Content-Based: 0.03
```

**Collaborative Filtering:**
```python
y_test_pred = model.predict(x_test).flatten()
y_test_pred_scaled = y_test_pred * 5.0
y_test_scaled = y_test * 5.0
mse_collab = mean_squared_error(y_test_scaled, y_test_pred_scaled)
print("MSE Collaborative Filtering:", mse_collab)
# MSE Collaborative Filtering: 1.39
```

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
