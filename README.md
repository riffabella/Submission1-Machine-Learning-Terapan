# Laporan Proyek Machine Learning - Riffa Bella Wahyu S
## Domain Proyek - House Price Predictive
  Rumah merupakan kebutuhan dasar manusia yang tidak hanya berfungsi sebagai tempat tinggal, tetapi juga sebagai tempat berkumpul dan beristirahat[1]. Karena pentingnya fungsi rumah, banyak orang ingin membeli atau menjual properti, yang menuntut adanya penentuan harga rumah secara tepat[2].

Menentukan harga rumah bukan hal mudah, karena dipengaruhi berbagai faktor seperti lokasi, luas tanah, jumlah kamar, dan fasilitas sekitar[3]. Penilaian secara manual sering memakan waktu, subjektif, dan rawan kesalahan. Untuk itu, dibutuhkan solusi berbasis teknologi yang cepat, akurat, dan objektif.

Machine learning telah terbukti mampu memprediksi harga rumah berdasarkan data historis dan atribut properti. Studi oleh Purwanto & Putra (2023) menunjukkan bahwa algoritma Random Forest memberikan hasil akurat dengan risiko overfitting rendah[4]. Sementara itu, regresi linear dan ridge regression juga terbukti efektif dalam studi Nuzuliarini (2024)[5].

Tanpa sistem prediksi yang andal, pembeli dan penjual berisiko membuat keputusan yang salah. Oleh karena itu, proyek ini bertujuan membangun model prediksi harga rumah menggunakan algoritma K-Nearest Neighbor, Random Forest, dan Boosting, dengan evaluasi pada dataset dari [Kaggle](https://www.kaggle.com/datasets/zafarali27/house-price-prediction-dataset/data), yang digunakan untuk membandingkan kinerja perfoma ketiga algoritma ML dalam memprediksi harga rumah, sesuai dengan fitur yang telah disediakan.

## Business Understanding

### Problem Statements
1. **Pernyataan Masalah 1**
   Penentuan harga rumah masih sering dilakukan secara manual, yang bersifat subjektif dan kurang efisien, sehingga rawan menyebabkan kesalahan dalam memprediksi harga rumah.
2. **Pernyataan Masalah 2**
   Harga rumah dipengaruhi oleh banyak faktor, seperti lokasi, luas bangunan, jumlah kamar, dan fasilitas di sekitarnya, yang membuat proses estimasi harga menjadi kompleks jika tidak dibantu teknologi.

### Goals
1. Membangun model machine learning untuk memprediksi harga rumah berdasarkan fitur-fitur seperti lokasi, luas, jumlah kamar, dan atribut lainnya yang tersedia dalam dataset.
2. Mengevaluasi dan membandingkan performa beberapa algoritma machine learning dalam hal akurasi prediksi harga rumah.

### Solution Statement
1. **Solusi 1 : Penerapan Algoritma Machine Learning**
   Dengan membangun model ML menggunakan algoritma Random Forest, dan Boosting Algorithm dalam memprediksi harga rumah, dengan membandingkan kedua model algoritma machine learning untuk mendapatkan model yang terbaik. 
3. **Solusi 2 : Penerapan Hyperparameter dengan GridSearchCV**
   Melakukan optimasi terhadap hyperparameter dengan GridSearchCV untuk meningkatkan akurasi prediksi dan mengurangi kesalahan prediksi maupun overfitting dalam pelatihan pertama.
   
## Data Understanding
Pada proyek ini, data yang digunakan adalah _House Price Prediction Dataset_ yang tersedia secara publik di[Kaggle](https://www.kaggle.com/datasets/zafarali27/house-price-prediction-dataset) Dengan informasi dataset sebagai berikut :
1. Jumlah data : 2000 baris
2. Jumlah fitur : 10 kolom (termasuk target)
3. Tipe data : Numerik dan Kategorikal
4. Target/Lable : Price (harga rumah)

### Variabel-variabel House Price Prediction pada sebagai berikut :
- Area : Luas bangunan rumah (dalam persegi) yang merupakan salah satu faktor paling penting dalam memprediksi harga.
- Badrooms dan Bathrooms : Jumlah kamar (kamar tidur dan kamat mandi) dalam sebuah rumah sangat mempengaruhi nilainya. Rumah dengan lebih banyak kamar cenderung memiliki harga yang lebih tinggi.
- Floors : Jumlah lantai pada rumah dapat menunjukkan bahwa rumah tersebut lebih besar atau mewah, sehingga berpotensi meningkatkan harganya.
- Year Built : usia rumah dapat memepengaruhi kondisi dan nilainya. Rumah yang baru dibangun umumnya lebih mahal dibandingkan rumah yang lebih tua.
- Location : Rumah yang berasa di lokasi strategis seperti pusat kota atau area urban cenderung memiliki harga lebih tinggi dibandingkan rumah di daerah pinggiran atau pedesaan.
- Condition : Kondisi rumah sangat penting. Rumah yang terawat dengan baik (dalam kondisi "Excellent" atau "Good") akan memiliki harga jual lebih tinggi dibandingkan rumah dalam kondisi "Fiar" atau "Poor"
- Garage : Ketersediaan garansi dapat meningkatkan harga rumah karena memberikan kenyamanan dan ruang tambahan
- Price : Variabel target, yaitu harga jual rumah, yang digunakan untuk melatih model machine learning agar dapat memprediksi harga rumah berdasarkan fitur-fitur lainnya.

### Tahapan yang dilakukan untuk memahami data 
**EDA (_Exploratory Data Analysis_)**
1. Menampilkan Jumlah Baris dan Kolom serta Jenis Data setiap Kolom
    ```House.info()```
   Terlihat dibawah ini, bahwa terdapat 7 kolom numerikal, dan 3 kolom kategorikal dibawah ini
   
   ![Image](https://github.com/user-attachments/assets/f560b79f-4fa6-490d-b700-36ea8b8a0c12)
  
3. Memeriksa jumlah nilai yang hilang di setiap kolom
   ```print(House.isnull().sum())```
   Terlihat dibawah ini, bahwa tidak ada nilai yang hilang pada setiap kolom
   
   ![Image](https://github.com/user-attachments/assets/d6e74a66-5954-4ac3-a528-96084721ae70)
   
5. Menampilkan visualisasi berdasarkan fitur Numerik dan Kategorikal

- Fitur Kategorikal
1. Fitur Location

```
feature = categorical_features[0]
count = House[feature].value_counts()
percent = 100*House[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```

![Image](https://github.com/user-attachments/assets/18166e0e-0051-43f7-8cdb-a1197f5bd8c3)

Terdapat 4 kategori pada fitur Location, secara berurutan dari jumlah yang paling banyak yaitu: Downtown, Urban, Suburban, Rural. Dari data presentase dapat disimpulkan bahwa lebih dari 75% sampel merupakan rumah yang berada di lokasi Downtown, Urban, dan Suburban.


  2. Fitur Condition


```
feature = categorical_features[1]
count = House[feature].value_counts()
percent = 100*House[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```

![Image](https://github.com/user-attachments/assets/c4e63a5c-399a-4bf8-a877-4053f538cb75)


Terdapat 4 kategori pada fitur Condition, secara berurutan dari jumlah yang paling banyak yaitu Fair, Excellent, Poor, dan Good. Dengan mayoritas rumah dalam sampel memiliki kondisi menengah ke atas yaitu Fair (26%).


  3. Fitur Garage


```
feature = categorical_features[2]
count = House[feature].value_counts()
percent = 100*House[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```

![Image](https://github.com/user-attachments/assets/13e2521b-7d8a-48ca-951e-6a503d5e6d82)


```
plt.figure(figsize=(6, 6))
House['Garage'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen','lightcoral'])
plt.title('Distribusi Garage')
plt.ylabel('')  # Menghilangkan label default
plt.show()
```

![Image](https://github.com/user-attachments/assets/6c26b9c0-064c-4189-93c4-c7483a1ed34d)


Terdapat 2 kategori pada fitur Garage, secara berurutan dari jumlah yang paling banyak yaitu No, dan Yes. Dengan mayoritas rumah pada sampel tidak memiliki garansi didalamnya.


- Fitur Numerik
     Terdiri dari fitur Id, Area, Bedrooms, Bathrooms, Floors, YearBuilt, dan Price

     
```
House.hist(bins=50, figsize=(20,15))
plt.show()
```

![Image](https://github.com/user-attachments/assets/d90f886a-a817-4e2f-99ea-906f4efbdbad)


## Data Preparation
- Melakukan **Encoding Fitur Kategori**, dengan mengubah data kategori menjadi bentuk numerik agar dapat diproses oleh algoritma machine learning. Metode Encoding yang digunakan adalah One-Hot Encoding, dengan membuat kolom baru untuk setiap nilai kategori dengan nilai biner.

```
from sklearn.preprocessing import  OneHotEncoder

# Mengubah data kategorikal menjadi numerik
House = pd.concat([House, pd.get_dummies(House['Location'], prefix='Location', dtype='int64')],axis=1)
House = pd.concat([House, pd.get_dummies(House['Condition'], prefix='Condition',dtype='int64')],axis=1)
House = pd.concat([House, pd.get_dummies(House['Garage'], prefix='Garage', dtype='int64')],axis=1)
House.drop(['Location','Condition','Garage'], axis=1, inplace=True)
House.head()
```

| Id | Area | Bedrooms | Bathrooms | Floors | YearBuilt |  Price  | Location_Downtown | Location_Rural | Location_Suburban | Location_Urban | Condition_Excellent | Condition_Fair | Condition_Good | Condition_Poor | Garage_No | Garage_Yes |
|----|------|----------|-----------|--------|------------|---------|-------------------|----------------|-------------------|----------------|----------------------|----------------|----------------|----------------|-----------|------------|
|  1 | 1360 |        5 |         4 |      3 |       1970 | 149919  |                 1 |              0 |                 0 |              0 |                    1 |              0 |              0 |              0 |         1 |          0 |
|  2 | 4272 |        5 |         4 |      3 |       1958 | 424998  |                 1 |              0 |                 0 |              0 |                    1 |              0 |              0 |              0 |         1 |          0 |
|  3 | 3592 |        2 |         2 |      3 |       1938 | 266746  |                 1 |              0 |                 0 |              0 |                    0 |              0 |              1 |              0 |         1 |          0 |
|  4 |  966 |        4 |         2 |      2 |       1902 | 244020  |                 0 |              0 |                 1 |              0 |                    0 |              1 |              0 |              0 |         0 |          1 |
|  5 | 4926 |        1 |         4 |      2 |       1975 | 636056  |                 1 |              0 |                 0 |              0 |                    0 |              1 |              0 |              0 |         0 |          1 |


- Splitting Data, dengan membagi dataset menjadi dua bagian menggunakan ```train_test_split```  yaitu 80% untuk data pelatihan dan 20% untuk data pengujian.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

![Image](https://github.com/user-attachments/assets/8146cd94-4176-480b-a6e6-a13987ffd31c)

- Melakukan **Standarisasi**, dengan mengubah nilai rata-rata (mean) menjadi 0 dan nilai standar deviasi menjadi 1. Yang digunakan untuk membuat semua fitur berada dalam skala yang sama, dan menghindari bias dalam perhitungan jarak atau bobot. Dengan menerapkan StandarScaler pada data sebagai berikut : ```scaler = StandardScaler()```

![Image](https://github.com/user-attachments/assets/d4992f5a-c736-49a1-b630-0ed4ec0d0439)

## Modeling
Dalam proyek ini, digunakan dua algoritma untuk memprediksi harga rumah berdasarkan fitur numerik dan kategorikal, seperti luas area, jumlah kamar, lokasi, kondisi bangunan, jumlah lantai, ketersediaan garasi, dan tahun pembangunan. Untuk menyelesaikan permasalahan ini, digunakan dua algoritma machine learning, yaitu Random Forest, dan Gradient Boosting, yang dievaluasi untuk membandingkan performa prediksi harga secara akurat.

**Tahapan Pemodelan**
1. Mempersiapkan dataframe untuk analisis model, yang berisi index = train_mse, dan test_mse untuk metrix evaluasi pelatihan pertama, dan berisi columns algoritma RandomForest dan Gradient Boosting
   
2. Membuat model:
   - Pada model **Random Forest**, model akan membuat 50 pohon keputusan, jumlah ini dipilih untuk menjaga keseimbangan antara akurasi dan kecepatan. Setiap pohon akan dibatasi hingga kedalaman 8 tingkat, ini mencegah model menjadi teralu rumit dan mengurangi resiko overfitting. Kemudian setiap cabang pohon harus memiliki minimal 10 data untuk bisa dibagi lagi, sehingga membantu mencegah pembagian yang terlalu kecil. Setelah itu setiap daun (akhir pohon), harus memiliki minimal 6 data, untuk memastikan prediksi lebih stabil. Kemudian menerapakan random state=55 untuk memastikan hasil model selalu sama setiap kali dijalankan, sehingga hasilnya dapat diulang.
   - Pada model **Gradient Boosting**, model ini disetting dengan menggunakan 100 pohon keputusan untuk membuat prediksi, dengan setiap pohon keputusan dibatasi hingga 3 tingkat kedalaman, agar model tidak terlalu rumit, kemudian dilanjutkan menggunakan random_state=55, untuk memastikan hasil model konsisten setiap kali dijalankan.
  
**Parameter yang Digunakan**
- **Random Forest:**
  - ```n_estimators=50``` : Jumlah pohon dalam hutan.
  - ```max_depth=8``` : Kedalaman maksimum setiap pohon.
  - ```min_samples_split=10``` : Jumlah minimum sampel yang diperlukan untuk membagi node internal.
  - ```min_samples_leaf=6``` : Jumlah minimum sampel yang diperlukan pada daun pohon.
  - ```random_state=55``` : digunakan untuk mengontrol random number generator yang digunakan. 
  - ```n_jobs=-1``` : jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel.
- **Gradient Boosting:**
  - ```n_estimators=0.05``` : Jumlah tahap boosting yang dilakukan
  - ```learning_rate=100``` : Kontribusi setiap pohon dalam ensemble.
  - ```max_depth=3``` : Kedalaman maksimum pohon individu.
  - ```random_state=55``` : digunakan untuk mengontrol random number generator yang digunakan.

- **Tuning Hyperarameter:**
**Random Forest:**
    
```
# Definisikan parameter grid untuk Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 6, 8, 10],
    'min_samples_split': [10, 15, 20],
    'min_samples_leaf': [5, 8, 10]
}    
```

**Gradient Boosting:**

    
```
# Definisikan parameter grid untuk Gradient Boosting
gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'min_samples_split': [10, 15, 20],
    'min_samples_leaf': [5, 8, 10],
    'learning_rate': [0.01, 0.1, 0.2]
}
```


**Perbandingan Model**
### Kelebihan dan Kekurangan dari Setiap Algoritma

| Algoritma           | Kelebihan                                                                                                              | Kekurangan                                                                                          |
|---------------------|------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Random Forest**   | - Mengurangi overfitting dibandingkan pohon tunggal  <br> - Menangani fitur hilang dan variabel kategorikal  <br> - Estimasi fitur penting | - Kurang interpretatif  <br> - Konsumsi memori tinggi  <br> - Potensi overfit jika terlalu banyak pohon |
| **Gradient Boosting** | - Akurasi prediksi tinggi  <br> - Fleksibel terhadap berbagai fungsi loss  <br> - Cocok untuk data kompleks            | - Rentan overfitting tanpa tuning  <br> - Latihannya lebih lambat  <br> - Perlu tuning parameter yang teliti |

**Model Terbaik**
Dari dua model algoritma ML yang telah dibangun, hingga dilakukan proses tuning hyerparameter dengan GridSearchCV, menghasilkan model yang cukup baik dalam prediksi harga rumah yaitu pada model algoritma Gradient Boosting.  
 
## Evaluation
Dalam proyek ini, menggunakan tiga metriks evaluasi untuk mengukur kinerja model seperti Mean Absolute Error (MAE), Mean Squared Error (MSE), dan R^2 Score. Metrik ini dipilih untuk mengevaluasi performa model Random Forest dan Gradient Boosting dalam memprediksi harga rumah berdasarkan data yang diberikan. 
1. **MAE**, dipilih karena memberikan gambaran rkesalahan rata-rata dalam dolar, yang mudah diinterpretasikan dan relevan untuk mengatasi prediksi subjektif yang rawan kesalahan. Sehingga dapat mengukur rata-rata kesalahan absolut antara nilai prediksi dan nilai aktual. Semakin kecil nilai MAE, semakin baik performa model. Berikut formulanya:

$$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$
   
2. **MSE**, digunakan untuk menghitung rata-rata kuadrat dari selisih antara nilai prediksi dan aktual. MSE memberi penalti lebih besar pada kesalahan besar. Sehingga membantu mendeteksi kesalahan besar dan selaras dengan proses tuning hyperparameter menggunakan GridSearchCV, serta cocok untuk mengevaluasi model dalam konteks data yang kompleks. Nilai yang lebih kecil menunjukkan model lebih baik. Berikut formulanya:

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 $$   
   
3. **R^2**, digunakan untuk mengukur proporsi variasi pada target (harga rumah) yang dapat dijelaskan oleh fitur. Nilai R² mendekati 1 menunjukkan model sangat baik, sedangkan nilai negatif menunjukkan kinerja yang lebih buruk daripada model rata-rata. Sehingga mampu menunjukkan seberapa baik model menangkap hubungan kompleks antara fitur dan harga rumah, serta memastikan model lebih efisien dan objektif dibandingkan metode manual. Berikut formulanya:

$$ R^2 = 1 - \frac{\sum{(y_i - \hat{y}_i)^2}}{\sum{(y_i - \bar{y})^2}} $$


### Hasil Proyek Setelah Melakukan Tuning Hyperparameter

**Visualisasi yang dihasilkan**

![Image](https://github.com/user-attachments/assets/e4da1505-cf63-49db-a35a-704767a5525b)

Diagram berikut memperlihatkan perbandingan Mean Squared Error (MSE) antara dua model yang telah dilakukan tuning, yaitu Random Forest (RF) dan Gradient Boosting (Boosting).

- **Sumbu X**: Nilai MSE (semakin kecil semakin baik)
- **Sumbu Y**: Model
- **Biru**: MSE pada data training
- **Oranye**: MSE pada data testing

**Insight:**
- Boosting memiliki MSE yang sedikit lebih rendah dibanding RF, baik di train maupun test set.
- Perbedaan MSE train dan test relatif kecil, menunjukkan overfitting tidak terlalu parah.
- Namun, berdasarkan R² yang negatif, kedua model masih belum mampu memodelkan variansi target secara baik.

Sehingga Boosting (Algoritma Gradient Boosting) menunjukkan performa sedikit lebih baik dari RF, namun masih perlu perbaikan model atau fitur.

**Hasil Metriks Evaluasi**

| Model    | MSE          | RMSE      | MAE       | R²    |
| -------- | ------------ | --------- | --------- | ----- |
| RF       | 7.924159e+10 | 281498.83 | 244792.79 | -0.02 |
| Boosting | 7.824462e+10 | 279722.40 | 243382.47 | -0.01 |


**Penjelasan Hasil Metriks Evaluasi** :
- MSE (Mean Squared Error) : Boosting memiliki MSE sedikit lebih rendah dibandingkan RF (selisih ~0.1 miliar), menunjukkan bahwa Boosting sedikit lebih baik dalam mengurangi kesalahan kuadrat rata-rata.
- MAE (Mean Absolute Error) : Boosting memiliki MAE sedikit lebih rendah, artinya prediksinya sedikit lebih dekat ke nilai sebenarnya dibanding RF.
- R2 Score : Kedua model bernilai negatif, menunjukkan model tidak cocok dan lebih buruk dari rata-rata.
  
Berdasarkan hasil evaluasi, **Algoritma Gradient Boosting** sedikit lebih baik dalam MSE, dan MAE, keduanya masih buruk karena nilai R² negatif. Ini menunjukkan bahwa model belum berhasil memprediksi harga rumah secara efektif, dan perbaikan seperti feature engineering atau tuning lebih lanjut sangat disarankan.

## Referensi

[1] A. Iskandar, “Fungsi Rumah dalam Kehidupan Manusia Modern,” Jurnal Sosial dan Kemanusiaan, vol. 15, no. 2, pp. 100–110, 2020.

[2] N. Hadi and J. Benedict, “Implementasi Machine Learning untuk Prediksi Harga Rumah Menggunakan Algoritma Random Forest,” Computer: Jurnal Computer Science and Information Systems, vol. 8, no. 1, pp. 50–61, 2024.

[3] T. Yuliani and R. Firmansyah, “Faktor-Faktor yang Mempengaruhi Harga Properti,” Jurnal Ekonomi dan Bisnis, vol. 9, no. 1, pp. 25–34, 2021.

[4] D. Purwanto and R. Putra, “Prediksi Harga Rumah dengan Random Forest,” Jurnal Teknologi Informasi, vol. 12, no. 1, pp. 45–55, 2023.

[5] N. Nuris, “Analisis Prediksi Harga Rumah pada Machine Learning Menggunakan Metode Regresi Linear,” EXPLORE, vol. 14, no. 2, pp. 1–10, 2024.
