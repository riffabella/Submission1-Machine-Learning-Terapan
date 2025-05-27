# Laporan Proyek Machine Learning - Riffa Bella Wahyu S
## Domain Proyek 
  Rumah merupakan kebutuhan dasar manusia yang tidak hanya berfungsi sebagai tempat tinggal, tetapi juga sebagai tempat berkumpul dan beristirahat (Iskandar, 2020). Karena pentingnya fungsi rumah, banyak orang ingin membeli atau menjual properti, yang menuntut adanya penentuan harga rumah secara tepat.

Menentukan harga rumah bukan hal mudah, karena dipengaruhi berbagai faktor seperti lokasi, luas tanah, jumlah kamar, dan fasilitas sekitar (Yuliani & Firmansyah, 2021). Penilaian secara manual sering memakan waktu, subjektif, dan rawan kesalahan. Untuk itu, dibutuhkan solusi berbasis teknologi yang cepat, akurat, dan objektif.

Machine learning telah terbukti mampu memprediksi harga rumah berdasarkan data historis dan atribut properti. Studi oleh Purwanto & Putra (2023) menunjukkan bahwa algoritma Random Forest memberikan hasil akurat dengan risiko overfitting rendah. Sementara itu, regresi linear dan ridge regression juga terbukti efektif dalam studi Abdurrahman & Kurniawan (2023).

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
  
2. Memeriksa jumlah nilai yang hilang di setiap kolom
   ```print(House.isnull().sum())```
   Terlihat dibawah ini, bahwa tidak ada nilai yang hilang pada setiap kolom 
   ![Image](https://github.com/user-attachments/assets/d6e74a66-5954-4ac3-a528-96084721ae70)
   
3. Menampilkan visualisasi berdasarkan fitur Numerik dan Kategorikal
   - Fitur Kategorikal
     - Fitur Location
```
feature = categorical_features[0]
count = House[feature].value_counts()
percent = 100*House[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```
![Image](https://github.com/user-attachments/assets/0bf9e89f-8934-48b2-af00-1b0ad1bc0d84)
Terdapat 4 kategori pada fitur Location, secara berurutan dari jumlah yang paling banyak yaitu: Downtown, Urban, Suburban, Rural. Dari data presentase dapat disimpulkan bahwa lebih dari 75% sampel merupakan rumah yang berada di lokasi Downtown, Urban, dan Suburban.

     - Fitur Condition
```
feature = categorical_features[1]
count = House[feature].value_counts()
percent = 100*House[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```
![Image](https://github.com/user-attachments/assets/1c3bb4e3-a2aa-41d8-9d2c-e886642f9b41)
Terdapat 4 kategori pada fitur Condition, secara berurutan dari jumlah yang paling banyak yaitu Fair, Excellent, Poor, dan Good. Dengan mayoritas rumah dalam sampel memiliki kondisi menengah ke atas yaitu Fair (26%).

     - Fitur Garage
```
feature = categorical_features[2]
count = House[feature].value_counts()
percent = 100*House[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```
   - Fitur Numerik
```
sns.countplot(x=labels)
plt.title("Distribusi Gambar Tiap Kelas")
```
![Image](https://github.com/user-attachments/assets/7831242b-4f5f-415e-bf33-606dca3e3dcb)

## Data Preparation
- Data Balencing, dengan melakukan augmentasi pada data yang minoritas seperti kelas Potato_healthy agar seimbang dengan kelas lainnya. Diperlukan proses tersebut agar dapat memperbaiki kualitas distribusi data, dan mampu meningkatkan performa model terutama pada kelas minoritas.
```
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1
)
```
- Splitting Data, dengan membagi dataset menjadi tiga bagian menggunakan train_test_split, yaitu
train 80%, validation 10% dan test 10% untuk dilakuakn training model.
```
# Split data: 80% train, 20% temp
X = df['path']
y = df['labels']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=300)

# Bagi 20% temp menjadi 50:50 → 10% val, 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=300)
```
- Resize dan Normalisasi, dengan melakukan rezising gambar menjadi (224,224) dan membagi piksel dengan 255.0
```
img = img.resize((224, 224))
img = np.array(img)/255.0
```
- Data Augmentasi, digunakan untuk membuat objek augmentasi gambar secara real-time saat pelatihan model, sehingga akan memperbanyak dan memvariasikan data gambar secara otomatis. Dilakukan proses ini, agar model tidak overfitting, dan mampu melakukan generalisasi lebih baik.
```
augmentor = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest"
)

AUGMENT_PER_IMAGE = 3
```
## Modeling
Dalam proyek ini, menggunakan dua algoritma untuk mendeteksi penyakit kentang berdasarkan daun, dengan mengklasifikasikan kondisi daun kentang menjadi tiga kelas, yaitu **Early blight**, **Late blight**, dan **Healthy**. Untuk menyelesaikan permasalahan ini, digunakan pendekatan transfer learning dengan memanfaatkan dua arsitektur deep learning populer, yaitu **DenseNet201** dan **MobileNetV2** yang telah dilatih sebelumnya pada dataset ImageNet.

**Tahapan Pemodelan**
1. Pra-Pemrosesan Data
- Semua gambar diubah ukurannya menjadi `224x224`.
- Augmentasi citra dilakukan untuk meningkatkan variasi data.
- Data dibagi menjadi tiga bagian: train, validation, dan test.

2. Model yang Digunakan
**DenseNet201**
- Pretrained: `ImageNet`
- Layer tambahan:
  - Conv2D + MaxPooling2D
  - Flatten → Dropout(0.5)
  - Dense(128) → Dense(64) → Dense(3, softmax)
- Optimizer: `Adam(1e-4)`
- Callbacks:
  - `EarlyStopping(patience=5)`
  - `ReduceLROnPlateau(factor=0.5, patience=2)`
- Fine-tuning: Unfreeze 30 layer terakhir

**MobileNetV2**
- Pretrained: `ImageNet`
- Layer tambahan:
  - Conv2D → GlobalAveragePooling2D
  - Dense(128, L2) + Dropout(0.5)
  - Dense(64) + Dropout(0.4)
  - Dense(3, softmax)
- Optimizer:
  - Awal: `Adam(1e-4)`
  - Fine-tuning: `Adam(1e-5)`
- Callbacks:
  - `EarlyStopping(patience=5)`
  - `ReduceLROnPlateau(factor=0.2, patience=3)`
- Fine-tuning: Unfreeze 30 layer terakhir

3. Perbandingan Model

| Kriteria         | DenseNet201                      | MobileNetV2                      |
|------------------|----------------------------------|----------------------------------|
| Akurasi Latih     | Tinggi (mendekati 100%)          | Tinggi                           |
| Akurasi Validasi  | Fluktuatif, indikasi overfitting | Stabil dan konsisten             |
| Kompleksitas      | Berat dan lambat                | Ringan dan cepat                 |
| Ukuran Model      | Besar                            | Kecil dan efisien  


4. Model Terbaik
Dari dua arsitektur model CNN yang digunakan, Model MobileNetV2 menjadi model terbaik dan unggul dalam hal kestabilan, dan efisiensi. Sehingga model ini menjadi solusi terbaik untuk klasifikasi penyakit daun kentang . 
 
## Evaluation
Dalam proyek kalsifikasi citra penyakit kentang ini, digunakan beberapa metrik evaluasi untuk mengukur performa model yaitu:
1. **Accuarcy**, digunakan untuk mengukur prediksi yang benar terhadap seluruh prediksi. Pada proyek ini, accuracy yang didapat pada model yang terbaik yaitu MobileNetV2 mencapai 97.31%. Yang menunjukkan bahwa model mampu mengklasifikasikan sebgaian besar gambar dengan benar. Berikut rumusnya:

$$
Accuracy = \frac{Jumlah Prediksi Benar}{Total Jumlah Prediksi}
$$           

2. **Loss (_Categorical Crossentropy_)**, digunakan sebagai fungsi objektif, untuk klasifikasi multi-kelas. Bukan hanya itu loss ini juga penting untuk memandu proses pelatihan model dan membantu mencegah overfitting. Berikut rumusnya:

$$
Loss = -\sum_{i=1}^{N} y_i \cdot \log(\hat{y}_i)
$$

3.**Classification Report**, digunakan untuk menghasilkan laporan berbentuk tabel yang mencakup metrik evaluasi penting untuk setiap kelas dalam masalah klasifikasi, yaitu:
  - Precision, digunakan untuk mengukur ketepatan prediksi positif

$$
Precision = \frac{\text{True Positive}}{\text{True Positive} + \text{False Positive}}
$$

  - Recall, digunakan untuk mengukur seberapa baik model menemukan seluruh kasus positif

$$
Recall = \frac{\text{True Positive}}{\text{True Positive} + \text{False Negative}}
$$

  - F1-Score, digunakan untuk menghitung rata-rata dari precision dan recall

$$
F1 Score = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Berikut ini Classification Report per model:
### 📊 Classification Report (DenseNet201)

| Class                     | Precision | Recall | F1-Score | Support |
|---------------------------|-----------|--------|----------|---------|
| Potato___Early_blight     | 0.95      | 1.00   | 0.98     | 100     |
| Potato___Late_blight      | 1.00      | 0.95   | 0.97     | 100     |
| Potato__healthy_augmented| 1.00      | 1.00   | 1.00     | 97      |
| **Accuracy**              |           |        | **0.98** | **297** |
| **Macro avg**             | 0.98      | 0.98   | 0.98     | 297     |
| **Weighted avg**          | 0.98      | 0.98   | 0.98     | 297     |

### 📊 Classification Report (MobileNetV2)

| Class                      | Precision | Recall | F1-Score | Support |
|----------------------------|-----------|--------|----------|---------|
| Potato___Early_blight      | 0.9615    | 1.0000 | 0.9804   | 100     |
| Potato___Late_blight       | 1.0000    | 0.9200 | 0.9583   | 100     |
| Potato__healthy_augmented | 0.9604    | 1.0000 | 0.9798   | 97      |
|                            |           |        |          |         |
| **Accuracy**               |           |        | **0.9731** | 297   |
| **Macro Avg**              | 0.9740    | 0.9733 | 0.9728   | 297     |
| **Weighted Avg**           | 0.9741    | 0.9731 | 0.9728   | 297     |


4. **Confusion Matrix**, digunakan untuk menunjukkan jumlah prediksi benar dan salah untuk masing-masing kelas, membantu mengidentifikasi di mana model sering melakukan kesalahan klasifikasi.
- **Confusion Matrix (DenseNet201)**

![Image](https://github.com/user-attachments/assets/bca194d9-75ec-4382-9b0b-88cfa4174faa)
  
- **Confusion Matrix (MobileNetV2)**

![Image](https://github.com/user-attachments/assets/fb578b11-1ecb-4db7-8d1c-0de9a9ed9d90)

### Hasil Evaluasi 

| Model        | Training Accuracy | Validation Accuracy       | Final Loss    | Catatan             |
|--------------|-------------------|--------------------------|---------------|---------------------|
| DenseNet201  | 99.7%            | 98% (naik turun)     | Tidak stabil  | Cenderung overfitting|
| MobileNetV2  | 98.5%            | 97.3% (stabil)          | Konsisten     | Generalisasi lebih baik |

Berdasarkan hasil evaluasi, **MobileNetV2** menunjukkan performa klasifikasi yang lebih stabil dan mampu menghindari overfitting dibandingkan DenseNet201, menjadikannya pilihan model yang lebih optimal untuk digunakan dalam klasifikasi penyakit kentang ini.

## Referensi

[1] M. F. Nauval dan M. I. Habibie, "Implementasi Algoritma Convolutional Neural Network (CNN) dalam Mengidentifikasi Penyakit Daun Kentang Menggunakan Citra Digital", *Jurnal Teknologi dan Sistem Informasi*, vol. 4, no. 2, pp. 116–122, 2023.

[2] N. Amatullah, R. Rizal, dan I. Irawan, "Identifikasi Penyakit Daun Kentang Berdasarkan Citra Digital Menggunakan Metode CNN (Convolutional Neural Network)", *Jurnal Riset Ilmu Komputer dan Aplikasinya (JRIKA)*, vol. 8, no. 1, pp. 73–79, 2023.

[3] F. Maulana, A. Rachman, dan D. Wicaksono, "Analisis Dampak Penyakit Hawar Daun Terhadap Produksi Kentang di Dataran Tinggi Dieng", Jurnal Hortikultura Indonesia, vol. 12, no. 1, pp. 45–52, 2024.

[4] B. Prasetyo dan R. Mahenra, "Efektivitas Deteksi Penyakit Tanaman Menggunakan Teknologi Citra Digital dan Deep Learning", Jurnal Teknologi Pertanian Terapan, vol. 9, no. 2, pp. 88–96, 2025.
