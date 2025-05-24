# Laporan Proyek Machine Learning - Riffa Bella Wahyu S
## Domain Proyek 
  Kentang (_Solanum tuberosum_) merupakan salah satu komoditas hortikultura penting di Indonesia yang berperan sebagai sumber pangan dan pendapatan petani, khususnya di daerah dataran tinggi. Tanaman ini mengandung karbohidrat tinggi yang dibutuhkan manusia sebagai sumber energi utama [1]. Namun, dalam pembudidayaannya, tanaman kentang rentan terhadap berbagai serangan penyakit, terutama yang menyerang bagian daun.
  
  Dua jenis penyakit utama yang sering menyerang tanaman kentang adalah hawar daun (late blight) yang disebabkan oleh _Phytophthora infestans_ dan bercak kering (early blight) yang disebabkan oleh _Alternaria salani_. Kedua penyakit ini, kerap menyerang pada fase pertumbuhan vegetatif, yakni sekitar usia 5-6 minggu [2]. Penyakit ini, dapat menyebar cepat ke seluruh bagian tanaman termasuk batang dan umbi. Jika tidak ditangai secara dini, dapat menyebabkan kerusakan signifikan pada tanaman, mengurangi hasil panen, dan menimbulkan kerugian ekonomi bagi petani mencapai lebih dari 50% hasil panen [3]. 
  
  Deteksi dini terhadap penyakit ini sangat penting untuk mencegah penyebaran lebih lanjut dan mengurangi resiko kerugian. Namun, metode identifikasi penyakit secara manual oleh petani atau ahli pertanian membutuhkan pengalaman khusus, bersifat subjektif, serta tidak efisien ketika dilakukan pada skala pertanian yang luas [4]. Dengan perkembangan teknologi kecerdasan buatan (AI) dan deep learning saat ini, dapat memberikan peluang baru dalam pengembangan sistem deteksi penyakit tanaman berbasis citra. Salah satu pendekatan yang digunakan adalah penggunaan _Convolutional Neural Network_ (CNN), karena kemampuannya dalam mengenali pola visual dari gambar secara otomatis dan akurat.
  
  Proyek ini bertujuan untuk mengembangkan sistem kalsifikasi yang mampu mempermudah pekerjaan petani dalam mendeteksi gejala penyakit kentang lewat citra daun kentang. Dengan melalui proses identifikasi, yang terbagi menjadi tiga kategori, yaitu _healthy_ (daun sehat), _early blight_ (bercak kering), dan _late blight_ (hawar daun). Dalam melakukan identifikasi terhadap penyakit pada daun tanaman kentang, proyek ini menggunakan Arsitektur pada CNN yaitu MobileNetV2 dan DenseNet dalam pengklasifikasian gambar untuk mendeteksi penyakit pada tanaman kentang. Data yang dipergunakan pada proyek ini diperoleh dari dataset PlantVillage yang tersedia di situs _Kaggle_. 

## Business Understanding

### Problem Statements
1. **Pernyataan Masalah 1**
   Petani kesulitan dalam mengidentifikasi penyakit _Early Blight_ dan _Late Blight_ pada tanaman kentang secara akurat dan cepat karena gejala awal penyakit ini memiliki kemiripan visual yang membingungkan.
2. **Pernyataan Masalah 2**
   Proses identifikasi penyakit secara manual oleh petani bersifat subjektif, memerlukan keahlian khusus, dan tidak dapat dilakukan secara skala besar dalam waktu singkat.

### Goals
1. Mengembangkan sistem klasifikasi otomatis berbasis gambar daun kentang untuk membedakan antara potato___early_blight, potato___late_blight, dan potato___healthy dengan tingkat akurasi tinggi
2. Membangun model deep learning (CNN) yang dapat mengidentifikasi penyakit daun secara konsisten dan objektif tanpa ketergantungan pada keahlian pengguna.
### Solution Statement
1. **Solusi 1 : Penerapan dua arsitektur _Convolutional Neural Network_ (CNN)**
   Dengan membangun model CNN menggunakan arsitektur seperti MobileNet dan DenseNet, yang cocok untuk image classification, sehingga mampu membandingkan kinerja dua model untuk mendapatkan akurasi yang baik dalam mendeteksi penyakit kentang berdasarkan daunnya.
2. **Solusi 2 : Penerapan Hyperparameter Tuning, Augmentasi Data, dan Fine Tuning**
   Melakukan optimasi terhadap hyperparameter seperti learning rate, batch size, dan jumlah epoch untuk meningkatkan akurasi model. Augmentasi data dilakukan untuk memperkaya data latih dan meningkatkan generalisasi model. Serta fine-tuning dilakukan untuk menyesuaikan dan mengoptimalkan model deep learning agar berkeja lebih baik pada task atau dataset spesifik yang baru. 
## Data Understanding
Pada proyek ini, data yang digunakan adalah gambar daun tanaman kentang yang berasal dari dataset publik PlantVillage. Dataset ini tersedia secara bebas dan dapat diundul melalui [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/hafiznouman786/potato-plant-diseases-data). Dataset PlantVillage memiliki tiga label kelas yaitu :
1. Potato___Early_Blight -> daun kentang yang terkena penyakit _Early Blight_ atau bercak daun kering.
2. Potato___Late_Blight -> daun kentang yang terkena penyakit _Late Blight_ atau hawar daun.
3. Potati___healthy -> daun kentang dalam kondisi sehat.
**Struktur Dataset Mentah**
```
PlantVillage/
├── Potato___Early_Blight/        # Kumpulan gambar penyakit early blight (mentah)
├── Potato___Late_Blight/         # Kumpulan gambar penyakit late blight (mentah)
└── Potato___Healthy/             # Kumpulan gambar daun kentang sehat (mentah)
```

**Jumlah Data Tiap Kelas (Sebelum Augmentasi untuk keperluan Balencing Data)**

| Kelas           | Jumlah Gambar |
|----------------|----------------|
| Early Blight   | 1000           |
| Late Blight    | 1000           |
| Healthy        | 152            |
| **Total**      | **2152**       |

**Struktur Dataset Setelah diolah**
```
PlantVillage/
├── Final_Potato/
│   ├── augmented_train/              # Data pelatihan hasil augmentasi
│   ├── test/                         # Data pengujian
│   ├── train/                        # Data pelatihan asli
│   └── val/                          # Data validasi
├── Potato___Early_blight/           # Kumpulan gambar penyakit early blight (mentah)
├── Potato___Late_blight/            # Kumpulan gambar penyakit late blight (mentah)
└── Potato__healthy_augmented/       # Gambar daun kentang sehat (hasil augmentasi)
```

### Variabel-variabel pada Potato Disease sebagai berikut :
- Image Data : berupa matrix pixel dari gambar dengan dimensi umumnya 256x256@ atau resize sesuai dengan kebutuhan model
- Label : nama kelas dari masing-masing gambar yang menunjukkan kondisi daun kentang (early_blight, late_blight, healthy).
- Fiename : nama file gambar, berguna untuk identifikasi dan mapping ke label.

### Tahapan yang dilakukan untuk memahmi data 
**EDA (_Exploratory Data Analysis_)**
1. Menampilkan contoh 5 gambar tiap kelas

```
# Menampilkan 5 gambar acak dari setiap kelas
fig, axs = plt.subplots(len(lung_image), 5, figsize=(15, 3 * len(lung_image)))

for i, class_name in enumerate(lung_image.keys()):
    images = np.random.choice(lung_image[class_name], 5, replace=False)

    for j, image_name in enumerate(images):
        img_path = os.path.join(path, class_name, image_name)
        img = Image.open(img_path).convert("RGB")  # tampilkan dalam warna
        axs[i, j].imshow(img)
        axs[i, j].set(xlabel=class_name, xticks=[], yticks=[])

fig.tight_layout()
plt.show()

```

![Image](https://github.com/user-attachments/assets/8143aa3c-25c6-4349-a17e-6cc9e0e3e796)
  
2. Menampilkan visualisasi distribusi jumlah gambar tiap kelas

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
