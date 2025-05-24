# Laporan Proyek Machine Learning - Riffa Bella Wahyu S
## Domain Proyek 
  Kentang (_Solanum tuberosum_) merupakan salah satu komoditas pertaninan penting di Indonesia. Kandungan karbohidrat didalamnya dibutuhkan manusia sebagai sumber energi utama bagi tubuh. Dalam pembudidayaan tanaman kentang terdapat tantangan yang harus dihadapi, salah satunya adalah tantangan dalam menghadapi penyakit yang menjangkiti tanaman kentang. Terdapat dua penyakit yang sangat rentan menjangkiti area daun kentang yaitu hawar daun (late blight) dan bercak kering (early blight) yang seringkali ditemukan pada tanaman kentang berusia 5 sampai 6 minggu (Amatullah dkk., 2021). Kedua penyakit ini dapat menyebabkan kerusakan signifikan pada tanaman, mengurangi hasil panen, dan menimbulkan kerugian ekonomi bagi petani.

  Deteksi dini penyakit ini sangat penting untuk mencegah penyebaran lebih lanjut dan meminimalkan kerugian. Namun, identifikasi manual oleh petani atau ahli pertanian seringkali memerlukan keahlian khusus dan dapat bersifat subjektif. Oleh karena itu, diperlukan sistem deteksi otomatis yang dapat mengenali penyakit berdasarkan citra daun kentang secara cepat dan akurat. Dengan memanfaatkan kecerdasaran buatan (AI), dalam bidang pengolahan citra dan pembelajaran mendalam (_deep learning_). Memungkinkan pembuatan sistem klasifikasi otomatis yang dapat mendeteksi penyakit tanaman kentang berdasarkan gambar daun. 

  Proyek ini digunakan untuk mengembangkan sistem yang mempermudah pekerjaan petani dalam mendeteksi gejala penyakit kentang lewat citra daun kentang. Proses identifikasi ini, terbagi menjadi tiga kategori, yaitu _healthy_ (daun sehat), _early blight_ (bercak kering), dan _late blight_ (hawar daun). Dalam melakukan identifikasi terhadap penyakit pada daun tanaman kentang, proyek ini menerapkan _Convolutional Neural Network_ (CNN), yang termasuk ke dalam salah satu metode dalam _Deep Learning_. Dengan menggunakan Arsitektur pada CNN yaitu MobileNetV2 dan DenseNet dalam menerapkan model untuk mendeteksi penyakit kentang. Data yang dipergunakan pada proyek ini diperoleh dari dataset PlantVillage yang tersedia di situs _Kaggle_. 

## Business Understanding

### Problem Statements
1. **Pernyataan Masalah 1**
   Petani kesulitan dalam mengidentifikasi penyakit _Early Blight_ dan _Late Blight_ pada tanaman kentang secara akurat dan cepat karena gejala awal penyakit ini memiliki kemiripan visual yang membingungkan.
2. **Pernyataan Masalah 2**
   Proses identifikasi penyakit secara manual oleh penyuluh atau petani bersifat subjektif, memerlukan keahlian khusus, dan tidak dapat dilakukan secara skala besar dalam waktu singkat.
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
**Struktur Dataset**
```
PlantVillage/
├── Potato___Early_Blight/
├── Potato___Late_Blight/
└── Potato___Healthy/
```

### Variable-variabel pada Potato Disease sebagai berikut :
- Image Data : berupa matrix pixel dari gambar dengan dimensi umumnya 256x256@ atau resize sesuai dengan kebutuhan model
- Label : nama kelas dari masing-masing gambar yang menunjukkan kondisi daun kentang (early_blight, late_blight, healthy).
- 
### Tahapan yang dilakukan untuk memahmi data 
**EDA (_Exploratory Data Analysis_)**
1. Menampilkan contoh 5 gambar tiap kelas
2. Menampilkan visualisasi distribusi jumlah gambar tiap kelas

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
Pada proyek ini, menggunakan dua algoritma untuk mendeteksi penyakit kentang berdasarkan daun, dengan menggunakan arsitektur MobileNetV2 dan DenseNet. Sehingga perlu dilakukan proses improvement dengan hyperparameter tuning dan fine-tuning beberapa layer atas model pretrained. Proses ini bertujuan untuk mendapatkan performa terbaik dengan akurasi tnggi dan meminimalkan overfitting dari kedua model arsitektur MobileNetV2 dan DenseNet.
**Tahapan Pemodelan**
1. Transfer learning dengan MobileNetV2 yang telah dilatih pada dataset ImageNet digunakan sebagai feature extractor. Kemudian layer terakhir dihapus dan digantikan dengan:
   ~ Global Average Pooling Layer untuk mereduksi dimensi fitur.
   ~ Dense Layer dengan 128 neuron dan aktivas ReLU.
   ~ Dropout Layer (rate 0.3) untuk mencegah overfitting.
   ~ Output Dense Layer dengan 3 neuron dan aktivas softmax untuk kalsifikasi multi-kelas.
```
model_mobilenet = Sequential([
    base_model,
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),  # Dropout ditingkatkan
    Dense(64, activation='relu'),
    Dropout(0.4),  # Dropout ditingkatkan
    Dense(3, activation='softmax')
])
```
2. Pengaturan Parameter
```
# Kompilasi awal
model_mobilenet.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```
   ~ Input shape: (224, 224, 3) sesuai ukuran gambar yang diresize.
   ~ Optimizer: Adam dengan learning rate awal 0.001.
   ~ Loss function: Categorical Crossentropy, sesuai dengan label yang sudah di-one-hot encoding.
   ~ Metrics: Akurasi untuk evaluasi performa model.
3. Fine-tuning, dilakukan setelah training awal dengan frozen backbone MobileNetV2, beberapa layer atas backbone kemudian di-unfreeze untuk dilatih ulang dengan learning rate yang lebih kecil yaitu 1e-5 agar model dapat menyesuaikan fitur khusus pada dataset penyakit kentang.
```
# Kompilasi ulang dengan LR kecil
model_mobilenet.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```
4. Hyperparameter Tuning (Improvement), dengan menyesuaikan learning rate secara bertahap menggunakan callback ReduceLROnPlateau, kemudian mengatur Batch size sebesar 64 untuk keseimbangan antara kecepatan pelatihan dan penggunaan memori, serta mengatur Epoch 10 - 20 pada training pertama dan pada saat di latih kembali dengan fine-tuning. 
## Evaluation
Dalam proyek kalsifikasi citra penyakit kentang ini, digunakan beberapa metrik evaluasi untuk mengukur performa model yaitu:
1. Accuarcy, digunakan untuk mengukur prediksi yang benar terhadap seluruh prediksi. Pada proyek ini, accuracy yang didapat mencapai 96.63%. Yang menunjukkan bahwa model mampu mengklasifikasikan sebgaian besar gambar dengan benar. Berikut rumusnya:

$$
Recall = \frac{TP + TN}{TP + TN + FP + FN}
$$

2. Precision, digunakan untuk mengukur prediksi positif yang benar dari seluruh prediksi positif. Sehingga semakin tinggi precision, semakin sediit false positive. Berikut rumusnya:

$$
Precision = \frac{TP}{TP + FP}
$$

3. Recall, digunakan untuk mengukur seberapa banyak data positif berhasil dikenali oleh model. Berikut rumusnya:

$$
Recall = \frac{TP}{TP + FN}
$$

4. F1-Score, digunakan untuk menyeimbangkan antara precision dan recall, berikut rumusnya:

$$
F1\text{-}score = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

5. Confusion Matrix, digunakan untuk menunjukkan jumlah prediksi benar dan salah berdasarkan masing-masing kelas. Serta membantu melihat performa model secara lebih detail.
