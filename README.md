# 🧠 Prediksi Stres Psikis Berbasis Web dengan Data Fisiologis dan Machine Learning

Proyek ini bertujuan untuk membangun aplikasi web yang mampu memprediksi tingkat stres psikis pengguna berdasarkan data fisiologis seperti detak jantung, pola tidur, dan hasil kuesioner psikologis. Sistem ini didukung oleh model Machine Learning yang telah dilatih untuk mengenali pola-pola stres guna memberikan prediksi yang akurat dan membantu pengguna dalam memantau kesehatan mental mereka.

---

## 📌 Fitur Utama

- ✅ Prediksi tingkat stres berdasarkan data detak jantung, tidur, dan kuesioner psikologis.
- 📊 Visualisasi data pengguna dalam bentuk grafik yang informatif.
- 📝 Formulir input kuesioner berbasis DASS atau skala stres lainnya.
- 🚨 Deteksi dini terhadap indikasi stres berat.
- 📋 Dashboard hasil dan rekomendasi umum berbasis prediksi model.

---

## 🛠 Cara Mereplikasi Repository

Ikuti langkah-langkah berikut untuk menjalankan proyek ini di komputer lokal Anda.

### 1. Clone Repository

```bash
git clone https://github.com/SaukiFutaki/strescope
cd strescope
```



---

### 2. Instalasi Dependensi Frontend

Masuk ke folder frontend:

```bash
cd FE  
npm install
```

---

### 3. Menjalankan Frontend

Setelah proses instalasi selesai:

```bash
npm run dev
```

Jika menggunakan Webpack secara manual:

```bash
npm run start
```

Aplikasi akan berjalan di [http://localhost:3000](http://localhost:3000) secara default.

---

### 4. Instalasi dan Menjalankan Modul Machine Learning

Masuk ke folder `ML`:

```bash
cd ../ML
```



```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

Instal semua dependensi Python:

```bash
pip install -r requirements.txt
```

Jalankan server model prediksi:

```bash
python app.py
```

Server akan berjalan di [http://localhost:5000](http://localhost:5000) atau sesuai port yang dikonfigurasi.

---

### 5. Environment Variables (Opsional)



```env
REACT_APP_API_URL=http://localhost:5000/predict
```

---

### 6. Struktur Direktori Proyek

```
📁 FE/              
📁 ML/               
📁 public/           
📁 src/             
📄 .babelrc
📄 .gitignore
📄 package.json
📄 package-lock.json
📄 webpack.config.js
📄 webpack.prod.js
📄 postcss.config.mjs
📄 README.md
```

---

## 🧪 Teknologi yang Digunakan

### Frontend
- Vue js
- Webpack

### Backend ML
- Python
- Flask / FastAPI
- Pandas, Scikit-learn, atau TensorFlow
- Model prediktif dari data detak jantung, pola tidur, dan kuesioner

---


## 🙋‍♂️ Kontak

Untuk pertanyaan, saran, atau laporan bug, silakan hubungi:
- hai@gmail.com