# **1. Gambaran Besar Cara XGBoost Membuat Prediksi**
XGBoost bekerja dengan konsep **Gradient Boosting**, yaitu membangun model secara bertahap dengan menambahkan **pohon keputusan (decision tree)** untuk **memperbaiki kesalahan prediksi sebelumnya**.

Langkah-langkah utama dalam prosesnya:

1. **Inisialisasi Model**  
   - Model memulai dengan **prediksi awal** berdasarkan **log-odds** dari target.
   - Nilai ini disebut **logit awal** (\(\hat{y}_{\text{init}}\)), dihitung dengan:
     \[
     \hat{y}_{\text{init}} = \log \left( \frac{\sum y}{N - \sum y} \right)
     \]
   - Contoh (dengan data Anda):  
     \[
     \hat{y}_{\text{init}} = \log \left( \frac{105}{312 - 105} \right) = -0.679
     \]
   - Konversi ke probabilitas menggunakan fungsi **sigmoid**:
     \[
     \hat{p} = \frac{1}{1 + e^{-\hat{y}}}
     \]
     Sehingga:
     \[
     \hat{p} = \frac{1}{1 + e^{0.679}} = 0.336
     \]
   - Ini berarti, sebelum melihat fitur apa pun, model mengasumsikan bahwa setiap karyawan memiliki **33.6% probabilitas untuk keluar**.

2. **Hitung Gradien dan Hessian (Langkah Optimasi)**
   - **Gradien**: Seberapa besar kesalahan prediksi dibandingkan dengan target nyata.
     \[
     g_i = \hat{p}_i - y_i
     \]
   - **Hessian**: Seberapa cepat gradien berubah (berguna untuk optimasi).
     \[
     h_i = \hat{p}_i (1 - \hat{p}_i)
     \]

   Contoh perhitungan untuk satu sampel:

   | Sample | Termd (\( y \)) | Probabilitas Awal (\( \hat{p} \)) | Gradien (\( g_i \)) | Hessian (\( h_i \)) |
   |--------|----------------|----------------|--------------|--------------|
   | 1      | 0              | 0.336          | 0.336        | 0.223        |
   | 2      | 1              | 0.336          | -0.664       | 0.223        |
   | 3      | 1              | 0.336          | -0.664       | 0.223        |

3. **Membangun Pohon Keputusan untuk Memisahkan Data**
   - Model mencari fitur yang **paling baik membagi data** berdasarkan **nilai gradien dan hessian**.
   - Untuk setiap fitur (misalnya **Absences, Age, Salary**), model mencari nilai split yang memisahkan data dengan **maksimal impurity reduction**.
   - **Formula untuk memilih split terbaik**:  
     \[
     Gain = \frac{1}{2} \left[ \frac{(\sum g_{left})^2}{\sum h_{left} + \lambda} + \frac{(\sum g_{right})^2}{\sum h_{right} + \lambda} - \frac{(\sum g)^2}{\sum h + \lambda} \right] - \gamma
     \]
     - \(\sum g_{left}, \sum g_{right}\): Jumlah gradien di masing-masing sisi split.
     - \(\sum h_{left}, \sum h_{right}\): Jumlah hessian di masing-masing sisi split.
     - \(\lambda\): Regularisasi untuk mencegah overfitting.
     - \(\gamma\): Penalti kompleksitas split.
   - Model mencoba **berbagai split** dan memilih yang memberikan **Gain terbesar**.

   **Contoh: XGBoost memilih fitur "Absences" dan split \( <10 \) dan \( \geq10 \):**
   ```
       Absences
       ├── ≤ 10 → Leaf Node 1
       └── > 10 → Leaf Node 2
   ```
   - **Mengapa memilih split ini?**
     - Model menemukan bahwa karyawan yang memiliki **Absences > 10 lebih sering keluar**.
     - **Perhitungan gain** menunjukkan bahwa split ini **mengurangi impurity lebih baik dibandingkan fitur lainnya**.

4. **Menghitung Bobot di Setiap Leaf Node**
   - Setiap leaf node mendapatkan bobot yang digunakan untuk **memperbarui logit awal**:
     \[
     w^* = - \frac{\sum g_i}{\sum h_i + \lambda}
     \]
   - Contoh bobot pembaruan:
     - Leaf Node 1: **-0.1**
     - Leaf Node 2: **+0.2**
   - Update logit:
     \[
     \hat{y}^{\text{baru}} = \hat{y}^{\text{lama}} + w^*
     \]
     - Jika seorang karyawan berada di Leaf Node 1, maka:
       \[
       \hat{y} = -0.679 - 0.1 = -0.779
       \]
     - Jika seorang karyawan berada di Leaf Node 2, maka:
       \[
       \hat{y} = -0.679 + 0.2 = -0.479
       \]
   - Konversi logit ke probabilitas baru menggunakan sigmoid:
     \[
     \hat{p}^{\text{baru}} = \frac{1}{1 + e^{-\hat{y}^{\text{baru}}}}
     \]

5. **Iterasi Berulang dan Membangun Pohon Baru**
   - **Langkah 2-4 diulang** untuk iterasi berikutnya.
   - Model menambahkan pohon baru untuk memperbaiki kesalahan dari pohon sebelumnya.
   - Setiap pohon baru menangani **sisa error** dengan menyesuaikan bobot logit.

6. **Prediksi Akhir**
   - Setelah beberapa iterasi, model menghasilkan **nilai logit akhir**.
   - Logit dikonversi ke probabilitas:
     \[
     \hat{p}_{\text{final}} = \frac{1}{1 + e^{-\hat{y}_{\text{final}}}}
     \]
   - Jika **\( \hat{p}_{\text{final}} > 0.5 \)**, maka karyawan diprediksi akan keluar (**Termd = 1**).
   - Jika **\( \hat{p}_{\text{final}} \leq 0.5 \)**, maka karyawan diprediksi tetap bertahan (**Termd = 0**).

---

# **Kesimpulan**
- XGBoost dimulai dengan **logit awal berdasarkan log-odds dari data target**.
- Model menghitung **gradien dan hessian** untuk menentukan seberapa besar kesalahan prediksi.
- Model memilih **fitur terbaik** untuk membangun pohon keputusan dengan **Gain terbesar**.
- Model **mengupdate logit** dengan bobot di setiap leaf node.
- Proses ini **diulang berkali-kali** hingga model cukup baik dalam memprediksi.
- Prediksi akhir diperoleh dengan **konversi logit ke probabilitas menggunakan sigmoid**.

---

# **Gambaran Proses dalam Contoh Sederhana**
1. **Langkah 1 (Logit Awal):**  
   Semua karyawan dimulai dengan **logit -0.679**, probabilitas **33.6% keluar**.
2. **Langkah 2 (Pohon Pertama):**  
   Model membagi berdasarkan **Absences** (misal, <10 vs ≥10).
3. **Langkah 3 (Pembaruan Logit):**  
   Logit diperbarui berdasarkan leaf node yang didapat.
4. **Langkah 4 (Pohon Baru Ditambahkan):**  
   Model mencari fitur lain, misal **Salary** atau **EngagementSurvey** untuk split tambahan.
5. **Langkah 5 (Prediksi Akhir):**  
   Model mengonversi logit ke probabilitas dan menentukan apakah karyawan keluar atau tidak.

---

**Berapa kali XGBoost mengulangi langkah 2-4?** Bergantung pada beberapa faktor, tetapi ada beberapa cara utama untuk menentukan **jumlah iterasi (jumlah pohon dalam boosting)**.  

---

# **1. Apa yang Menentukan Jumlah Iterasi di XGBoost?**  
XGBoost **tidak memiliki angka pasti** untuk jumlah iterasi, tetapi kita bisa mengendalikannya melalui **parameter hyperparameter berikut**:  

### **(a) `n_estimators` (Jumlah Maksimum Pohon)**
- Parameter ini menentukan **batas maksimal** jumlah pohon yang akan dibangun.
- Misalnya, jika kita mengatur `n_estimators=100`, maka XGBoost **akan membuat maksimal 100 pohon** kecuali jika berhenti lebih awal.

> **Analogi:** Bayangkan kita sedang **mengasah keterampilan memasak**. Kita mencoba memasak 100 kali, tetapi jika setelah 20 kali kita sudah sangat ahli, kita bisa berhenti lebih awal.  

---

### **(b) `early_stopping_rounds` (Penghentian Dini)**
- Jika model melihat bahwa **peningkatan kinerja mulai melambat atau berhenti**, ia bisa **berhenti lebih awal** sebelum mencapai `n_estimators`.
- Misalnya, jika kita mengatur `early_stopping_rounds=10`, maka:
  - Model akan berhenti jika **selama 10 iterasi berturut-turut tidak ada perbaikan yang signifikan**.

**Bagaimana cara kerja `early_stopping_rounds`?**
1. Model memonitor **validation loss** pada data validasi.
2. Jika setelah beberapa iterasi **loss tidak membaik**, model akan berhenti.
3. Ini mencegah **overfitting** karena kita tidak membuat pohon terlalu banyak.

> **Contoh:**  
> Jika kita mengatur `n_estimators=1000` dan `early_stopping_rounds=10`, model bisa **berhenti di iterasi ke-30 jika sudah cukup baik**, sehingga tidak harus membangun 1000 pohon.

---

### **(c) Konvergensi dari Gradien**
- Secara matematis, **XGBoost terus menambahkan pohon sampai gradien (kesalahan) sangat kecil**.
- Kita bisa menghentikannya jika **gradien sudah mendekati nol** (artinya model sudah cukup bagus dan tidak perlu diperbaiki lebih jauh).
- Parameter `min_child_weight` membantu mengontrol ini:
  - Jika **gradien terlalu kecil**, model akan berhenti membuat pohon baru.
  - Ini berguna untuk mencegah **overfitting**.

---

### **(d) Regularisasi `learning_rate` (Shrinkage)**
- Jika `learning_rate` kecil, model **butuh lebih banyak pohon** untuk konvergen.
- Jika `learning_rate` besar, model **akan konvergen lebih cepat** tetapi bisa kurang stabil.
- Biasanya:
  - **`learning_rate=0.1` → 100-500 pohon** sering digunakan.
  - **`learning_rate=0.01` → 500-2000 pohon** untuk hasil lebih akurat.
  - **`learning_rate=0.3` → ≤ 100 pohon** bisa digunakan untuk hasil cepat.

> **Kesimpulan:** Semakin kecil `learning_rate`, semakin lama (banyak iterasi) model belajar.

---

# **2. Rumus untuk Menentukan Jumlah Pohon**
Secara teori, **jumlah iterasi optimal tidak bisa dihitung dengan satu rumus**, tetapi kita bisa mendekatinya dengan:  

$$T = \frac{1}{\text{learning rate}} \times \log \left( \frac{1}{\text{target error}} \right)$$

Di mana:
- \( T \) = jumlah iterasi yang diperlukan.
- `learning_rate` adalah tingkat penyesuaian per iterasi.
- `target error` adalah tingkat kesalahan minimum yang ingin dicapai.

> **Contoh Perhitungan:**
> Jika kita menggunakan `learning_rate=0.1` dan ingin mencapai **error 0.01**, maka:
> 
> $$T = \frac{1}{0.1} \times \log \left( \frac{1}{0.01} \right)$$
> 
> $$T = 10 \times \log(100) = 10 \times 2 = 20 \text{ iterasi}$$
> 
> Jadi, kira-kira kita butuh **20 pohon**.

---

# **3. Kapan XGBoost Berhenti?**
**XGBoost akan berhenti jika salah satu dari kondisi berikut terpenuhi**:
1. **Mencapai batas `n_estimators` yang ditentukan**.
2. **Tidak ada peningkatan performa dalam `early_stopping_rounds`**.
3. **Gradien (kesalahan) sudah cukup kecil** sehingga penambahan pohon tidak berpengaruh banyak.
4. **Regularisasi (`min_child_weight`, `gamma`, `lambda`, dll.) menghentikan pembentukan pohon baru**.

---

# **4. Kesimpulan**
- **Tidak ada angka pasti untuk jumlah iterasi, tetapi kita bisa mengontrolnya.**
- **Jika `early_stopping_rounds` digunakan, XGBoost akan otomatis berhenti lebih awal.**
- **Jika `learning_rate` kecil, kita butuh lebih banyak pohon untuk konvergen.**
- **Regularisasi (`min_child_weight`, `lambda`) juga bisa menghentikan iterasi.**
- **Dalam praktiknya, jumlah iterasi optimal biasanya ditemukan dengan uji coba (`grid search` atau `cross-validation`).**

> **Rekomendasi:**  
> **Gunakan `early_stopping_rounds=10-50` dengan `n_estimators` besar (misalnya 1000), agar model bisa berhenti lebih awal saat sudah optimal.**
