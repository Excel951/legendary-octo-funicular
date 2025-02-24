Baik! Saya akan menunjukkan **bagaimana XGBoost berjalan dalam satu iterasi** menggunakan data yang Anda berikan.  

Kita akan melewati **langkah-langkah utama** dalam satu iterasi **XGBoost**, termasuk **penentuan pohon keputusan**, **perhitungan logit**, **gradien**, **hessian**, **pembaruan bobot**, dan **prediksi akhir**.  

---

## **1. Data yang Digunakan**
| Sample | Salary  | EngagementSurvey | EmpSatisfaction | Absences | Age | MonthsWorking | Termd (Target) |
|--------|--------|------------------|-----------------|----------|-----|---------------|----------------|
| 1      | 62506  | 4.6              | 5               | 1        | 38  | 118           | **0**          |
| 2      | 104437 | 4.96             | 3               | 17       | 46  | 14            | **1**          |
| 3      | 64955  | 3.02             | 3               | 3        | 33  | 14            | **1**          |

Fitur utama yang digunakan:  
- **Salary** (Gaji)
- **EngagementSurvey** (Keterlibatan Karyawan)
- **EmpSatisfaction** (Kepuasan Karyawan)
- **Absences** (Kehadiran)
- **Age** (Usia)
- **MonthsWorking** (Lama Bekerja)  
- **Target (Termd)**: 0 berarti karyawan bertahan, 1 berarti karyawan keluar.  

---

## **2. Langkah 1: Hitung Logit Awal (\(\hat{y}_{\text{init}}\))**
Logit awal dihitung menggunakan **log-odds dari target (Termd)**:  

\[
\hat{y}_{\text{init}} = \log \left( \frac{\sum y}{N - \sum y} \right)
\]

Dengan **105 karyawan keluar** dari **312 total karyawan**, maka:  

\[
\hat{y}_{\text{init}} = \log \left( \frac{105}{312 - 105} \right) = \log(0.5072)
\]

\[
\hat{y}_{\text{init}} \approx -0.679
\]

**Jadi, semua sampel dimulai dengan logit -0.679 dan probabilitas awal:**

\[
\hat{p} = \frac{1}{1 + e^{-(-0.679)}} = 0.336
\]

---

## **3. Langkah 2: Hitung Gradien (\( g_i \)) dan Hessian (\( h_i \))**
- **Gradien (\( g_i \))** menunjukkan kesalahan antara probabilitas prediksi dan target nyata:
  \[
  g_i = \hat{p}_i - y_i
  \]
- **Hessian (\( h_i \))** menunjukkan seberapa besar perubahan gradien:
  \[
  h_i = \hat{p}_i (1 - \hat{p}_i)
  \]

| Sample | Probabilitas Awal (\(\hat{p}\)) | Target (\( y \)) | Gradien (\( g_i \)) | Hessian (\( h_i \)) |
|--------|----------------|------|--------------|--------------|
| 1      | 0.336          | 0    | 0.336        | 0.223        |
| 2      | 0.336          | 1    | -0.664       | 0.223        |
| 3      | 0.336          | 1    | -0.664       | 0.223        |

---

## **4. Langkah 3: Tentukan Pohon Keputusan**
XGBoost memilih fitur yang **paling baik membagi data** berdasarkan **maksimal impurity reduction (Gain terbesar)**.

**Mengapa memilih fitur tertentu untuk split?**  
- XGBoost mencoba semua fitur dan melihat **nilai split yang memberikan pembagian terbaik**.
- Model menggunakan **rumus Gain** untuk menentukan split terbaik:

  \[
  Gain = \frac{1}{2} \left[ \frac{(\sum g_{left})^2}{\sum h_{left} + \lambda} + \frac{(\sum g_{right})^2}{\sum h_{right} + \lambda} - \frac{(\sum g)^2}{\sum h + \lambda} \right] - \gamma
  \]

- Setelah mencoba berbagai fitur, model memilih **Absences = 10** sebagai split terbaik karena memberikan **Gain tertinggi**.

**Pohon keputusan yang terbentuk:**
```
        Absences
       ├── ≤ 10 → Leaf Node 1
       └── > 10 → Leaf Node 2
```

Distribusi data berdasarkan split:
- **Leaf Node 1 (Absences ≤ 10):**  
  - Sample 1 → **Target: 0**
- **Leaf Node 2 (Absences > 10):**  
  - Sample 2 → **Target: 1**
  - Sample 3 → **Target: 1**

---

## **5. Langkah 4: Hitung Bobot Pembaruan untuk Setiap Leaf**
Bobot optimal di setiap **Leaf Node** dihitung menggunakan rumus:  

\[
w^* = - \frac{\sum g_i}{\sum h_i + \lambda}
\]

Misalkan **\(\lambda = 1\)**, maka:  

**Untuk Leaf Node 1 (Absences ≤ 10):**
\[
w^* = - \frac{0.336}{0.223 + 1} = - \frac{0.336}{1.223} = -0.275
\]

**Untuk Leaf Node 2 (Absences > 10):**
\[
w^* = - \frac{-0.664 + (-0.664)}{0.223 + 0.223 + 1} = - \frac{-1.328}{1.446} = 0.918
\]

---

## **6. Langkah 5: Update Logit dan Prediksi Akhir**
Setiap **sampel diperbarui berdasarkan leaf node tempatnya berada**.

| Sample | Logit Awal (\(\hat{y}\)) | Bobot Pembaruan (\( w^* \)) | Logit Baru (\(\hat{y}^{\text{baru}}\)) | Probabilitas Baru (\(\hat{p}^{\text{baru}}\)) |
|--------|-----------------|------------------|-----------------|-----------------|
| 1      | -0.679          | -0.275           | -0.954          | 0.278           |
| 2      | -0.679          | 0.918            | 0.239           | 0.560           |
| 3      | -0.679          | 0.918            | 0.239           | 0.560           |

- **Sample 1** tetap memiliki probabilitas rendah untuk keluar (\( 27.8\% \)).
- **Sample 2 dan 3** memiliki probabilitas lebih tinggi untuk keluar (\( 56.0\% \)).

### **Prediksi Akhir:**
- Jika **\( \hat{p}^{\text{baru}} > 0.5 \)** → Model memprediksi **keluar (Termd = 1)**.
- Jika **\( \hat{p}^{\text{baru}} \leq 0.5 \)** → Model memprediksi **tetap bertahan (Termd = 0)**.

**Hasil Prediksi:**
| Sample | Probabilitas Baru (\(\hat{p}^{\text{baru}}\)) | Prediksi Akhir |
|--------|-----------------|----------------|
| 1      | 0.278           | **0 (Bertahan)** |
| 2      | 0.560           | **1 (Keluar)** |
| 3      | 0.560           | **1 (Keluar)** |

---

## **Kesimpulan**
1. **XGBoost memulai dengan logit awal berdasarkan log-odds target.**  
2. **Model menghitung gradien dan hessian untuk mengukur kesalahan.**  
3. **Model memilih fitur terbaik untuk membagi data berdasarkan Gain tertinggi.**  
4. **Model menghitung bobot optimal untuk setiap leaf node.**  
5. **Logit diperbarui, dan probabilitas baru dihitung menggunakan sigmoid.**  
6. **Model menghasilkan prediksi apakah karyawan keluar atau tidak.**  

