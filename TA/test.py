import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import random
from datetime import datetime
import sys

# ==============================================================================
# LANGKAH 0: MEMUAT DATA DARI FILE EXCEL
# ==============================================================================

try:
	# 1. Masukkan nama file data karyawan utama Anda (yang berisi 21 kolom)
	file_path_utama = 'Aktif24Res24.xlsx'

	# 2. Masukkan nama file yang HANYA berisi data karyawan resign dan gaji mereka
	file_path_gaji_resign = 'resign.xlsx'

	# 3. Kolom unik untuk menggabungkan data (biasanya NAMA atau ID Karyawan)
	kolom_unik = 'NAMA'

	# 4. Nama kolom yang berisi nilai gaji di file kedua
	nama_kolom_gaji = 'GAJI'

	print(f"Mencoba memuat data dari '{file_path_utama}' dan '{file_path_gaji_resign}'...")

	# Memuat data utama, pastikan kolom tanggal terbaca dengan benar
	df_karyawan = pd.read_excel(
		file_path_utama,
		parse_dates=['MULAI_KERJA', 'TGL_KELUAR'],
		header=0,
	)

	# Memuat data gaji karyawan resign
	df_gaji_resign = pd.read_excel(
		file_path_gaji_resign,
		header=0,
	)

	print("✓ Data berhasil dimuat.")
	print("\n--- 5 Baris Pertama Data Karyawan Utama ---")
	print(df_karyawan.head())
	print("\n--- 5 Baris Pertama Data Gaji Karyawan Resign ---")
	print(df_gaji_resign.head())

except FileNotFoundError as e:
	print(f"\n!!! ERROR: File tidak ditemukan !!!")
	print(f"Detail: {e}")
	print("Pastikan nama file sudah benar dan file berada di folder yang sama dengan skrip ini.")
	sys.exit()
except Exception as e:
	print(f"\n!!! ERROR: Terjadi kesalahan saat memuat data !!!")
	print(f"Detail: {e}")
	sys.exit()

# Membersihkan nama kolom dari spasi ekstra
df_karyawan.columns = [col.strip() for col in df_karyawan.columns]
df_gaji_resign.columns = [col.strip() for col in df_gaji_resign.columns]

# --- Perhitungan LAMA_KERJA ---
today = pd.to_datetime(datetime.now().date())


def hitung_lama_kerja(row):
	end_date = row['TGL_KELUAR'] if pd.notna(row['TGL_KELUAR']) else today
	if pd.isna(row['MULAI_KERJA']):
		return 0
	return (end_date - row['MULAI_KERJA']).days / 365.25


df_karyawan['LAMA_KERJA'] = df_karyawan.apply(hitung_lama_kerja, axis=1).round(2)

# ==============================================================================
# LANGKAH 1: SINTESIS GAJI YANG DIPERBAIKI
# ==============================================================================
print("\n--- Memulai Sintesis Gaji yang Diperbaiki ---")

print("Kode KARY_TYPE yang ditemukan dalam data:")
print(df_karyawan['KARY_TYPE'].value_counts())


def generate_gaji_realistis_v2(row):
	"""
	Menghasilkan gaji yang lebih realistis berdasarkan:
	- UMR Jawa Timur 2024: IDR 4,961,753
	- Kode KARY_TYPE dengan range yang lebih akurat
	- Pengalaman kerja dengan bonus yang lebih realistis
	- Faktor umur yang disesuaikan dengan kondisi pasar kerja Indonesia
	"""

	# Base salary berdasarkan UMR Jawa Timur 2024
	umr_jatim = 4_961_753

	# Mapping kode KARY_TYPE yang lebih realistis
	kary_type_mapping = {
		'TD': {  # Tidak Tetap/Kontrak
			'base_multiplier': 1.8,
			'min_multiplier': 1.2,
			'max_multiplier': 2.8,
			'typical_range': '6-14 juta'
		},
		'T': {  # Tetap - gaji paling tinggi karena job security
			'base_multiplier': 2.8,
			'min_multiplier': 2.0,
			'max_multiplier': 4.5,
			'typical_range': '10-22 juta'
		},
		'K': {  # Kontrak
			'base_multiplier': 1.9,
			'min_multiplier': 1.3,
			'max_multiplier': 3.0,
			'typical_range': '6-15 juta'
		},
		'I': {  # Intern - gaji entry level
			'base_multiplier': 1.2,
			'min_multiplier': 0.8,
			'max_multiplier': 1.8,
			'typical_range': '4-9 juta'
		},
		'PT': {  # Part Time
			'base_multiplier': 1.4,
			'min_multiplier': 1.0,
			'max_multiplier': 2.2,
			'typical_range': '5-11 juta'
		},
		'DH': {  # Daily Worker/Harian
			'base_multiplier': 1.3,
			'min_multiplier': 0.9,
			'max_multiplier': 2.0,
			'typical_range': '4-10 juta'
		}
	}

	# Ambil mapping berdasarkan KARY_TYPE
	kary_mapping = kary_type_mapping.get(row['KARY_TYPE'], {
		'base_multiplier': 2.0,
		'min_multiplier': 1.5,
		'max_multiplier': 3.0,
		'typical_range': '7-15 juta'
	})

	# Hitung experience bonus yang lebih realistis
	# Tahun 1-2: bonus 0-10%
	# Tahun 3-5: bonus 10-25%
	# Tahun 6-10: bonus 25-40%
	# Tahun 11+: bonus 40-60%
	lama_kerja = row['LAMA_KERJA']
	if lama_kerja <= 2:
		experience_bonus = lama_kerja * 0.05  # 0-10%
	elif lama_kerja <= 5:
		experience_bonus = 0.10 + (lama_kerja - 2) * 0.05  # 10-25%
	elif lama_kerja <= 10:
		experience_bonus = 0.25 + (lama_kerja - 5) * 0.03  # 25-40%
	else:
		experience_bonus = min(0.40 + (lama_kerja - 10) * 0.02, 0.60)  # max 60%

	# Faktor umur yang disesuaikan
	umur = row['UMUR']
	if umur < 25:
		age_factor = 0.85  # Fresh graduate
	elif 25 <= umur <= 30:
		age_factor = 0.95  # Early career
	elif 30 < umur <= 40:
		age_factor = 1.10  # Prime earning years
	elif 40 < umur <= 50:
		age_factor = 1.15  # Peak experience
	elif 50 < umur <= 55:
		age_factor = 1.05  # Late career
	else:
		age_factor = 0.95  # Pre-retirement

	# Faktor gender (disesuaikan dengan realitas pasar kerja Indonesia)
	gender_factor = 0.95 if row.get('JENIS_KELAMIN', 'L') == 'P' else 1.0

	# Faktor status pernikahan (married employees might get family allowance)
	marriage_factor = 1.05 if row.get('STS_NIKAH', 'TK') in ['K', 'KAWIN'] else 1.0

	# Hitung base multiplier dengan variasi
	base_multiplier = kary_mapping['base_multiplier']
	min_mult = kary_mapping['min_multiplier']
	max_mult = kary_mapping['max_multiplier']

	# Tambahkan variasi random dalam range yang masuk akal
	multiplier_variance = np.random.uniform(0.85, 1.15)
	final_multiplier = base_multiplier * multiplier_variance

	# Pastikan tidak keluar dari range yang wajar
	final_multiplier = max(min_mult, min(max_mult, final_multiplier))

	# Perhitungan gaji final
	base_salary = umr_jatim * final_multiplier
	final_salary = base_salary * (1 + experience_bonus) * age_factor * gender_factor * marriage_factor

	# Tambahkan noise kecil untuk variasi
	noise_factor = np.random.uniform(0.95, 1.05)
	final_salary *= noise_factor

	# Bulatkan ke 50 ribu terdekat untuk lebih realistis
	return round(final_salary / 50000) * 50000


def analyze_resign_data(df_resign_merged):
	"""
	Analisis mendalam data resign untuk memahami pola gaji
	"""
	print("\n--- Analisis Data Gaji Karyawan Resign ---")
	print(f"Jumlah data resign dengan gaji: {len(df_resign_merged)}")

	if len(df_resign_merged) > 0:
		print(f"Statistik Gaji Resign:")
		print(f"  Mean: IDR {df_resign_merged[nama_kolom_gaji].mean():,.0f}")
		print(f"  Median: IDR {df_resign_merged[nama_kolom_gaji].median():,.0f}")
		print(f"  Std: IDR {df_resign_merged[nama_kolom_gaji].std():,.0f}")
		print(f"  Min: IDR {df_resign_merged[nama_kolom_gaji].min():,.0f}")
		print(f"  Max: IDR {df_resign_merged[nama_kolom_gaji].max():,.0f}")

		# Analisis per KARY_TYPE
		print("\n--- Statistik Gaji Resign per KARY_TYPE ---")
		for ktype in df_resign_merged['KARY_TYPE'].unique():
			subset = df_resign_merged[df_resign_merged['KARY_TYPE'] == ktype]
			if len(subset) > 0:
				print(f"  {ktype}: Mean IDR {subset[nama_kolom_gaji].mean():,.0f}, "
				      f"Median IDR {subset[nama_kolom_gaji].median():,.0f}, n={len(subset)}")

	return df_resign_merged


# Coba gunakan data resign untuk kalibrasi
training_data = None
resign_salary_stats = {}

if not df_gaji_resign.empty:
	try:
		# Merge data resign dengan data karyawan utama
		df_karyawan_for_training = df_karyawan.copy()
		if nama_kolom_gaji in df_karyawan_for_training.columns:
			df_karyawan_for_training = df_karyawan_for_training.drop(columns=[nama_kolom_gaji])

		training_data = pd.merge(
			df_karyawan_for_training,
			df_gaji_resign[[kolom_unik, nama_kolom_gaji]],
			on=kolom_unik,
			how='inner'
		)

		if not training_data.empty:
			training_data = analyze_resign_data(training_data)

			# Simpan statistik gaji resign per kategori untuk kalibrasi
			for ktype in training_data['KARY_TYPE'].unique():
				subset = training_data[training_data['KARY_TYPE'] == ktype]
				if len(subset) >= 2:  # Minimal 2 data untuk statistik yang meaningful
					resign_salary_stats[ktype] = {
						'mean': subset[nama_kolom_gaji].mean(),
						'median': subset[nama_kolom_gaji].median(),
						'std': subset[nama_kolom_gaji].std(),
						'count': len(subset)
					}

			print(f"✓ Berhasil menganalisis data resign dari {len(resign_salary_stats)} kategori KARY_TYPE")
		else:
			print("⚠ Tidak ada data yang cocok dari file resign.")
			training_data = None
	except Exception as e:
		print(f"⚠ Error saat memproses data resign: {e}")
		training_data = None


def generate_calibrated_salary(row, resign_stats=None):
	"""
	Generate gaji yang dikalibrasi dengan data resign jika tersedia
	"""
	# Generate gaji dasar menggunakan fungsi yang sudah diperbaiki
	base_salary = generate_gaji_realistis_v2(row)

	# Jika ada data resign untuk kategori ini, lakukan kalibrasi
	if resign_stats and row['KARY_TYPE'] in resign_stats:
		resign_data = resign_stats[row['KARY_TYPE']]

		# Hitung faktor kalibrasi berdasarkan data resign
		# Gunakan weighted average antara synthetic dan resign data
		resign_weight = min(0.4, resign_data['count'] * 0.1)  # Max 40% weight untuk resign data
		synthetic_weight = 1 - resign_weight

		# Kalibrasi berdasarkan median resign (lebih robust dari mean)
		calibration_factor = resign_data['median'] / (base_salary * 0.8)  # Asumsi base salary sedikit underestimate
		calibration_factor = max(0.7, min(1.5, calibration_factor))  # Batasi faktor kalibrasi

		calibrated_salary = base_salary * (synthetic_weight + resign_weight * calibration_factor)

		# Tambahkan sedikit variasi berdasarkan standar deviasi resign data
		if resign_data['std'] > 0:
			noise = np.random.normal(0, resign_data['std'] * 0.1)
			calibrated_salary += noise

		return max(3_000_000, calibrated_salary)  # Minimum 3 juta

	return base_salary


# STRATEGI BARU: Kalibrasi dengan data resign + sintesis yang lebih akurat
print("\n--- Menggunakan Strategi Kalibrasi Baru ---")

if training_data is not None and len(resign_salary_stats) > 0:
	print("✓ Menggunakan data resign untuk kalibrasi gaji sintetis")
	df_karyawan['gaji_sintetis'] = df_karyawan.apply(
		lambda row: generate_calibrated_salary(row, resign_salary_stats), axis=1
	).astype(int)

	# Evaluasi hasil kalibrasi
	print("\n--- Evaluasi Hasil Kalibrasi ---")
	for ktype in resign_salary_stats.keys():
		synthetic_subset = df_karyawan[df_karyawan['KARY_TYPE'] == ktype]['gaji_sintetis']
		resign_median = resign_salary_stats[ktype]['median']
		synthetic_median = synthetic_subset.median()

		difference_pct = abs(synthetic_median - resign_median) / resign_median * 100
		print(f"  {ktype}: Resign median IDR {resign_median:,.0f} vs "
		      f"Synthetic median IDR {synthetic_median:,.0f} "
		      f"(selisih {difference_pct:.1f}%)")

else:
	print("✓ Menggunakan sintesis standar yang diperbaiki (tanpa data resign)")
	df_karyawan['gaji_sintetis'] = df_karyawan.apply(generate_gaji_realistis_v2, axis=1).astype(int)

print("✓ Sintesis Gaji selesai untuk SEMUA karyawan.")

# ==============================================================================
# VALIDASI DAN ANALISIS HASIL
# ==============================================================================
print("\n--- Validasi Hasil Sintesis Gaji ---")

# Statistik keseluruhan
print(f"Statistik Gaji Sintetis:")
print(f"  Mean: IDR {df_karyawan['gaji_sintetis'].mean():,.0f}")
print(f"  Median: IDR {df_karyawan['gaji_sintetis'].median():,.0f}")
print(f"  Std: IDR {df_karyawan['gaji_sintetis'].std():,.0f}")
print(f"  Min: IDR {df_karyawan['gaji_sintetis'].min():,.0f}")
print(f"  Max: IDR {df_karyawan['gaji_sintetis'].max():,.0f}")

# Validasi range gaji per KARY_TYPE
print("\n--- Validasi Range Gaji per KARY_TYPE ---")
for kary_type in sorted(df_karyawan['KARY_TYPE'].unique()):
	subset = df_karyawan[df_karyawan['KARY_TYPE'] == kary_type]['gaji_sintetis']
	count = len(subset)
	mean_val = subset.mean()
	median_val = subset.median()
	min_val = subset.min()
	max_val = subset.max()

	print(f"{kary_type}: IDR {mean_val:,.0f} (rata-rata), IDR {median_val:,.0f} (median)")
	print(f"    Range: IDR {min_val:,.0f} - IDR {max_val:,.0f}, n={count}")

# Cek distribusi gaji
print("\n--- Distribusi Gaji ---")
gaji_ranges = [
	(0, 5_000_000, "< 5 juta"),
	(5_000_000, 8_000_000, "5-8 juta"),
	(8_000_000, 12_000_000, "8-12 juta"),
	(12_000_000, 20_000_000, "12-20 juta"),
	(20_000_000, float('inf'), "> 20 juta")
]

for min_sal, max_sal, label in gaji_ranges:
	count = len(df_karyawan[(df_karyawan['gaji_sintetis'] >= min_sal) &
	                        (df_karyawan['gaji_sintetis'] < max_sal)])
	pct = count / len(df_karyawan) * 100
	print(f"  {label}: {count} orang ({pct:.1f}%)")

# ==============================================================================
# SINTESIS PENILAIAN KINERJA & ALAMAT (SAMA SEPERTI SEBELUMNYA)
# ==============================================================================
print("\n--- Memulai Sintesis Penilaian Kinerja & Alamat ---")


def hitung_skor_kinerja(row):
	skor = 3.0
	skor -= row.get('A', 0) * 1.0
	skor -= row.get('LTI', 0) * 0.2

	if row['KARY_TYPE'] == 'T':
		skor += 0.4
	elif row['KARY_TYPE'] == 'TD':
		skor += 0.2
	elif row['KARY_TYPE'] == 'K':
		skor += 0.1
	elif row['KARY_TYPE'] in ['I', 'PT', 'DH']:
		skor -= 0.1

	skor += np.random.uniform(-0.2, 0.2)
	return round(max(1.0, min(5.0, skor)), 2)


df_karyawan['penilaian_kinerja_sintetis'] = df_karyawan.apply(hitung_skor_kinerja, axis=1)

# Sintesis Alamat
kota_tinggal = {
	'Surabaya': 0.40,
	'Sidoarjo': 0.20,
	'Gresik': 0.15,
	'Lamongan': 0.10,
	'Mojokerto': 0.08,
	'Bangkalan': 0.05,
	'Pasuruan': 0.02
}


def generate_kota_tinggal(row):
	if row['gaji_sintetis'] > 12_000_000 or row['KARY_TYPE'] == 'T':
		pilihan_kota = ['Surabaya', 'Sidoarjo', 'Gresik']
		probabilitas = [0.70, 0.20, 0.10]
	elif row['gaji_sintetis'] > 8_000_000:
		pilihan_kota = ['Surabaya', 'Sidoarjo', 'Gresik', 'Lamongan', 'Mojokerto']
		probabilitas = [0.35, 0.25, 0.20, 0.15, 0.05]
	else:
		pilihan_kota = list(kota_tinggal.keys())
		probabilitas = list(kota_tinggal.values())

	return np.random.choice(pilihan_kota, p=probabilitas)


df_karyawan['kota_tinggal_sintetis'] = df_karyawan.apply(generate_kota_tinggal, axis=1)
print("✓ Sintesis Penilaian Kinerja dan Kota Tinggal selesai.")

# ==============================================================================
# HASIL AKHIR
# ==============================================================================
print("\n--- Hasil Akhir: DataFrame dengan Kolom Sintetis yang Diperbaiki ---")

output_filename = 'hasil_sintesis_data_karyawan_v2_3.xlsx'
try:
	df_karyawan.to_excel(output_filename, index=False)
	print(f"\n✓ Hasil sintesis DIPERBAIKI telah disimpan ke file: '{output_filename}'")
except Exception as e:
	print(f"\n!!! ERROR saat menyimpan file: {e} !!!")

# Tampilkan pratinjau
kolom_untuk_ditampilkan = [
	'NAMA', 'KARY_TYPE', 'STATUS', 'LAMA_KERJA', 'UMUR',
	'gaji_sintetis', 'penilaian_kinerja_sintetis', 'kota_tinggal_sintetis'
]
print("\n--- Pratinjau Hasil Akhir ---")
print(df_karyawan[kolom_untuk_ditampilkan].head(10))

print("\n--- PERBAIKAN YANG DILAKUKAN ---")
print("✓ Range gaji disesuaikan dengan kondisi pasar Indonesia 2024")
print("✓ Faktor pengalaman kerja diperbaiki dengan progression yang realistis")
print("✓ Kalibrasi dengan data resign jika tersedia")
print("✓ Validasi ekstensif terhadap hasil sintesis")
print("✓ Gaji minimum dipastikan tidak di bawah standar yang wajar")
print("✓ Variasi gaji lebih natural dengan pembulatan yang realistis")

# Final summary
print(f"\n--- RINGKASAN FINAL ---")
print(f"Total karyawan: {len(df_karyawan)}")
print(f"Rata-rata gaji sintetis: IDR {df_karyawan['gaji_sintetis'].mean():,.0f}")
print(f"Range gaji: IDR {df_karyawan['gaji_sintetis'].min():,.0f} - IDR {df_karyawan['gaji_sintetis'].max():,.0f}")
if resign_salary_stats:
	print(f"Menggunakan kalibrasi dari {len(resign_salary_stats)} kategori data resign")
print("✓ Sintesis selesai dengan hasil yang lebih realistis!")