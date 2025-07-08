# Mengimpor library yang diperlukan
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import os
import io
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi halaman Streamlit
st.set_page_config(
	page_title="Dasbor Prediksi Turnover Karyawan",
	page_icon="🎯",
	layout="wide",
	initial_sidebar_state="collapsed"
)

# CSS Kustom untuk tampilan yang lebih baik
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.2rem;
        font-weight: normal;
    }
    .metric-card .value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-card .delta {
        font-size: 1rem;
        font-weight: normal;
    }
</style>
""", unsafe_allow_html=True)

# Judul dan header
st.markdown('<h1 class="main-header">🎯 Dasboard Prediksi Turnover Karyawan</h1>', unsafe_allow_html=True)
# st.markdown(
# 	'<p style="text-align: center; font-size: 1.2rem; color: #666;">Analisis prediktif untuk mengidentifikasi karyawan yang berisiko keluar menggunakan Model XGBoost.</p>',
# 	unsafe_allow_html=True)


# --- FUNGSI-FUNGSI ---

@st.cache_data
def get_kota_coord(kota, _geolocator):
	"""Fungsi untuk mendapatkan koordinat kota dengan caching."""
	try:
		location = _geolocator.geocode(kota + ", Indonesia")
		time.sleep(1)
		if location:
			return (location.latitude, location.longitude)
	except Exception:
		return None
	return None


def preprocess_data_for_prediction(df_raw, model_columns, all_kary_types):
	"""Fungsi ini mengambil DataFrame raw dan melakukan semua langkah preprocessing."""
	df = df_raw.copy()
	if 'UMUR' in df.columns and df['UMUR'].isnull().any():
		mean_umur = df['UMUR'].mean()
		df['UMUR'].fillna(round(mean_umur), inplace=True)
	df['MULAI_KERJA'] = pd.to_datetime(df['MULAI_KERJA'], errors='coerce')
	df['TGL_KELUAR'] = pd.to_datetime(df['TGL_KELUAR'], errors='coerce')
	today = pd.to_datetime(datetime.now().date())

	def hitung_lama_kerja(row):
		end_date = row['TGL_KELUAR'] if pd.notna(row['TGL_KELUAR']) else today
		if pd.isna(row['MULAI_KERJA']): return 0
		return (end_date - row['MULAI_KERJA']).days / 365.25

	df['LAMA_KERJA'] = df.apply(hitung_lama_kerja, axis=1).round(2)

	def hitung_skor_kehadiran(row):
		skor = 100
		skor -= row.get('A', 0) * 10
		skor -= row.get('LTI', 0) * 5
		skor -= row.get('I', 0) * 2
		skor -= row.get('LDI', 0) * 1
		sakit_berlebih = max(0, row.get('S', 0) - 5)
		skor -= sakit_berlebih * 2
		return max(0, skor)

	df['SKOR_KEHADIRAN'] = df.apply(hitung_skor_kehadiran, axis=1)
	if 'KOTA_TINGGAL' in df.columns:
		KANTOR_COORDS = (-7.310826754349532, 112.782069)
		geolocator = Nominatim(user_agent="turnover_dashboard_app_v8")
		distances_km = [geodesic(KANTOR_COORDS, get_kota_coord(kota, geolocator)).kilometers if get_kota_coord(kota,
		                                                                                                       geolocator) else np.nan
		                for kota in df['KOTA_TINGGAL']]
		df['JARAK_TINGGAL'] = distances_km
		if df['JARAK_TINGGAL'].isnull().any():
			mean_jarak = df['JARAK_TINGGAL'].mean()
			df['JARAK_TINGGAL'].fillna(mean_jarak, inplace=True)
	if 'STS_NIKAH' in df.columns and 'JENIS_KELAMIN' in df.columns:
		df['STS_NIKAH'] = LabelEncoder().fit_transform(df['STS_NIKAH'])
		df['JENIS_KELAMIN'] = LabelEncoder().fit_transform(df['JENIS_KELAMIN'])
	if 'KARY_TYPE' in df.columns:
		df['KARY_TYPE'] = pd.Categorical(df['KARY_TYPE'], categories=all_kary_types)
		df = pd.get_dummies(df, columns=['KARY_TYPE'], prefix='KARY_TYPE')
	cols_to_drop = ['NAMA', 'STATUS', 'MULAI_KERJA', 'TGL_KELUAR', 'KOTA_TINGGAL', 'T', 'LDI', 'LTI', 'I', 'S', 'A',
	                'D', 'CD', 'CP', 'CN', 'CL', 'CB', 'CSJ']
	df_processed = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
	return df_processed.reindex(columns=model_columns, fill_value=0)


@st.cache_resource
def load_model_and_features():
	"""Memuat model XGBoost dan nama fiturnya dengan caching."""
	model_path = 'xgboost_model.json'
	if not os.path.exists(model_path):
		return None, None, None
	model = xgboost.Booster()
	model.load_model(model_path)
	model_feature_columns = model.feature_names
	all_kary_types = [col.replace('KARY_TYPE_', '') for col in model_feature_columns if col.startswith('KARY_TYPE_')]
	return model, model_feature_columns, all_kary_types


def categorize_salary(salary):
	"""Mengelompokkan gaji ke dalam kategori untuk keamanan tampilan."""
	if salary < 7000000:
		return "Rendah (< 7 Juta)"
	elif 7000000 <= salary < 15000000:
		return "Menengah (7-15 Juta)"
	elif 15000000 <= salary < 25000000:
		return "Tinggi (15-25 Juta)"
	else:
		return "Sangat Tinggi (> 25 Juta)"


# --- Logika Utama Aplikasi ---

model, MODEL_FEATURE_COLUMNS, ALL_KARY_TYPES = load_model_and_features()

if model:
	st.markdown("### 📁 Unggah Data Karyawan (Format Raw)")
	uploaded_file = st.file_uploader(
		"Pilih file CSV atau Excel",
		type=['csv', 'xlsx'],
		label_visibility="collapsed"
	)

	# DIUBAH: Membuat template data yang lebih lengkap dan informatif
	with st.expander("Lihat atau Unduh Template Data"):
		st.markdown(
			"Gunakan template ini untuk memastikan format data Anda benar. Kolom `KARY_TYPE` dapat diisi dengan: `T`, `TD`, `K`, `I`, `PT`, `DH`, `DC`, `PC`.")

		template_data = {
			'NAMA': ['Budi Santoso', 'Ani Wijaya', 'Candra Kusuma', 'Dewi Lestari', 'Eko Prasetyo', 'Fitriani',
			         'Gilang Ramadhan', 'Hesti Wulandari'],
			'KARY_TYPE': ['T', 'TD', 'K', 'I', 'PT', 'DH', 'DC', 'PC'],
			'STATUS': ['M', 'M', 'M', 'P', 'P', 'M', 'M', 'P'],
			'MULAI_KERJA': ['2018-03-15', '2021-07-20', '2022-01-10', '2023-06-01', '2023-02-15', '2024-01-05',
			                '2019-05-20', '2020-08-11'],
			'TGL_KELUAR': [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
			'UMUR': [35, 28, 30, 22, 25, 29, 38, 33],
			'JENIS_KELAMIN': ['Pria', 'Wanita', 'Pria', 'Wanita', 'Pria', 'Wanita', 'Pria', 'Wanita'],
			'STS_NIKAH': ['Menikah', 'Belum Menikah', 'Menikah', 'Belum Menikah', 'Belum Menikah', 'Menikah', 'Menikah',
			              'Janda'],
			'T': [220, 215, 225, 150, 180, 200, 210, 218],
			'LDI': [5, 2, 3, 0, 1, 4, 6, 1],
			'LTI': [1, 0, 2, 0, 0, 3, 4, 0],
			'I': [2, 1, 0, 0, 3, 1, 2, 0],
			'S': [3, 4, 1, 2, 0, 2, 3, 5],
			'A': [0, 1, 0, 0, 0, 1, 2, 0],
			'D': [10, 5, 8, 0, 2, 7, 12, 6],
			'CD': [0, 0, 0, 0, 0, 0, 0, 0],
			'CP': [5, 3, 4, 1, 2, 3, 5, 4],
			'CN': [0, 0, 0, 0, 0, 0, 0, 0],
			'CL': [0, 0, 0, 0, 0, 0, 0, 0],
			'CB': [0, 0, 0, 0, 0, 0, 0, 0],
			'CSJ': [0, 0, 0, 0, 0, 0, 0, 0],
			'GAJI': [20000000, 12000000, 9000000, 4000000, 6000000, 5000000, 18000000, 10000000],
			'PENILAIAN_KINERJA': [4.5, 3.8, 4.0, 4.1, 3.5, 3.9, 4.3, 4.6],
			'KOTA_TINGGAL': ['Surabaya', 'Sidoarjo', 'Gresik', 'Surabaya', 'Sidoarjo', 'Surabaya', 'Gresik', 'Sidoarjo']
		}
		template_df = pd.DataFrame(template_data)
		st.dataframe(template_df, hide_index=True)

		output = io.BytesIO()
		template_df.to_csv(output, index=False)
		st.download_button(label="📥 Unduh Template CSV", data=output.getvalue(), file_name="template_data_karyawan.csv",
		                   mime="text/csv")

	if uploaded_file is not None:
		try:
			if uploaded_file.name.endswith('.csv'):
				df_raw = pd.read_csv(uploaded_file, parse_dates=['MULAI_KERJA', 'TGL_KELUAR'])
			else:
				df_raw = pd.read_excel(uploaded_file)
			st.success(f"✅ Data berhasil diunggah! Total: {len(df_raw)} karyawan.")

			with st.spinner("🔄 Memproses data dan melakukan prediksi..."):
				df_for_pred = df_raw.drop(columns=['TERMINATED'], errors='ignore')
				df_processed = preprocess_data_for_prediction(df_for_pred, MODEL_FEATURE_COLUMNS, ALL_KARY_TYPES)
				dmatrix_pred = xgboost.DMatrix(df_processed)
				probabilities = model.predict(dmatrix_pred)
				predictions = (probabilities > 0.5).astype(int)
				df_results = df_raw.copy()
				df_results['PREDICTION'] = ['Akan Keluar' if p == 1 else 'Akan Bertahan' for p in predictions]
				df_results['PROBABILITY_KELUAR'] = probabilities.round(3)
				df_results['RISK_LEVEL'] = pd.cut(probabilities, bins=[-0.1, 0.3, 0.7, 1.1],
				                                  labels=['Rendah', 'Sedang', 'Tinggi'], right=True)

				if 'GAJI' in df_results.columns:
					df_results['KATEGORI_GAJI'] = df_results['GAJI'].apply(categorize_salary)

			st.markdown("---")
			st.markdown("### 📈 Ringkasan Hasil Prediksi")
			col1, col2, col3 = st.columns(3)
			total_employees = len(df_results)
			will_leave = (df_results['PREDICTION'] == 'Akan Keluar').sum()
			will_stay = total_employees - will_leave
			with col1:
				st.markdown(
					f'<div class="metric-card" style="background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);"><h3>👥 Total Karyawan</h3><p class="value">{total_employees}</p></div>',
					unsafe_allow_html=True)
			with col2:
				st.markdown(
					f'<div class="metric-card" style="background: linear-gradient(135deg, #d62728 0%, #ff7f0e 100%);"><h3>📉 Diprediksi Keluar</h3><p class="value">{will_leave}</p><p class="delta">({will_leave / total_employees * 100 if total_employees > 0 else 0:.1f}%)</p></div>',
					unsafe_allow_html=True)
			with col3:
				st.markdown(
					f'<div class="metric-card" style="background: linear-gradient(135deg, #2ca02c 0%, #98df8a 100%);"><h3>📈 Diprediksi Bertahan</h3><p class="value">{will_stay}</p><p class="delta">({will_stay / total_employees * 100 if total_employees > 0 else 0:.1f}%)</p></div>',
					unsafe_allow_html=True)

			st.markdown("<br>", unsafe_allow_html=True)
			st.markdown("---")
			st.markdown("### 📊 Analisis Visual")

			col1_viz, col2_viz = st.columns(2)
			with col1_viz:
				fig_pie = px.pie(values=df_results['PREDICTION'].value_counts().values,
				                 names=df_results['PREDICTION'].value_counts().index,
				                 title="Distribusi Prediksi Turnover",
				                 color_discrete_map={'Akan Bertahan': '#2ca02c', 'Akan Keluar': '#d62728'})
				st.plotly_chart(fig_pie, use_container_width=True)
			with col2_viz:
				risk_counts = df_results['RISK_LEVEL'].value_counts().sort_index()
				fig_risk = px.pie(values=risk_counts.values, names=risk_counts.index,
				                  title="Distribusi Tingkat Risiko Karyawan",
				                  color_discrete_map={'Rendah': '#27ae60', 'Sedang': '#f39c12', 'Tinggi': '#e74c3c'})
				st.plotly_chart(fig_risk, use_container_width=True)

			st.markdown("### 🔍 Faktor Pendorong Turnover (Feature Importance)")
			try:
				importance_scores = model.get_score(importance_type='weight')
				feature_importance_df = pd.DataFrame({
					'Feature': list(importance_scores.keys()),
					'Importance': list(importance_scores.values())
				}).sort_values('Importance', ascending=True)

				fig_importance = px.bar(
					feature_importance_df,
					x='Importance',
					y='Feature',
					orientation='h',
					title="Tingkat Kepentingan Fitur dalam Prediksi",
					color='Importance',
					color_continuous_scale='viridis'
				)
				fig_importance.update_layout(height=500)
				st.plotly_chart(fig_importance, use_container_width=True)
			except Exception as e:
				st.warning(f"Tidak dapat menampilkan Feature Importance: {e}")

			st.markdown("---")
			st.markdown("### 👥 Daftar Karyawan & Hasil Prediksi")
			display_cols = ['UMUR', 'KARY_TYPE', 'KATEGORI_GAJI', 'LAMA_KERJA', 'PENILAIAN_KINERJA', 'PREDICTION', 'RISK_LEVEL', 'PROBABILITY_KELUAR']

			st.dataframe(df_results[[col for col in display_cols if col in df_results.columns]].round(3), use_container_width=True)


			@st.cache_data
			def convert_df(df_to_convert):
				return df_to_convert.to_csv(index=False).encode('utf-8')


			csv = convert_df(df_results)
			st.download_button(label="📥 Download Hasil Prediksi (CSV)", data=csv,
			                   file_name='hasil_prediksi_turnover.csv', mime='text/csv')
		except Exception as e:
			st.error(f"Terjadi kesalahan saat memproses file: {e}")
			st.warning("Pastikan format file dan nama kolom sudah sesuai.")

# Footer
st.markdown("---")
st.markdown(
	"<div style='text-align: center; color: #666;'>Dasbor Prediksi Turnover Karyawan | Dibangun dengan Streamlit & XGBoost</div>",
	unsafe_allow_html=True)
