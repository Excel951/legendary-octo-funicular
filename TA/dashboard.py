import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, \
	f1_score
import warnings

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
	page_title="Employee Turnover Prediction Dashboard",
	page_icon="🎯",
	layout="wide",
	initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">🎯 Employee Turnover Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown(
	'<p style="text-align: center; font-size: 1.2rem; color: #666;">Analisis prediksi turnover karyawan menggunakan Machine Learning</p>',
	unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📋 Menu Navigation")
# menu = st.sidebar.selectbox("Pilih Menu", ["📊 Dashboard", "🔍 Analisis Detail", "⚙️ Model Settings"])
menu = st.sidebar.radio("Pilih Menu", ["📊 Dashboard", "🔍 Analisis Detail", "⚙️ Model Settings"])

# Initialize session state
if 'data' not in st.session_state:
	st.session_state.data = None
if 'predictions' not in st.session_state:
	st.session_state.predictions = None
if 'model' not in st.session_state:
	st.session_state.model = None


def preprocess_data(df):
	"""Preprocess the employee data"""
	# Create a copy to avoid modifying original data
	processed_df = df.copy()

	# Create target variable based on business rules
	processed_df['will_leave'] = 0

	# Define conditions for likely turnover
	conditions = (
			(processed_df['gaji'] < 4000000) |  # Low salary
			(processed_df['ketidakhadiran'] > 3) |  # High absenteeism
			(processed_df['penilaian_kinerja'] < 60) |  # Poor performance
			((processed_df['masa_kerja'] < 1) & (processed_df['penilaian_kinerja'] < 70)) |  # New + poor performance
			(processed_df['masa_kerja'] > 8)  # Very long tenure
	)

	# Apply some randomness to make it more realistic
	np.random.seed(42)
	random_factor = np.random.random(len(processed_df)) < 0.3
	processed_df.loc[conditions | random_factor, 'will_leave'] = 1

	# Encode categorical variables
	le_jabatan = LabelEncoder()
	le_jenis_kelamin = LabelEncoder()
	le_status = LabelEncoder()
	le_alamat = LabelEncoder()

	processed_df['jabatan_encoded'] = le_jabatan.fit_transform(processed_df['jabatan'])
	processed_df['jenis_kelamin_encoded'] = le_jenis_kelamin.fit_transform(processed_df['jenis_kelamin'])
	processed_df['status_pernikahan_encoded'] = le_status.fit_transform(processed_df['status_pernikahan'])
	processed_df['alamat_encoded'] = le_alamat.fit_transform(processed_df['alamat'])

	return processed_df, le_jabatan, le_jenis_kelamin, le_status, le_alamat


def train_model(df):
	"""Train the machine learning model"""
	# Features for training
	features = ['gaji', 'masa_kerja', 'ketidakhadiran', 'penilaian_kinerja', 'usia', 'jabatan_encoded', 'jenis_kelamin_encoded', 'status_pernikahan_encoded', 'alamat_encoded']

	X = df[features]
	y = df['will_leave']

	# Split data
	skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

	accuracy_scores, precision_scores, recall_scores, f1_scores, conf_matrices = [], [], [], [], []

	# Train Random Forest model
	model = XGBClassifier(
		n_estimators=100,
		max_depth=10,
		learning_rate=0.1,
		eval_metric='logloss',
		random_state=42
	)

	# try:
	# 	# Load with DMatrix first
	# 	model = xgboost.Booster()
	# 	model.load_model('xgboost_model.json')
	# 	model_type = 'dmatrix'
	# except:
	# 	try:
	# 		# Fallback to joblib if DMatrix fails
	# 		model = joblib.load('xgboost_model.joblib')
	# 		model_type = 'sklearn'
	# 	except:
	# 		st.error("❌ Model file not found! Please ensure 'xgb_dmatrix_model.json' or 'xgb_classifier_model.joblib' exists.")
	# 		return None, 0, np.zeros((2, 2)), None, None, None, None

	for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
		X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
		y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

		model.fit(X_train, y_train)

		# Make predictions
		y_pred = model.predict(X_val)
		y_pred_proba = model.predict_proba(X_val)[:, 1]

		# Make predictions based on model type
		# if model_type == 'dmatrix':
		# 	# For DMatrix model
		# 	dval = xgboost.DMatrix(X_val)
		# 	y_pred_proba = model.predict(dval)
		# 	# Apply optimal threshold (research-based recommendation)
		# 	optimal_threshold = 0.4
		# 	y_pred = (y_pred_proba > optimal_threshold).astype(int)
		# else:
		# 	# For sklearn model
		# 	y_pred = model.predict(X_val)
		# 	y_pred_proba = model.predict_proba(X_val)[:, 1]

		# Calculate metrics
		accuracy_scores.append(accuracy_score(y_val, y_pred))
		precision_scores.append(precision_score(y_val, y_pred))
		recall_scores.append(recall_score(y_val, y_pred))
		f1_scores.append(f1_score(y_val, y_pred))
		conf_matrices.append(confusion_matrix(y_val, y_pred))

		if fold == 4:
			last_y_val = y_val
			last_y_pred = y_pred
			last_y_pred_proba = y_pred_proba

	total_conf_matrix = np.sum(conf_matrices, axis=0)

	return model, np.mean(accuracy_scores), total_conf_matrix, X_val, last_y_val, last_y_pred, last_y_pred_proba


def create_prediction_charts(df):
	"""Create prediction visualization charts"""
	col1, col2 = st.columns(2)

	with col1:
		# Prediction distribution
		pred_counts = df['prediction'].value_counts()
		fig_pie = px.pie(
			values=pred_counts.values,
			names=['Akan Bertahan', 'Akan Keluar'],
			title="📊 Distribusi Prediksi Turnover",
			color_discrete_sequence=['#27ae60', '#e74c3c']
		)
		fig_pie.update_layout(height=400)
		st.plotly_chart(fig_pie, use_container_width=True)

	with col2:
		# Risk level distribution
		risk_counts = df['risk_level'].value_counts()
		fig_risk = px.pie(
			values=risk_counts.values,
			names=risk_counts.index,
			title="⚠️ Distribusi Risk Level",
			color_discrete_sequence=['#e74c3c', '#f39c12', '#27ae60']
		)
		fig_risk.update_layout(height=400)
		st.plotly_chart(fig_risk, use_container_width=True)


def create_detailed_charts(df):
	"""Create detailed analysis charts"""
	col1, col2 = st.columns(2)

	with col1:
		# Turnover by position
		jabatan_analysis = df.groupby('jabatan')['prediction'].apply(
			lambda x: (x == 'Will Leave').sum() / len(x) * 100
		).reset_index()
		jabatan_analysis.columns = ['jabatan', 'turnover_rate']

		fig_jabatan = px.bar(
			jabatan_analysis,
			x='jabatan',
			y='turnover_rate',
			title="🏢 Turnover Rate per Jabatan (%)",
			color='turnover_rate',
			color_continuous_scale='Reds'
		)
		fig_jabatan.update_layout(height=400)
		st.plotly_chart(fig_jabatan, use_container_width=True)

	with col2:
		# Salary vs Performance scatter
		fig_scatter = px.scatter(
			df,
			x='gaji',
			y='penilaian_kinerja',
			color='prediction',
			size='masa_kerja',
			hover_data=['jabatan', 'usia'],
			title="💰 Gaji vs Penilaian Kinerja",
			color_discrete_map={'Will Leave': '#e74c3c', 'Will Stay': '#27ae60'}
		)
		fig_scatter.update_layout(height=400)
		st.plotly_chart(fig_scatter, use_container_width=True)


def create_confusion_matrix_plot(cm):
	"""Create confusion matrix visualization"""
	fig, ax = plt.subplots(figsize=(8, 6))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
	ax.set_xlabel('Predicted')
	ax.set_ylabel('Actual')
	ax.set_title('Confusion Matrix')
	return fig


# Main Dashboard
if menu == "📊 Dashboard":
	# File upload section
	st.markdown("### 📁 Upload Data Karyawan")

	col1, col2 = st.columns([2, 1])

	with col1:
		uploaded_file = st.file_uploader(
			"Pilih file CSV",
			type=['csv'],
			help="Upload file CSV dengan kolom: employee_id, gaji, masa_kerja, ketidakhadiran, penilaian_kinerja, jabatan, usia, jenis_kelamin, status_pernikahan, alamat"
		)

	with col2:
		st.markdown("**Format Data:**")
		st.code("""
employee_id,gaji,masa_kerja,ketidakhadiran,
penilaian_kinerja,jabatan,usia,jenis_kelamin,
status_pernikahan,alamat
        """)

	if uploaded_file is not None:
		# Load and process data
		df = pd.read_csv(uploaded_file)
		st.session_state.data = df

		# Show data info
		st.success(f"✅ Data berhasil diupload! Total: {len(df)} karyawan")

		# Preprocess data
		# if 'predictions' not in st.session_state or uploaded_file is not None:
		# if 'df' in locals():
		with st.spinner("🔄 Memproses data dan melatih model..."):
			processed_df, *encoders = preprocess_data(df)
			model, accuracy, cm, X_test, y_test, y_pred, y_pred_proba = train_model(processed_df)

			# Make predictions for all data
			features = ['gaji', 'masa_kerja', 'ketidakhadiran', 'penilaian_kinerja', 'usia',
			            'jabatan_encoded', 'jenis_kelamin_encoded', 'status_pernikahan_encoded', 'alamat_encoded']

			all_predictions = model.predict(processed_df[features])
			all_probabilities = model.predict_proba(processed_df[features])[:, 1]

			# Add predictions to dataframe
			processed_df['prediction'] = ['Will Leave' if x == 1 else 'Will Stay' for x in all_predictions]
			processed_df['leave_probability'] = all_probabilities
			processed_df['risk_level'] = pd.cut(
				all_probabilities,
				bins=[0, 0.3, 0.7, 1.0],
				labels=['Low', 'Medium', 'High']
			)

			st.session_state.predictions = processed_df
			st.session_state.model = model

		# Display metrics
		st.markdown("### 📈 Ringkasan Hasil Analisis")

		col1, col2, col3, col4 = st.columns(4)

		total_employees = len(processed_df)
		will_leave = len(processed_df[processed_df['prediction'] == 'Will Leave'])
		will_stay = total_employees - will_leave

		with col1:
			st.metric("👥 Total Karyawan", total_employees)
		with col2:
			st.metric("📉 Akan Keluar", will_leave, delta=f"{will_leave / total_employees * 100:.1f}%")
		with col3:
			st.metric("📈 Akan Bertahan", will_stay, delta=f"{will_stay / total_employees * 100:.1f}%")
		with col4:
			st.metric("🎯 Model Accuracy", f"{accuracy * 100:.1f}%")

		# Charts
		st.markdown("### 📊 Visualisasi Data")
		create_prediction_charts(processed_df)

		# Confusion Matrix
		st.markdown("### 🎯 Confusion Matrix")
		col1, col2 = st.columns([1, 1])

		with col1:
			fig_cm = create_confusion_matrix_plot(cm)
			st.pyplot(fig_cm)

		with col2:
			st.markdown("**Interpretasi Confusion Matrix:**")
			tn, fp, fn, tp = cm.ravel()
			st.write(f"• **True Negative**: {tn} (Prediksi tidak keluar, aktual tidak keluar)")
			st.write(f"• **False Positive**: {fp} (Prediksi keluar, aktual tidak keluar)")
			st.write(f"• **False Negative**: {fn} (Prediksi tidak keluar, aktual keluar)")
			st.write(f"• **True Positive**: {tp} (Prediksi keluar, aktual keluar)")

			precision = tp / (tp + fp) if (tp + fp) > 0 else 0
			recall = tp / (tp + fn) if (tp + fn) > 0 else 0
			f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

			st.write(f"• **Precision**: {precision:.3f}")
			st.write(f"• **Recall**: {recall:.3f}")
			st.write(f"• **F1-Score**: {f1:.3f}")

		# Employee table
		st.markdown("### 👥 Daftar Karyawan & Prediksi")

		tab1, tab2, tab3 = st.tabs(["📋 Semua Karyawan", "📉 Akan Keluar", "📈 Akan Bertahan"])

		with tab1:
			st.dataframe(
				processed_df[['employee_id', 'usia', 'jabatan', 'gaji', 'masa_kerja',
				              'penilaian_kinerja', 'prediction', 'risk_level', 'leave_probability']].round(3),
				use_container_width=True
			)

		with tab2:
			leave_df = processed_df[processed_df['prediction'] == 'Will Leave']
			st.dataframe(
				leave_df[['employee_id', 'usia', 'jabatan', 'gaji', 'masa_kerja',
				          'penilaian_kinerja', 'risk_level', 'leave_probability']].round(3),
				use_container_width=True
			)

		with tab3:
			stay_df = processed_df[processed_df['prediction'] == 'Will Stay']
			st.dataframe(
				stay_df[['employee_id', 'usia', 'jabatan', 'gaji', 'masa_kerja',
				         'penilaian_kinerja', 'risk_level', 'leave_probability']].round(3),
				use_container_width=True
			)

		# Download section
		st.markdown("### 📥 Download Hasil")
		col1, col2 = st.columns(2)

		with col1:
			csv = processed_df.to_csv(index=False)
			st.download_button(
				label="📊 Download CSV",
				data=csv,
				file_name="employee_turnover_predictions.csv",
				mime="text/csv"
			)

	elif st.session_state.data is not None:
		df = st.session_state.data

elif menu == "🔍 Analisis Detail":
	if st.session_state.predictions is not None:
		df = st.session_state.predictions

		st.markdown("### 🔍 Analisis Mendalam")

		# Detailed charts
		create_detailed_charts(df)

		# Feature importance
		if st.session_state.model is not None:
			st.markdown("### 🎯 Feature Importance")

			feature_names = ['Gaji', 'Masa Kerja', 'Ketidakhadiran', 'Penilaian Kinerja', 'Usia',
			                 'Jabatan', 'Jenis Kelamin', 'Status Pernikahan', 'Alamat']

			importances = st.session_state.model.feature_importances_
			feature_importance_df = pd.DataFrame({
				'Feature': feature_names,
				'Importance': importances
			}).sort_values('Importance', ascending=True)

			fig_importance = px.bar(
				feature_importance_df,
				x='Importance',
				y='Feature',
				orientation='h',
				title="📊 Tingkat Kepentingan Fitur dalam Prediksi",
				color='Importance',
				color_continuous_scale='viridis'
			)
			fig_importance.update_layout(height=500)
			st.plotly_chart(fig_importance, use_container_width=True)

		# Age and salary analysis
		st.markdown("### 👥 Analisis Demografi")

		col1, col2 = st.columns(2)

		with col1:
			# Age distribution by prediction
			fig_age = px.box(
				df,
				x='prediction',
				y='usia',
				color='prediction',
				title="📊 Distribusi Usia berdasarkan Prediksi",
				color_discrete_map={'Will Leave': '#e74c3c', 'Will Stay': '#27ae60'}
			)
			st.plotly_chart(fig_age, use_container_width=True)

		with col2:
			# Salary distribution by prediction
			fig_salary = px.box(
				df,
				x='prediction',
				y='gaji',
				color='prediction',
				title="💰 Distribusi Gaji berdasarkan Prediksi",
				color_discrete_map={'Will Leave': '#e74c3c', 'Will Stay': '#27ae60'}
			)
			st.plotly_chart(fig_salary, use_container_width=True)

	else:
		st.warning("⚠️ Silakan upload data terlebih dahulu di menu Dashboard!")

elif menu == "⚙️ Model Settings":
	st.markdown("### ⚙️ Pengaturan Model")

	if st.session_state.data is not None:
		st.markdown("#### 🔧 Parameter Model")

		col1, col2 = st.columns(2)

		with col1:
			n_estimators = st.slider("Jumlah Trees", 50, 200, 100)
			max_depth = st.slider("Kedalaman Maksimum", 3, 20, 10)

		with col2:
			min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
			min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)

		if st.button("🔄 Retrain Model"):
			with st.spinner("🔄 Melatih ulang model..."):
				processed_df, *encoders = preprocess_data(st.session_state.data)

				# Custom model with new parameters
				features = ['gaji', 'masa_kerja', 'ketidakhadiran', 'penilaian_kinerja', 'usia',
				            'jabatan_encoded', 'jenis_kelamin_encoded', 'status_pernikahan_encoded', 'alamat_encoded']

				X = processed_df[features]
				y = processed_df['will_leave']

				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

				model = RandomForestClassifier(
					n_estimators=n_estimators,
					max_depth=max_depth,
					min_samples_split=min_samples_split,
					min_samples_leaf=min_samples_leaf,
					random_state=42
				)
				model.fit(X_train, y_train)

				y_pred = model.predict(X_test)
				accuracy = accuracy_score(y_test, y_pred)

				st.success(f"✅ Model berhasil dilatih ulang! Akurasi: {accuracy * 100:.1f}%")
				st.session_state.model = model

	else:
		st.warning("⚠️ Silakan upload data terlebih dahulu di menu Dashboard!")

# Footer
st.markdown("---")
st.markdown(
	"<div style='text-align: center; color: #666;'>"
	"🎯 Employee Turnover Prediction Dashboard | Built with Streamlit & Scikit-learn"
	"</div>",
	unsafe_allow_html=True
)
