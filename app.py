import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import joblib, io, os
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage

# ============================================================
# CONFIGURACIÓN Y COLORES
# ============================================================
st.set_page_config(page_title="Detector de Anomalías · Clínico", page_icon="🏥", layout="wide")

MODEL_PATH = "modelo_isolation_forest.pkl"
LOGO_PATH  = "logo.png"

COLORS = {"primary": "#0EA5E9", "primary_d": "#0369A1", "teal": "#14B8A6", "critico": "#DC2626", "alto": "#F59E0B", "medio": "#EAB308", "bajo": "#10B981", "neutral": "#64748B", "border": "#E2E8F0", "text_dark": "#0F172A", "text_soft": "#334155"}
RIESGO_ORDEN = ["🔴 CRÍTICO", "🟠 ALTO", "🟡 MEDIO", "🟢 BAJO"]
RIESGO_HEX = {"🔴 CRÍTICO": COLORS["critico"], "🟠 ALTO": COLORS["alto"], "🟡 MEDIO": COLORS["medio"], "🟢 BAJO": COLORS["bajo"]}

st.markdown(f"""
<style>
/* 1. FORZAR MODO CLARO Y TEXTOS MÁS GRANDES */
:root {{ color-scheme: light !important; }}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: #F8FAFC !important; color: {COLORS['text_dark']} !important;
}}
.stApp p, .stApp span, .stApp div, .stApp label, .stApp a, .stApp li, .stApp td, .stApp th, .stMarkdown, .stText {{
    color: {COLORS['text_dark']} !important; font-size: 16px !important; 
}}
.stApp h1 {{ color: {COLORS['primary_d']} !important; font-weight: 800 !important; font-size: 34px !important; }}
.stApp h2 {{ color: {COLORS['primary_d']} !important; font-weight: 700 !important; font-size: 26px !important; }}
.stApp h3 {{ color: {COLORS['primary_d']} !important; font-weight: 700 !important; font-size: 22px !important; }}
[data-testid="stCaptionContainer"] * {{ font-size: 16px !important; color: #475569 !important; font-weight: 500 !important; }}

/* 2. ARREGLAR EL CUADRO NEGRO DEL UPLOADER */
[data-testid="stFileUploaderDropzone"] {{
    background-color: #FFFFFF !important; border: 2px dashed {COLORS['primary']} !important;
    border-radius: 10px !important; padding: 20px !important;
}}
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] div, [data-testid="stFileUploaderDropzone"] p {{
    color: {COLORS['text_dark']} !important; opacity: 1 !important; font-weight: 600 !important; font-size: 16px !important;
}}
[data-testid="stFileUploaderDropzone"] button {{
    background-color: {COLORS['primary']} !important; border-radius: 8px !important; border: none !important;
}}
[data-testid="stFileUploaderDropzone"] button * {{ color: #FFFFFF !important; }}
[data-testid="stFileUploaderFile"] {{ background-color: #F1F5F9 !important; border-radius: 8px !important; }}
[data-testid="stFileUploaderFile"] * {{ color: {COLORS['text_dark']} !important; }}

/* 3. ARREGLAR INPUTS DE TEXTO Y SELECTS */
.stApp input, .stApp textarea, [data-baseweb="select"] *, [data-baseweb="input"] * {{
    color: {COLORS['text_dark']} !important; background-color: #FFFFFF !important; font-size: 16px !important;
}}
[data-baseweb="select"] [role="button"] {{ background-color: #FFFFFF !important; border: 1px solid {COLORS['border']} !important; }}

/* 4. DISEÑO GENERAL Y BARRA LATERAL (SIDEBAR) */
section[data-testid="stSidebar"] {{ background-color: #FFFFFF !important; border-right: 1px solid {COLORS['border']}; padding-top: 1rem; }}
section[data-testid="stSidebar"] * {{ color: {COLORS['text_soft']} !important; }}

.about-box {{
    background-color: #F0F9FF; border-left: 5px solid {COLORS['primary']};
    padding: 20px; border-radius: 10px; margin-bottom: 26px;
}}
.about-box h4 {{ margin-top: 0; color: {COLORS['primary_d']} !important; margin-bottom: 8px; font-size: 20px !important; }}
.about-box p {{ margin-bottom: 0; font-size: 16px !important; line-height: 1.6; }}

.kpi-card {{ background: #FFFFFF !important; border-radius: 14px; padding: 20px; border-left: 6px solid {COLORS['primary']}; box-shadow: 0 4px 14px rgba(15,23,42,0.06); height: 100%; margin-bottom: 15px; }}
.kpi-card.critico {{ border-left-color: {COLORS['critico']}; }}
.kpi-card.alto {{ border-left-color: {COLORS['alto']}; }}
.kpi-card.medio {{ border-left-color: {COLORS['medio']}; }}
.kpi-card.bajo {{ border-left-color: {COLORS['bajo']}; }}
.kpi-card.teal {{ border-left-color: {COLORS['teal']}; }}
.kpi-label {{ font-size: 14px; text-transform: uppercase; color: {COLORS['neutral']} !important; font-weight: 800; margin-bottom: 8px; }}
.kpi-value {{ font-size: 38px; font-weight: 900; color: {COLORS['text_dark']} !important; line-height: 1.1; }}
.kpi-sub {{ font-size: 15px; font-weight: 500; color: {COLORS['neutral']} !important; margin-top: 6px; }}

.report-card {{ background: #FFFFFF !important; border-radius: 12px; padding: 20px; box-shadow: 0 4px 14px rgba(15,23,42,0.06); height: 100%; border-top: 4px solid {COLORS['primary']}; margin-bottom: 15px; }}
.report-title {{ font-size: 20px; font-weight: 800; color: {COLORS['text_dark']} !important; margin-bottom: 12px; border-bottom: 2px solid {COLORS['border']}; padding-bottom: 8px; }}
.report-list {{ margin: 0; padding-left: 20px; }}
.report-list li {{ margin-bottom: 10px; font-size: 16px; color: {COLORS['text_soft']} !important; }}
.report-list strong {{ color: {COLORS['text_dark']} !important; }}

.section-title {{ font-size: 24px; font-weight: 800; color: {COLORS['text_dark']} !important; margin: 30px 0 15px 0; padding-bottom: 8px; border-bottom: 2px solid {COLORS['border']}; }}
.stDownloadButton button {{ background: linear-gradient(135deg, {COLORS['primary_d']} 0%, {COLORS['primary']} 100%) !important; border: none; border-radius: 10px; padding: 14px 18px; font-weight: 700; width: 100%; font-size: 16px; }}
.stDownloadButton button * {{ color: #FFFFFF !important; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCIONES CORE
# ============================================================
@st.cache_resource(max_entries=1, show_spinner=False)
def cargar_modelo(_mtime):
    paquete = joblib.load(MODEL_PATH)
    return paquete if isinstance(paquete, dict) else {"modelo": paquete}

def clasificar_riesgo(score, pred):
    if pred == 1 and score >= 0.75: return "🔴 CRÍTICO"
    if pred == 1 and score >= 0.50: return "🟠 ALTO"
    if pred == 1: return "🟡 MEDIO"
    return "🟢 BAJO"

def predecir(df_input, paquete):
    modelo, scaler = paquete["modelo"], paquete.get("scaler")
    feature_cols, encoders = paquete["feature_cols"], paquete.get("encoders", {})
    umbral, imputer = paquete.get("umbral"), paquete.get("imputer")
    df_work = df_input.copy()

    if "Timestamp" in df_work.columns:
        df_work["Timestamp"] = pd.to_datetime(df_work["Timestamp"], errors="coerce")
        df_work["Hour"] = df_work["Timestamp"].dt.hour.fillna(12).astype(int)
        df_work["DayOfWeek"] = df_work["Timestamp"].dt.dayofweek.fillna(0).astype(int)
        df_work["Is_Night"] = ((df_work["Hour"] >= 22) | (df_work["Hour"] < 6)).astype(int)
        df_work["Is_Weekend"] = (df_work["DayOfWeek"] >= 5).astype(int)

    for col, le in encoders.items():
        if col in df_work.columns:
            df_work[f"{col}_enc"] = df_work[col].astype(str).map(lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1)

    if "Patient_Age" in df_work.columns: df_work["Is_Pediatric"] = (pd.to_numeric(df_work["Patient_Age"], errors="coerce") < 18).fillna(False).astype(int)
    if "Duration_Seconds" in df_work.columns: df_work["Duration_log"] = np.log1p(pd.to_numeric(df_work["Duration_Seconds"], errors="coerce").clip(lower=0))
    if "Access_Reason" in df_work.columns: df_work["Suspicious_Reason"] = df_work["Access_Reason"].astype(str).str.strip().str.lower().isin(["unknown", "curiosity"]).astype(int)
    if "Access_Success" in df_work.columns: df_work["Access_Failed"] = df_work["Access_Success"].astype(str).str.strip().str.lower().eq("no").astype(int)
    if "Access_Type" in df_work.columns: df_work["Is_Destructive"] = df_work["Access_Type"].astype(str).str.strip().str.lower().isin(["delete", "edit"]).astype(int)

    risk_cols = ["Suspicious_Reason", "Access_Failed", "Is_Night", "Is_Pediatric", "Is_Destructive"]
    avail = [c for c in risk_cols if c in df_work.columns]
    if avail: df_work["Context_Risk_Score"] = df_work[avail].sum(axis=1)

    if "User_ID" in df_work.columns:
        df_work["User_Access_Count"] = df_work.groupby("User_ID")["User_ID"].transform("count")
        if "Patient_ID" in df_work.columns: df_work["User_Unique_Patients"] = df_work.groupby("User_ID")["Patient_ID"].transform("nunique")
    if "Patient_ID" in df_work.columns:
        df_work["Patient_Access_Count"] = df_work.groupby("Patient_ID")["Patient_ID"].transform("count")
        if "User_ID" in df_work.columns: df_work["Patient_Unique_Users"] = df_work.groupby("Patient_ID")["User_ID"].transform("nunique")
    
    if "User_Access_Count" in df_work.columns:
        df_work["User_Access_Ratio"] = df_work["User_Access_Count"] / max(df_work["User_Access_Count"].mean(), 1)
        if "User_Unique_Patients" not in df_work.columns: df_work["User_Unique_Patients"] = 0
        df_work["User_Patient_Diversity"] = df_work["User_Unique_Patients"] / df_work["User_Access_Count"].clip(lower=1)
    if "Patient_Access_Count" in df_work.columns:
        df_work["Patient_Access_Ratio"] = df_work["Patient_Access_Count"] / max(df_work["Patient_Access_Count"].mean(), 1)

    for c in feature_cols:
        if c not in df_work.columns: df_work[c] = 0

    X = df_work[feature_cols].apply(pd.to_numeric, errors="coerce").values
    X = imputer.transform(X) if imputer is not None else SimpleImputer(strategy="median").fit_transform(X)
    if scaler is not None: X = scaler.transform(X)

    scores_raw = -modelo.score_samples(X)
    preds = (scores_raw > umbral).astype(int) if umbral is not None else (modelo.predict(X) == -1).astype(int)
    
    p5, p95 = np.percentile(scores_raw, [5, 95])
    scores_norm = np.full_like(scores_raw, 0.5) if (p95 - p5 < 1e-10) else np.clip((scores_raw - p5) / (p95 - p5), 0, 1)

    df_work["Score_Anomalia"] = scores_norm
    df_work["Prediccion"] = preds
    df_work["Riesgo"] = [clasificar_riesgo(s, p) for s, p in zip(scores_norm, preds)]
    
    return df_work, X, preds

def kpi_card(label, value, sub="", kind="primary"):
    st.markdown(f"""
    <div class='kpi-card {kind}' style="margin-bottom: 15px;">
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value'>{value}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# LAYOUT (BARRA LATERAL CON DISEÑO CLÍNICO)
# ============================================================
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.markdown("<div style='text-align:center;font-size:72px;'>🏥</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align:center; margin-top: 10px; margin-bottom: 20px;'>
        <h2 style='color: #0369A1; font-weight: 900; font-size: 28px; margin-bottom: 0;'>MEDIDATA</h2>
        <h2 style='color: #0EA5E9; font-weight: 800; font-size: 24px; margin-top: 0;'>HEALTH</h2>
        <p style='font-size: 14px; font-weight: 700; color: #64748B; letter-spacing: 1px;'>CENTRO CLÍNICO</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <div style='text-align:center; padding: 30px 15px; background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%); border-radius: 12px; border: 2px dashed #7DD3FC;'>
        <div style='font-size: 32px; margin-bottom: 15px; color: #0EA5E9;'>✚</div>
        <p style='font-size: 18px; font-weight: 800; color: #0369A1; margin-bottom: 8px;'>Protegemos tu información,</p>
        <p style='font-size: 18px; font-weight: 800; color: #0EA5E9; margin-top: 0;'>cuidamos tu salud.</p>
    </div>
    """, unsafe_allow_html=True)

# HEADER PRINCIPAL CON EMOJI GIGANTE
st.markdown("""
<div style="background: linear-gradient(135deg, #0369A1 0%, #0EA5E9 100%); padding: 40px 50px; border-radius: 20px; margin-bottom: 30px; display: flex; align-items: center; gap: 35px; box-shadow: 0 15px 35px -10px rgba(14,165,233,0.5);">
    <div style="font-size: 110px; line-height: 1; filter: drop-shadow(0px 8px 12px rgba(0,0,0,0.2));">🩺</div>
    <div>
        <h1 style="margin: 0; font-size: 48px !important; font-weight: 900 !important; color: #FFFFFF !important; letter-spacing: -0.5px;">Detector de Anomalías Clínicas</h1>
        <p style="margin: 10px 0 0 0; font-size: 24px !important; color: #E0F2FE !important; font-weight: 500;">Auditoría inteligente de accesos a historiales · Isolation Forest</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="about-box">
    <h4>ℹ️ ¿Qué hace este sistema?</h4>
    <p>Esta plataforma es un <b>motor de auditoría inteligente</b> diseñado para la seguridad clínica. Utiliza un modelo de Machine Learning no supervisado (<i>Isolation Forest</i>) que analiza patrones de comportamiento en los accesos al sistema. Evalúa la hora, el rol del usuario, el tipo de acción y el historial para detectar automáticamente anomalías y clasificarlas en 4 niveles de riesgo.</p>
</div>
""", unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ No se encontró {MODEL_PATH}.")
    st.stop()

paquete = cargar_modelo(os.path.getmtime(MODEL_PATH))
st.success("✅ Modelo Isolation Forest cargado correctamente")

archivo = st.file_uploader("📁 Sube el archivo CSV", type=["csv"])
if not archivo: 
    st.stop()

try:
    df_input = pd.read_csv(archivo)
    st.caption(f"📄 {len(df_input):,} registros cargados · {df_input.shape[1]} columnas")
    
    posibles_label = [c for c in df_input.columns if c.lower() in ["is_anomaly", "label", "target", "anomaly", "notes"]]
    col_real = posibles_label[0] if posibles_label else None

    with st.spinner("Analizando registros con Isolation Forest..."):
        df_res, X_modelo, preds_modelo = predecir(df_input, paquete)
except Exception as e:
    st.error(f"Error procesando el archivo: {e}")
    st.stop()

# ============================================================
# MÉTRICAS Y RESUMEN
# ============================================================
total = len(df_res)
anom  = int(df_res["Prediccion"].sum())
norm  = total - anom
pct   = (anom / total * 100) if total else 0.0

st.markdown('<div class="section-title">📊 Resumen General y Auditoría</div>', unsafe_allow_html=True)
# Ahora los cuadros muestran el número grande y el texto abajo. Eliminamos Score promedio.
k1, k2, k3 = st.columns(3)
with k1: kpi_card("Total registros", f"{total:,}", "accesos analizados", "primary")
with k2: kpi_card("Anomalías", f"{anom:,}", f"{pct:.2f}% del total", "critico")
with k3: kpi_card("Registros normales", f"{norm:,}", f"{100-pct:.2f}% del total", "bajo")

# MÉTRICAS DE DESEMPEÑO
st.markdown('<div class="section-title">🎯 Métricas de Desempeño del Modelo</div>', unsafe_allow_html=True)

metricas_modelo = paquete.get("metricas", {})
prec = metricas_modelo.get("precision", 0.0)
rec = metricas_modelo.get("recall", 0.0)
f1 = metricas_modelo.get("f1", 0.0)
acc = metricas_modelo.get("accuracy", 0.0)

m1, m2, m3, m4 = st.columns(4)
with m1: kpi_card("Precision", f"{prec*100:.2f}%", "Falsos positivos", "teal")
with m2: kpi_card("Recall", f"{rec*100:.2f}%", "Detección real", "alto")
with m3: kpi_card("F1-score", f"{f1*100:.2f}%", "Balance", "primary")
with m4: kpi_card("Accuracy", f"{acc*100:.2f}%", "Aciertos", "neutral")

st.markdown("<p style='font-size: 16px; color: #475569; font-weight: 500; margin-top: 10px;'>Nota: Estas métricas fueron calculadas y guardadas durante el entrenamiento del modelo en Jupyter con los datos etiquetados de prueba.</p>", unsafe_allow_html=True)

c_rep1, c_rep2 = st.columns(2)
with c_rep1:
    st.markdown(f"""
    <div class="report-card">
        <div class="report-title">📊 Resumen de Auditoría</div>
        <ul class="report-list">
            <li><strong>Total de registros analizados:</strong> {total:,}</li>
            <li><strong>Anomalías detectadas:</strong> <span style="color:#DC2626; font-weight:bold; font-size:22px;">{anom:,}</span></li>
            <li><strong>Comportamientos normales:</strong> {norm:,}</li>
            <li><strong>Privacidad (HIPAA):</strong> Seudonimización activa. No se procesan nombres reales.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
with c_rep2:
    st.markdown(f"""
    <div class="report-card">
        <div class="report-title">👥 Usuarios Involucrados</div>
        <ul class="report-list">
            <li><strong>Usuarios analizados:</strong> {df_res['User_ID'].nunique() if 'User_ID' in df_res.columns else 'N/A'}</li>
            <li><strong>Pacientes afectados:</strong> {df_res[df_res['Prediccion']==1]['Patient_ID'].nunique() if 'Patient_ID' in df_res.columns else 'N/A'}</li>
            <li><strong>Acción más frecuente:</strong> {df_res[df_res['Prediccion']==1]['Access_Type'].mode().iloc[0] if anom > 0 and 'Access_Type' in df_res.columns else 'N/A'}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# GRÁFICOS Y ESTADÍSTICAS
# ============================================================
st.markdown('<div class="section-title">📈 Análisis Visual y Estadísticas</div>', unsafe_allow_html=True)

layout_base = dict(
    plot_bgcolor="white", paper_bgcolor="white", 
    font=dict(family="Arial", size=18, color=COLORS["text_dark"]),
    margin=dict(l=20, r=20, t=70, b=20),
    title=dict(font=dict(size=22, color=COLORS["text_dark"]))
)

if "Hour" in df_res.columns and "User_ID" in df_res.columns:
    moda_hora = df_res["Hour"].mode().iloc[0] if not df_res["Hour"].mode().empty else 0
    cant_moda = len(df_res[df_res["Hour"] == moda_hora])
    pct_inicios_frecuentes = (cant_moda / total * 100) if total > 0 else 0

    c_stat1, c_stat2, c_stat3 = st.columns(3)
    with c_stat1: kpi_card("Total registros", f"{total:,}", "Evaluados", "alto")
    with c_stat2: kpi_card("Hora más frecuente", f"{moda_hora}:00 hrs", "Hora pico", "teal")
    with c_stat3: kpi_card("Inicios frecuentes", f"{pct_inicios_frecuentes:.2f}%", "Ocurren en la hora pico", "primary")
    st.write("")

# FILA 1: Top Usuarios y Árbol de Decisión
g1, g2 = st.columns(2)
with g1:
    st.markdown("<h3 style='font-size: 20px; font-weight:700;'>Top 10 Usuarios con más inicios</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px; color:#475569;'>Identifica a los usuarios que más veces ingresaron al sistema. Un volumen inusualmente alto puede indicar cuentas comprometidas o accesos indebidos.</p>", unsafe_allow_html=True)
    if "User_ID" in df_res.columns:
        top_users = df_res["User_ID"].value_counts().head(10).reset_index()
        top_users.columns = ["User_ID", "Inicios_Sesion"]
        top_users["User_ID"] = top_users["User_ID"].astype(str)
        fig_bar = px.bar(top_users, x="Inicios_Sesion", y="User_ID", orientation="h", color="Inicios_Sesion", color_continuous_scale="Blues")
        fig_bar.update_layout(**layout_base, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        fig_bar.update_xaxes(tickfont=dict(color=COLORS["text_dark"], size=18), title_font=dict(color=COLORS["text_dark"], size=20))
        fig_bar.update_yaxes(tickfont=dict(color=COLORS["text_dark"], size=18), title_font=dict(color=COLORS["text_dark"], size=20))
        st.plotly_chart(fig_bar, use_container_width=True)

with g2:
    st.markdown("<h3 style='font-size: 20px; font-weight:700;'>Árbol de Decisión</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 16px; color:#475569;'>Este árbol imita al modelo principal y muestra las reglas lógicas para clasificar un acceso.</p>", unsafe_allow_html=True)
    try:
        tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree_clf.fit(X_modelo, preds_modelo)
        
        fig_tree, ax_tree = plt.subplots(figsize=(15, 8))
        plot_tree(tree_clf, max_depth=3, 
                  feature_names=paquete['feature_cols'], 
                  class_names=['Normal', 'Anomalía'], 
                  filled=True, rounded=True, 
                  fontsize=12, ax=ax_tree, precision=1)
        plt.title("Reglas de Decisión del Modelo", fontsize=18, weight="bold", pad=20)
        plt.tight_layout()
        st.pyplot(fig_tree)
    except Exception as e:
        st.warning(f"Error generando el árbol: {e}")

# FILA 2: Histograma (Ocupa todo el ancho)
st.markdown("<h3 style='font-size: 20px; font-weight:700; margin-top: 20px;'>Distribución de Scores</h3>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px; color:#475569;'>La mayoría de los accesos deberían tener un score bajo (verde). Una gran cantidad de scores altos indica un incidente.</p>", unsafe_allow_html=True)
fig_hist = px.histogram(df_res, x="Score_Anomalia", color="Riesgo", color_discrete_map=RIESGO_HEX, category_orders={"Riesgo": RIESGO_ORDEN}, nbins=40)
fig_hist.update_layout(**layout_base, legend_title_text="")
fig_hist.update_xaxes(tickfont=dict(color=COLORS["text_dark"], size=18), title_font=dict(color=COLORS["text_dark"], size=20))
fig_hist.update_yaxes(tickfont=dict(color=COLORS["text_dark"], size=18), title_font=dict(color=COLORS["text_dark"], size=20))
st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================
# DESCARGAS Y PDF
# ============================================================
st.markdown('<div class="section-title">📥 Descargar Informe</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    csv_data = df_res.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar CSV Completo", csv_data, "resultados_completos.csv", "text/csv", use_container_width=True)

with c2:
    with st.spinner("Generando PDF con gráficos incrustados..."):
        import tempfile
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        sns.set_theme(style="whitegrid", rc={"axes.facecolor": "white", "figure.facecolor": "white"})
        imgs = {}

        # 1. Top Usuarios
        if "User_ID" in df_res.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            top_users = df_res["User_ID"].value_counts().head(10).reset_index()
            top_users.columns = ["User_ID", "Inicios"]
            sns.barplot(data=top_users, x="Inicios", y="User_ID", palette="Blues_r", ax=ax)
            ax.set_title("Top 10 Usuarios con más inicios de sesión", fontweight="bold", fontsize=14)
            ax.set_xlabel("Cantidad de Inicios", fontsize=12)
            ax.set_ylabel("Usuario", fontsize=12)
            plt.tight_layout()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(tmp.name, dpi=150)
            imgs["bar"] = tmp.name
            plt.close(fig)

        # 2. Árbol de Decisión
        try:
            fig_tree, ax_tree = plt.subplots(figsize=(14, 7))
            plot_tree(tree_clf, max_depth=3, 
                      feature_names=paquete['feature_cols'], 
                      class_names=['Normal', 'Anomalía'], 
                      filled=True, rounded=True, 
                      fontsize=11, ax=ax_tree, precision=1)
            plt.title("Reglas de Decisión del Modelo", fontsize=16, weight="bold", pad=20)
            plt.tight_layout()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig_tree.savefig(tmp.name, dpi=150)
            imgs["tree"] = tmp.name
            plt.close(fig_tree)
        except:
            pass

        # 3. Histograma
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(data=df_res, x="Score_Anomalia", hue="Riesgo", hue_order=RIESGO_ORDEN, palette=RIESGO_HEX, bins=40, multiple="stack", ax=ax)
        ax.set_title("Distribución de Scores de Anomalía", fontweight="bold", fontsize=14)
        ax.set_xlabel("Score", fontsize=12)
        ax.set_ylabel("Cantidad de Registros", fontsize=12)
        plt.tight_layout()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp.name, dpi=150)
        imgs["hist"] = tmp.name
        plt.close(fig)

        # --- CONSTRUCCIÓN DEL PDF ---
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        estilo_texto = styles["Normal"]
        estilo_texto.fontSize = 11
        
        riesgo_actual = "ALTO" if anom > 0 else "BAJO"
        
        story = [
            Paragraph("Reporte de Auditoría y Detección de Anomalías", styles["Heading1"]),
            Paragraph(f"<b>Fecha:</b> {datetime.now().strftime('%d/%m/%Y, %H:%M')}", estilo_texto),
            Paragraph("<b>Hospital:</b> MediData Health", estilo_texto),
            Paragraph("<b>Modelo:</b> Isolation Forest", estilo_texto),
            Spacer(1, 16),
            
            Paragraph("1. Resumen de la Situación", styles["Heading2"]),
            Paragraph(f"Durante el período analizado, se evaluaron un total de {total:,} registros de accesos a historiales clínicos.", estilo_texto),
            Paragraph(f"• <b>Accesos Normales:</b> {norm:,} ({(100-pct):.1f}%)", estilo_texto),
            Paragraph(f"• <b>Anomalías Detectadas:</b> {anom:,} ({pct:.1f}%)", estilo_texto),
            Spacer(1, 12),

            Paragraph("2. Métricas de Desempeño del Modelo", styles["Heading2"]),
            Paragraph("El modelo fue evaluado con los siguientes resultados globales:", estilo_texto),
            Spacer(1, 8)
        ]

        datos_metricas = [
            ["Métrica", "Valor"],
            ["Precision", f"{prec:.3f}"],
            ["Recall", f"{rec:.3f}"],
            ["F1-score", f"{f1:.3f}"],
            ["Accuracy", f"{acc:.3f}"]
        ]
        t_met = Table(datos_metricas, colWidths=[3*inch, 2*inch])
        t_met.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor(COLORS["primary_d"])),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
            ("ALIGN", (0,0), (1,-1), "CENTER"),
        ]))
        story.append(t_met)
        story.append(Spacer(1, 12))

        # --- PROTOCOLO HIPAA ---
        story.extend([
            Paragraph("3. Protocolo de Notificación (HIPAA)", styles["Heading2"]),
            Paragraph("Se ha establecido que este Reporte Ejecutivo PDF es el documento oficial ante el Compliance Officer. Si el sistema detecta un riesgo <b>CRÍTICO</b>, el equipo de analistas tiene la obligación de presentar este reporte en un máximo de 24 horas.", estilo_texto),
            Spacer(1, 12),
            
            Paragraph("4. Privacidad y Seudonimización", styles["Heading2"]),
            Paragraph("El modelo no procesa nombres reales ni Información de Salud Protegida (PHI). Utiliza identificadores para garantizar la privacidad total.", estilo_texto),
            Spacer(1, 12),
        ])

        # --- CONCLUSIÓN ---
        story.extend([
            Paragraph("5. Conclusión y Próximos Pasos", styles["Heading2"]),
            Paragraph(f"El nivel de riesgo actual es <b>{riesgo_actual}</b>. Se recomienda al equipo de Cumplimiento revisar el listado de usuarios con riesgo CRÍTICO.", estilo_texto),
            Spacer(1, 16)
        ])

        # Tabla Resumen
        data = [["Métrica", "Valor"],
                ["Total registros", f"{total:,}".replace(",", ".")],
                ["Anomalías totales", f"{anom:,}".replace(",", ".")],
                ["Registros normales", f"{norm:,}".replace(",", ".")]]
                
        t = Table(data, colWidths=[3*inch, 2*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor(COLORS["primary_d"])),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
            ("ALIGN", (0,0), (-1,-1), "CENTER")
        ]))
        story.append(t)
        
        # TABLA DE USUARIOS BLOQUEADOS (CRÍTICOS)
        story.append(Spacer(1, 20))
        story.append(Paragraph("6. Registro de Usuarios Bloqueados (Riesgo Crítico)", styles["Heading2"]))
        story.append(Paragraph("Los siguientes usuarios presentaron accesos con anomalías críticas:", estilo_texto))
        story.append(Spacer(1, 10))

        df_criticos = df_res[df_res["Riesgo"] == "🔴 CRÍTICO"]
        if not df_criticos.empty:
            cols_pdf = ["User_ID", "User_Role", "Access_Type", "Timestamp", "Score_Anomalia"]
            cols_pdf = [c for c in cols_pdf if c in df_criticos.columns]
            
            table_data = [cols_pdf]
            for _, row in df_criticos.head(20).iterrows():
                fila = []
                for c in cols_pdf:
                    val = row[c]
                    if isinstance(val, float): fila.append(f"{val:.3f}")
                    else: fila.append(str(val))
                table_data.append(fila)

            t_usr = Table(table_data, colWidths=[1*inch, 1*inch, 1*inch, 1.5*inch, 1*inch])
            t_usr.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DC2626")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE", (0,0), (-1,-1), 9),
                ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
                ("ALIGN", (0,0), (-1,-1), "CENTER")
            ]))
            story.append(t_usr)
            story.append(Paragraph(f"<i>* Mostrando los primeros 20 de {len(df_criticos):,} registros críticos totales.</i>", estilo_texto))
        else:
            story.append(Paragraph("No se detectaron usuarios con nivel de riesgo Crítico.", estilo_texto))

        # ANEXOS VISUALES
        story.append(Spacer(1, 30))
        story.append(Paragraph("7. Anexos Visuales (Gráficos Estadísticos)", styles["Heading2"]))
        story.append(Spacer(1, 12))

        if "bar" in imgs:
            story.append(Paragraph("<b>Top 10 Usuarios:</b> Identifica usuarios con volumen inusualmente alto de inicios de sesión.", estilo_texto))
            story.append(RLImage(imgs["bar"], width=6.5*inch, height=3.2*inch))
            story.append(Spacer(1, 15))
            
        if "tree" in imgs:
            story.append(Paragraph("<b>Reglas del Modelo (Árbol):</b> Muestra la ruta lógica que sigue el modelo para clasificar un acceso.", estilo_texto))
            story.append(RLImage(imgs["tree"], width=6.5*inch, height=3.5*inch))
            story.append(Spacer(1, 15))
            
        if "hist" in imgs:
            story.append(Paragraph("<b>Distribución de Scores:</b> Concentración de accesos normales vs la cola de anomalías.", estilo_texto))
            story.append(RLImage(imgs["hist"], width=6.5*inch, height=3.2*inch))

        doc.build(story)
        buf.seek(0)
        st.download_button("📄 Descargar Reporte Ejecutivo (PDF)", buf.getvalue(), "reporte_ejecutivo.pdf", "application/pdf", use_container_width=True)
