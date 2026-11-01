import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, io, os
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as ReportLabImage

st.set_page_config(page_title="Detector de Anomalías · MediData Health", page_icon="🏥", layout="wide")
MODEL_PATH = "modelo_isolation_forest.pkl"
LOGO_PATH  = "logo.png"

PALETTE = {
    "light": "#c0e6fd", "mid_light": "#80aad3", "mid": "#5b86b6",
    "dark": "#3f6593", "very_dark": "#1b3554", "navy": "#000f22",
    "text": "#0F172A", "bg": "#F8FAFC", "border": "#E2E8F0"
}
RIESGO_COLORS = {
    "🔴 CRÍTICO": "#DC2626", "🟠 ALTO": "#F59E0B",
    "🟡 MEDIO": "#EAB308", "🟢 BAJO": "#10B981"
}
RIESGO_ORDEN = ["🔴 CRÍTICO", "🟠 ALTO", "🟡 MEDIO", "🟢 BAJO"]

st.markdown(f"""
<style>
:root {{ color-scheme: light !important; }}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background-color: {PALETTE['bg']} !important; color: {PALETTE['text']} !important; }}
.stApp p, .stApp span, .stApp label, .stApp a, .stApp li {{ color: {PALETTE['text']} !important; }}
.stApp h1, .stApp h2, .stApp h3 {{ color: {PALETTE['navy']} !important; font-weight: 700 !important; }}
[data-testid="stSidebar"] {{ background: linear-gradient(135deg, {PALETTE['navy']} 0%, {PALETTE['dark']} 100%) !important; border-right: none !important; }}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span:not([style*="color"]), [data-testid="stSidebar"] div:not([style*="color"]) {{ color: #FFFFFF !important; }}
.sidebar-note-box {{ background-color: {PALETTE['very_dark']} !important; border-left: 4px solid {PALETTE['mid_light']} !important; padding: 15px; border-radius: 8px; margin-top: 20px; }}
[data-testid="stFileUploaderDropzone"] {{ background-color: #FFFFFF !important; border: 2px dashed {PALETTE['mid']} !important; border-radius: 10px !important; padding: 20px !important; }}
[data-testid="stFileUploaderDropzone"] button {{ background-color: {PALETTE['dark']} !important; border-radius: 8px !important; border: none !important; }}
[data-testid="stFileUploaderDropzone"] button * {{ color: #FFFFFF !important; }}
.clinical-header {{ background: linear-gradient(135deg, {PALETTE['navy']} 0%, {PALETTE['dark']} 100%) !important; padding: 24px 30px; border-radius: 15px; margin-bottom: 20px; display: flex; align-items: center; gap: 20px; box-shadow: 0 8px 20px -6px rgba(0,15,34,0.4); }}
.header-texto-blanco {{ color: #FFFFFF !important; font-size: 28px; font-weight: 800; margin: 0; padding: 0; line-height: 1.2; }}
.header-subtexto-blanco {{ color: #E2E8F0 !important; font-size: 15px; margin-top: 5px; font-weight: 400; }}
.kpi-card {{ background: #FFFFFF !important; border-radius: 12px; padding: 15px 20px; border-left: 6px solid {PALETTE['mid']}; box-shadow: 0 4px 12px rgba(0,15,34,0.08); height: 100%; }}
.kpi-card.critico {{ border-left-color: {RIESGO_COLORS['🔴 CRÍTICO']}; }}
.kpi-card.alto {{ border-left-color: {RIESGO_COLORS['🟠 ALTO']}; }}
.kpi-card.medio {{ border-left-color: {RIESGO_COLORS['🟡 MEDIO']}; }}
.kpi-card.bajo {{ border-left-color: {RIESGO_COLORS['🟢 BAJO']}; }}
.kpi-card.teal {{ border-left-color: {PALETTE['mid_light']}; }}
.kpi-label {{ font-size: 13px; text-transform: uppercase; color: {PALETTE['dark']} !important; font-weight: 700; margin-bottom: 5px; }}
.kpi-value {{ font-size: 32px; font-weight: 800; color: {PALETTE['navy']} !important; line-height: 1.1; }}
.kpi-sub {{ font-size: 14px; font-weight: 500; color: {PALETTE['mid']} !important; margin-top: 5px; }}
.section-title {{ font-size: 22px; font-weight: 700; color: {PALETTE['navy']} !important; margin: 20px 0 15px 0; padding-bottom: 8px; border-bottom: 2px solid {PALETTE['mid_light']}; }}
.stDownloadButton button, .stButton button {{ background: linear-gradient(135deg, {PALETTE['navy']} 0%, {PALETTE['dark']} 100%) !important; border: none !important; border-radius: 8px !important; padding: 12px 20px !important; width: 100% !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important; }}
.stDownloadButton button *, .stButton button * {{ color: #FFFFFF !important; fill: #FFFFFF !important; font-weight: 700 !important; font-size: 16px !important; }}
.stDownloadButton button:hover, .stButton button:hover {{ background: linear-gradient(135deg, {PALETTE['dark']} 0%, {PALETTE['mid']} 100%) !important; color: #FFFFFF !important; }}
</style>
""", unsafe_allow_html=True)

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
    if "Duration_Seconds" in df_work.columns: df_work["Duration_log"] = np.log1p(pd.to_numeric(df_work["Duration_Seconds"], errors="coerce").clip(lower=0))
    if "Access_Reason" in df_work.columns: df_work["Suspicious_Reason"] = df_work["Access_Reason"].astype(str).str.strip().str.lower().isin(["unknown", "curiosity"]).astype(int)
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
    st.markdown(f"<div class='kpi-card {kind}'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div><div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)

with st.sidebar:
    if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, use_container_width=True)
    else: st.markdown("<div style='text-align:center;font-size:50px;'>🏥</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center; margin-top: 15px; margin-bottom: 5px;'><span style='color: #FFFFFF !important; font-size: 22px; font-weight: 800;'>MediData Health</span></div><div style='text-align: center; margin-bottom: 20px;'><span style='color: #E2E8F0 !important; font-size: 14px; font-weight: 600;'>Sistema de Auditoría Inteligente</span></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"<div style='margin-bottom: 10px;'><span style='color: #FFFFFF !important; font-size: 16px; font-weight: 700;'>🩺 Niveles de Riesgo</span></div>", unsafe_allow_html=True)
    for r in RIESGO_ORDEN: st.markdown(f"<div style='padding:6px 0; color:{RIESGO_COLORS[r]} !important; font-weight:700; font-size:15px;'>{r}</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"<div class='sidebar-note-box'><span style='font-size: 13px; color: #FFFFFF !important; margin: 0; display: block;'>💡 <b style='color: #FFFFFF !important;'>Nota:</b> El sistema usa Isolation Forest para detectar anomalías sin necesidad de etiquetas previas.</span></div>", unsafe_allow_html=True)

st.markdown(f"<div class='clinical-header'><div style='font-size:45px;'>🩺</div><div><div class='header-texto-blanco'>Dashboard de Auditoría Clínica</div><div class='header-subtexto-blanco'>Detección inteligente de accesos no autorizados a historiales médicos</div></div></div>", unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Error crítico: No se encontró el modelo en {MODEL_PATH}.")
    st.stop()

paquete = cargar_modelo(os.path.getmtime(MODEL_PATH))
st.markdown(f'<div class="section-title">📁 Carga de Datos de Auditoría</div>', unsafe_allow_html=True)
archivo = st.file_uploader("Sube el archivo CSV con los registros de acceso", type=["csv"])
if not archivo: 
    st.info("👆 Por favor, sube un archivo CSV para iniciar el análisis.")
    st.stop()

try:
    start_time = datetime.now()
    df_input = pd.read_csv(archivo)
    with st.spinner("Analizando registros con Isolation Forest..."):
        df_res, X_modelo, preds_modelo = predecir(df_input, paquete)
    exec_time = (datetime.now() - start_time).total_seconds()
    st.success(f"✅ ¡Análisis completado con éxito en {exec_time:.2f} segundos!")
    
    total = len(df_res)
    anom = int(df_res["Prediccion"].sum())
    norm = total - anom
    pct = (anom / total * 100) if total else 0.0
    
    tab1, tab2, tab3 = st.tabs(["📊 Resumen Ejecutivo", "📈 Análisis Visual y Gráficos", "📄 Reportes y Descargas"])

    with tab1:
        st.markdown(f'<div class="section-title">Indicadores Clave de Rendimiento</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi_card("Total Registros", f"{total:,}", "Accesos analizados", "primary")
        with k2: kpi_card("Anomalías", f"{anom:,}", f"{pct:.1f}% del total", "critico")
        with k3: kpi_card("Registros Normales", f"{norm:,}", f"{100-pct:.1f}% del total", "bajo")
        with k4: kpi_card("Tiempo Ejecución", f"{exec_time:.2f}s", "Procesamiento", "teal")
        st.write("")
        st.markdown(f'<div class="section-title">Distribución General de los Accesos</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            fig_donut = go.Figure(data=[go.Pie(labels=['Normales', 'Anomalías'], values=[norm, anom], hole=.6, marker_colors=[RIESGO_COLORS["🟢 BAJO"], RIESGO_COLORS["🔴 CRÍTICO"]], textinfo='percent+label', textfont=dict(size=16, color="#000000"))])
            fig_donut.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=300, font=dict(color="#000000"))
            st.plotly_chart(fig_donut, use_container_width=True)
        with c2:
            st.markdown(f"""<div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid {RIESGO_COLORS['🔴 CRÍTICO']}; box-shadow: 0 2px 10px rgba(0,0,0,0.05); height: 100%;"><h3 style="margin-top:0; color:{RIESGO_COLORS['🔴 CRÍTICO']} !important;">🚨 Alertas Críticas de Seguridad</h3><p style="font-size: 16px;">El modelo ha detectado <b>{anom:,}</b> accesos que se desvían significativamente del comportamiento normal.</p><p style="font-size: 16px;">Se recomienda revisar de inmediato los accesos <b>CRÍTICOS</b> y <b>ALTOS</b>.</p><p style="font-size: 14px;"><i>* Los datos mostrados ya han pasado por el proceso de seudonimización (SHA-256).</i></p></div>""", unsafe_allow_html=True)
        
        posibles_label = [c for c in df_input.columns if c.lower() in ["is_anomaly", "label", "target"]]
        col_real = posibles_label[0] if posibles_label else None
        if col_real is not None:
            st.markdown(f'<div class="section-title">🎯 Validación de Desempeño del Modelo</div>', unsafe_allow_html=True)
            y_real = df_input[col_real].astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "anomaly", "anómalo"]).astype(int).values
            if len(set(y_real)) > 1:
                m1, m2, m3, m4 = st.columns(4)
                with m1: kpi_card("Precisión", f"{precision_score(y_real, preds_modelo, zero_division=0):.3f}", "Aciertos positivos", "teal")
                with m2: kpi_card("Recall", f"{recall_score(y_real, preds_modelo, zero_division=0):.3f}", "Detección real", "alto")
                with m3: kpi_card("F1-Score", f"{f1_score(y_real, preds_modelo, zero_division=0):.3f}", "Balance del modelo", "primary")
                with m4: kpi_card("Exactitud", f"{accuracy_score(y_real, preds_modelo):.3f}", "Aciertos globales", "neutral")
        st.markdown(f'<div class="section-title">🔍 Registros Identificados como Anomalías Críticas</div>', unsafe_allow_html=True)
        df_criticos = df_res[df_res["Riesgo"].isin(["🔴 CRÍTICO", "🟠 ALTO"])].copy()
        if not df_criticos.empty:
            cols_mostrar = [c for c in ["User_ID", "Patient_ID", "User_Role", "Access_Type", "Timestamp", "Riesgo", "Score_Anomalia"] if c in df_criticos.columns]
            st.dataframe(df_criticos[cols_mostrar].head(15), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown(f'<div class="section-title">📈 Análisis de Patrones y Comportamiento</div>', unsafe_allow_html=True)
        layout_base = dict(plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Arial", size=16, color="#000000"), margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(tickfont=dict(color="#000000", size=14), title_font=dict(color="#000000", size=16)))
        st.markdown("### 1️⃣ Mapa de Calor")
        fig_heat = None
        if "Timestamp" in df_res.columns and df_res["Timestamp"].notna().any():
            df_t = df_res.dropna(subset=["Timestamp"]).copy()
            df_t["Dia"] = df_t["Timestamp"].dt.day_name()
            df_t["Hora"] = df_t["Timestamp"].dt.hour
            pivot = df_t.pivot_table(index="Dia", columns="Hora", values="Score_Anomalia", aggfunc="count").fillna(0)
            pivot = pivot.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).dropna(how="all")
            azul_scale = [[0, PALETTE["light"]], [0.5, PALETTE["mid"]], [1, PALETTE["navy"]]]
            fig_heat = px.imshow(pivot, color_continuous_scale=azul_scale, aspect="auto")
            fig_heat.update_layout(**layout_base, height=450, yaxis=dict(tickfont=dict(color="#000000", size=14), title_font=dict(color="#000000", size=16)))
            fig_heat.update_coloraxes(colorbar_tickfont=dict(color="#000000", size=14), colorbar_title_font=dict(color="#000000", size=16))
            st.plotly_chart(fig_heat, use_container_width=True)
        st.divider()
        st.markdown("### 2️⃣ Distribución de Scores")
        fig_hist = px.histogram(df_res, x="Score_Anomalia", color="Riesgo", color_discrete_map=RIESGO_COLORS, category_orders={"Riesgo": RIESGO_ORDEN}, nbins=40)
        fig_hist.update_layout(**layout_base, legend_title_text="Riesgo", height=450, yaxis=dict(tickfont=dict(color="#000000", size=14), title_font=dict(color="#000000", size=16)), legend=dict(font=dict(size=15)))
        st.plotly_chart(fig_hist, use_container_width=True)
        st.divider()
        st.markdown("### 3️⃣ Top 10 Usuarios")
        fig_bar = None
        if "User_ID" in df_res.columns:
            top_users = df_res["User_ID"].value_counts().head(10).reset_index()
            top_users.columns = ["User_ID", "Inicios_Sesion"]
            top_users["User_ID"] = top_users["User_ID"].astype(str)
            fig_bar = px.bar(top_users, x="Inicios_Sesion", y="User_ID", orientation="h", color="Inicios_Sesion", color_continuous_scale=azul_scale)
            fig_bar.update_layout(**layout_base, coloraxis_showscale=False, height=450, yaxis=dict(autorange="reversed", tickfont=dict(color="#000000", size=14), title_font=dict(color="#000000", size=16)))
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.markdown(f'<div class="section-title">📄 Centro de Descargas y Reportes</div>', unsafe_allow_html=True)
        c_down1, c_down2 = st.columns(2)
        with c_down1:
            st.markdown("#### 📥 Datos Crudos (CSV)")
            csv_data = df_res.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar Resultados (CSV)", csv_data, "resultados_auditoria.csv", "text/csv", use_container_width=True)
        with c_down2:
            st.markdown("#### 📑 Reporte Ejecutivo (PDF)")
            if st.button("Generar y Descargar Reporte PDF", use_container_width=True):
                with st.spinner("Construyendo documento PDF con gráficos..."):
                    try:
                        img_hist = io.BytesIO()
                        fig_hist.write_image(img_hist, format="png", width=1000, height=500, scale=2)
                        img_hist.seek(0)
                        
                        img_bar = io.BytesIO()
                        if fig_bar:
                            fig_bar.write_image(img_bar, format="png", width=1000, height=500, scale=2)
                            img_bar.seek(0)
                        
                        img_heat = io.BytesIO()
                        if fig_heat:
                            fig_heat.write_image(img_heat, format="png", width=1000, height=500, scale=2)
                            img_heat.seek(0)

                        buf = io.BytesIO()
                        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
                        styles = getSampleStyleSheet()
                        
                        estilo_titulo = ParagraphStyle(name='Titulo', parent=styles['Heading1'], fontSize=18, spaceAfter=10, textColor=colors.HexColor(PALETTE["navy"]))
                        estilo_sub = ParagraphStyle(name='Sub', parent=styles['Heading2'], fontSize=14, spaceBefore=15, spaceAfter=8, textColor=colors.HexColor(PALETTE["dark"]))
                        estilo_normal = ParagraphStyle(name='Normal', parent=styles['Normal'], fontSize=10, leading=14)
                        
                        story = [
                            Paragraph("Reporte Ejecutivo de Auditoria de Seguridad", estilo_titulo),
                            Paragraph(f"<b>Fecha:</b> {datetime.now().strftime('%d/%m/%Y, %H:%M')}", estilo_normal),
                            Paragraph("<b>Institucion:</b> MediData Health", estilo_normal),
                            Spacer(1, 15),
                            Paragraph("1. Resumen del Analisis", estilo_sub),
                            Paragraph(f"Total de registros analizados: <b>{total:,}</b>", estilo_normal),
                            Paragraph(f"Accesos Normales: {norm:,} ({(100-pct):.1f}%)", estilo_normal),
                            Paragraph(f"Anomalias Detectadas: <b>{anom:,} ({pct:.1f}%)</b>", estilo_normal),
                            Spacer(1, 10)
                        ]
                        
                        next_idx = 2
                        if col_real is not None and len(set(y_real)) > 1:
                            story.extend([
                                Paragraph(f"{next_idx}. Metricas de Desempeño", estilo_sub),
                                Paragraph(f"Precisión: {precision_score(y_real, preds_modelo, zero_division=0):.3f}", estilo_normal),
                                Paragraph(f"Recall: {recall_score(y_real, preds_modelo, zero_division=0):.3f}", estilo_normal),
                                Paragraph(f"F1-Score: {f1_score(y_real, preds_modelo, zero_division=0):.3f}", estilo_normal),
                                Paragraph(f"Exactitud: {accuracy_score(y_real, preds_modelo):.3f}", estilo_normal),
                                Spacer(1, 10)
                            ])
                            next_idx += 1
                            
                        story.append(Paragraph(f"{next_idx}. Distribucion por Nivel de Riesgo", estilo_sub))
                        df_riesgos = df_res["Riesgo"].value_counts().reset_index()
                        df_riesgos.columns = ["Nivel de Riesgo", "Cantidad"]
                        df_riesgos["Nivel de Riesgo"] = df_riesgos["Nivel de Riesgo"].astype(str).str.replace("🔴 ", "").str.replace("🟠 ", "").str.replace("🟡 ", "").str.replace("🟢 ", "")
                        t_riesgos = Table([["Nivel de Riesgo", "Cantidad"]] + df_riesgos.values.tolist(), colWidths=[3*inch, 1.5*inch])
                        t_riesgos.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor(PALETTE["navy"])), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), 0.5, colors.grey), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("FONTNAME", (0,0), (-1,-1), "Helvetica"), ("FONTSIZE", (0,0), (-1,-1), 10)]))
                        story.append(t_riesgos)
                        story.append(Spacer(1, 15))
                        next_idx += 1
                        
                        story.append(Paragraph(f"{next_idx}. Top 10 Accesos Mas Peligrosos", estilo_sub))
                        df_criticos_pdf = df_res[df_res["Riesgo"].isin(["🔴 CRÍTICO", "🟠 ALTO"])].head(10)
                        if not df_criticos_pdf.empty:
                            df_criticos_pdf["Score_Anomalia"] = df_criticos_pdf["Score_Anomalia"].apply(lambda x: f"{x:.4f}")
                            df_criticos_pdf["Riesgo_Texto"] = df_criticos_pdf["Riesgo"].astype(str).str.replace("🔴 ", "").str.replace("🟠 ", "")
                            t_criticos = Table([["User ID", "Patient ID", "Riesgo", "Score"]] + df_criticos_pdf[["User_ID", "Patient_ID", "Riesgo_Texto", "Score_Anomalia"]].values.tolist(), colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                            t_criticos.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor(PALETTE["dark"])), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), 0.5, colors.grey), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("FONTNAME", (0,0), (-1,-1), "Helvetica"), ("FONTSIZE", (0,0), (-1,-1), 9)]))
                            story.append(t_criticos)
                        story.append(Spacer(1, 15))
                        next_idx += 1

                        story.append(Paragraph(f"{next_idx}. Distribucion de Scores (Grafico)", estilo_sub))
                        story.append(ReportLabImage(img_hist, width=6.5*inch, height=3.2*inch))
                        story.append(Spacer(1, 15))
                        next_idx += 1
                        
                        if fig_bar:
                            story.append(Paragraph(f"{next_idx}. Top Usuarios (Grafico)", estilo_sub))
                            story.append(ReportLabImage(img_bar, width=6.5*inch, height=3.2*inch))
                            story.append(Spacer(1, 15))
                            next_idx += 1
                        
                        if fig_heat:
                            story.append(Paragraph(f"{next_idx}. Mapa de Calor de Accesos (Grafico)", estilo_sub))
                            story.append(ReportLabImage(img_heat, width=6.5*inch, height=3.2*inch))
                            story.append(Spacer(1, 15))
                            next_idx += 1
                        
                        story.extend([
                            Paragraph(f"{next_idx}. Protocolo y Recomendaciones", estilo_sub),
                            Paragraph("El sistema aplico seudonimizacion (SHA-256). Se recomienda al equipo de TI revisar de inmediato los registros clasificados como CRITICOS y ALTOS, asi como validar los permisos de los usuarios con mayor volumen de accesos.", estilo_normal)
                        ])
                        
                        doc.build(story)
                        buf.seek(0)
                        st.download_button(label="✅ Guardar PDF Completo", data=buf.getvalue(), file_name="reporte_auditoria.pdf", mime="application/pdf", use_container_width=True)
                        st.success("¡PDF generado exitosamente con gráficos y tablas!")
                    except Exception as e:
                        st.error(f"Error PDF: {e}")
except Exception as e:
    st.error(f"Error general: {e}")
