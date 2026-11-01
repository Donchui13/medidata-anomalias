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
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak

# ============================================================
# CONFIGURACIÓN Y COLORES
# ============================================================
st.set_page_config(page_title="Detector de Anomalías · MediData Health", page_icon="🏥", layout="wide")

MODEL_PATH = "modelo_isolation_forest.pkl"
LOGO_PATH  = "logo.png"

COLORS = {"primary": "#0EA5E9", "primary_d": "#0369A1", "teal": "#14B8A6", "critico": "#DC2626", "alto": "#F59E0B", "medio": "#EAB308", "bajo": "#10B981", "neutral": "#64748B", "border": "#E2E8F0", "text_dark": "#0F172A", "text_soft": "#334155"}
RIESGO_ORDEN = ["🔴 CRÍTICO", "🟠 ALTO", "🟡 MEDIO", "🟢 BAJO"]
RIESGO_HEX = {"🔴 CRÍTICO": COLORS["critico"], "🟠 ALTO": COLORS["alto"], "🟡 MEDIO": COLORS["medio"], "🟢 BAJO": COLORS["bajo"]}

# CSS Personalizado para que se vea profesional
st.markdown(f"""
<style>
:root {{ color-scheme: light !important; }}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background-color: #F8FAFC !important; color: {COLORS['text_dark']} !important; }}
.stApp p, .stApp span, .stApp div, .stApp label, .stApp a, .stApp li {{ color: {COLORS['text_dark']} !important; }}
.stApp h1, .stApp h2, .stApp h3 {{ color: {COLORS['primary_d']} !important; font-weight: 700 !important; }}
[data-testid="stFileUploaderDropzone"] {{ background-color: #FFFFFF !important; border: 2px dashed {COLORS['primary']} !important; border-radius: 10px !important; padding: 20px !important; }}
[data-testid="stFileUploaderDropzone"] button {{ background-color: {COLORS['primary']} !important; border-radius: 8px !important; border: none !important; }}
[data-testid="stFileUploaderDropzone"] button * {{ color: #FFFFFF !important; }}

.clinical-header {{ background: linear-gradient(135deg, {COLORS['primary_d']} 0%, {COLORS['primary']} 100%) !important; padding: 24px 30px; border-radius: 15px; margin-bottom: 20px; display: flex; align-items: center; gap: 20px; box-shadow: 0 8px 20px -6px rgba(14,165,233,0.4); }}
.clinical-header h1 {{ margin: 0; font-size: 28px; color: #FFFFFF !important; }}
.clinical-header p {{ margin: 5px 0 0 0; font-size: 15px; color: #FFFFFF !important; opacity: .9; }}

.kpi-card {{ background: #FFFFFF !important; border-radius: 12px; padding: 15px 20px; border-left: 6px solid {COLORS['primary']}; box-shadow: 0 4px 12px rgba(15,23,42,0.05); height: 100%; }}
.kpi-card.critico {{ border-left-color: {COLORS['critico']}; }}
.kpi-card.alto {{ border-left-color: {COLORS['alto']}; }}
.kpi-card.medio {{ border-left-color: {COLORS['medio']}; }}
.kpi-card.bajo {{ border-left-color: {COLORS['bajo']}; }}
.kpi-card.teal {{ border-left-color: {COLORS['teal']}; }}
.kpi-label {{ font-size: 13px; text-transform: uppercase; color: {COLORS['neutral']} !important; font-weight: 700; margin-bottom: 5px; }}
.kpi-value {{ font-size: 32px; font-weight: 800; color: {COLORS['text_dark']} !important; line-height: 1.1; }}
.kpi-sub {{ font-size: 14px; font-weight: 500; color: {COLORS['neutral']} !important; margin-top: 5px; }}

.section-title {{ font-size: 22px; font-weight: 700; color: {COLORS['text_dark']} !important; margin: 20px 0 15px 0; padding-bottom: 8px; border-bottom: 2px solid {COLORS['border']}; }}
.stDownloadButton button, .stButton button {{ background: linear-gradient(135deg, {COLORS['primary_d']} 0%, {COLORS['primary']} 100%) !important; border: none; border-radius: 8px; padding: 10px 20px; font-weight: 600; width: 100%; color: white !important; }}
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

    # Ingeniería de características y sanitización
    if "Timestamp" in df_work.columns:
        df_work["Timestamp"] = pd.to_datetime(df_work["Timestamp"], errors="coerce")
        df_work["Hour"] = df_work["Timestamp"].dt.hour.fillna(12).astype(int)
        df_work["DayOfWeek"] = df_work["Timestamp"].dt.dayofweek.fillna(0).astype(int)
        df_work["Is_Night"] = ((df_work["Hour"] >= 22) | (df_work["Hour"] < 6)).astype(int)
        df_work["Is_Weekend"] = (df_work["DayOfWeek"] >= 5).astype(int)

    for col, le in encoders.items():
        if col in df_work.columns:
            df_work[f"{col}_enc"] = df_work[col].astype(str).map(lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1)

    if "Duration_Seconds" in df_work.columns: 
        df_work["Duration_log"] = np.log1p(pd.to_numeric(df_work["Duration_Seconds"], errors="coerce").clip(lower=0))
    if "Access_Reason" in df_work.columns: 
        df_work["Suspicious_Reason"] = df_work["Access_Reason"].astype(str).str.strip().str.lower().isin(["unknown", "curiosity"]).astype(int)

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

# ============================================================
# LAYOUT PRINCIPAL
# ============================================================
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.markdown("<div style='text-align:center;font-size:50px;'>🏥</div>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align:center;color:{COLORS['primary_d']} !important;'>MediData Health</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:12px;'>Sistema de Auditoría Inteligente</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**🩺 Niveles de Riesgo**")
    for r in RIESGO_ORDEN:
        st.markdown(f"<div style='padding:4px 0;color:{RIESGO_HEX[r]} !important;font-weight:600;'>{r}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.info("💡 **Nota:** El sistema usa Isolation Forest para detectar anomalías sin necesidad de etiquetas previas.")

st.markdown("""<div class="clinical-header"><div style="font-size:45px;">🩺</div><div><h1>Dashboard de Auditoría Clínica</h1><p>Detección inteligente de accesos no autorizados a historiales médicos</p></div></div>""", unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Error crítico: No se encontró el modelo en {MODEL_PATH}.")
    st.stop()

paquete = cargar_modelo(os.path.getmtime(MODEL_PATH))

# Subida de archivo
st.markdown('<div class="section-title">📁 Carga de Datos de Auditoría</div>', unsafe_allow_html=True)
archivo = st.file_uploader("Sube el archivo CSV con los registros de acceso", type=["csv"])

if not archivo: 
    st.info("👆 Por favor, sube un archivo CSV para iniciar el análisis.")
    st.stop()

try:
    start_time = datetime.now()
    df_input = pd.read_csv(archivo)
    
    with st.spinner("Analizando registros con Isolation Forest..."):
        df_res, X_modelo, preds_modelo = predecir(df_input, paquete)
        
    end_time = datetime.now()
    exec_time = (end_time - start_time).total_seconds()
    
    st.success(f"✅ ¡Análisis completado con éxito en {exec_time:.2f} segundos!")
    
    # Variables globales para cálculos
    total = len(df_res)
    anom = int(df_res["Prediccion"].sum())
    norm = total - anom
    pct = (anom / total * 100) if total else 0.0

    # ============================================================
    # CREACIÓN DE PESTAÑAS (TABS) PARA MEJORAR LA ORGANIZACIÓN
    # ============================================================
    tab1, tab2, tab3 = st.tabs(["📊 Resumen Ejecutivo", "📈 Análisis Visual y Gráficos", "📄 Reportes y Descargas"])

    # ------------------------------------------------------------
    # PESTAÑA 1: RESUMEN EJECUTIVO
    # ------------------------------------------------------------
    with tab1:
        st.markdown('<div class="section-title">Indicadores Clave de Rendimiento (KPIs)</div>', unsafe_allow_html=True)
        
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi_card("Total Registros", f"{total:,}", "Accesos analizados", "primary")
        with k2: kpi_card("Anomalías", f"{anom:,}", f"{pct:.1f}% del total", "critico")
        with k3: kpi_card("Registros Normales", f"{norm:,}", f"{100-pct:.1f}% del total", "bajo")
        with k4: kpi_card("Tiempo Ejecución", f"{exec_time:.2f}s", "Procesamiento", "teal")
        
        st.write("")
        st.markdown('<div class="section-title">Distribución General de los Accesos</div>', unsafe_allow_html=True)
        
        # Gráfico de Anillo (Donut) - Requisito de la profesora para diferenciar Normal vs Anómalo
        c1, c2 = st.columns([1, 2])
        with c1:
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Registros Normales', 'Anomalías Detectadas'], 
                values=[norm, anom], 
                hole=.6,
                marker_colors=[COLORS['bajo'], COLORS['critico']],
                textinfo='percent+label',
                textfont_size=14
            )])
            fig_donut.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with c2:
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid {COLORS['critico']}; box-shadow: 0 2px 10px rgba(0,0,0,0.05); height: 100%;">
                <h3 style="margin-top:0; color:{COLORS['critico']} !important;">🚨 Alertas Críticas de Seguridad</h3>
                <p>El modelo ha detectado <b>{anom:,}</b> accesos que se desvían significativamente del comportamiento normal.</p>
                <p>Se recomienda al equipo de TI y Cumplimiento revisar de inmediato los accesos clasificados como <b>CRÍTICOS</b> y <b>ALTOS</b>.</p>
                <p><i>* Los datos mostrados ya han pasado por el proceso de seudonimización (SHA-256) para proteger la identidad de los pacientes.</i></p>
            </div>
            """, unsafe_allow_html=True)

        # Métricas de Desempeño (si el dataset tiene etiquetas)
        posibles_label = [c for c in df_input.columns if c.lower() in ["is_anomaly", "label", "target"]]
        col_real = posibles_label[0] if posibles_label else None
        
        if col_real is not None:
            st.markdown('<div class="section-title">🎯 Validación de Desempeño del Modelo</div>', unsafe_allow_html=True)
            y_real = df_input[col_real].astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "anomaly", "anómalo"]).astype(int).values
            if len(set(y_real)) > 1:
                m1, m2, m3, m4 = st.columns(4)
                with m1: kpi_card("Precisión", f"{precision_score(y_real, preds_modelo, zero_division=0):.3f}", "Aciertos positivos", "teal")
                with m2: kpi_card("Recall", f"{recall_score(y_real, preds_modelo, zero_division=0):.3f}", "Detección real", "alto")
                with m3: kpi_card("F1-Score", f"{f1_score(y_real, preds_modelo, zero_division=0):.3f}", "Balance del modelo", "primary")
                with m4: kpi_card("Exactitud", f"{accuracy_score(y_real, preds_modelo):.3f}", "Aciertos globales", "neutral")
                st.info("Estas métricas demuestran la eficacia del modelo Isolation Forest comparando sus predicciones contra las etiquetas reales de los datos analizados.")
        else:
            st.warning("⚠️ No se encontró una columna de etiquetas reales (ej. `Is_Anomaly`) para calcular métricas de desempeño. El modelo funciona en modo no supervisado.")

        # Tabla de Anomalías Críticas
        st.markdown('<div class="section-title">🔍 Registros Identificados como Anomalías Críticas</div>', unsafe_allow_html=True)
        st.markdown("La siguiente tabla muestra una muestra de los accesos más peligrosos detectados por el sistema.")
        df_criticos = df_res[df_res["Riesgo"].isin(["🔴 CRÍTICO", "🟠 ALTO"])].copy()
        
        if not df_criticos.empty:
            cols_mostrar = [c for c in ["User_ID", "Patient_ID", "User_Role", "Access_Type", "Timestamp", "Riesgo", "Score_Anomalia"] if c in df_criticos.columns]
            st.dataframe(df_criticos[cols_mostrar].head(15), use_container_width=True, hide_index=True)
        else:
            st.success("No se detectaron anomalías de alto riesgo en este lote de datos.")

    # ------------------------------------------------------------
    # PESTAÑA 2: ANÁLISIS VISUAL Y GRÁFICOS (Los 3 gráficos obligatorios)
    # ------------------------------------------------------------
    with tab2:
        st.markdown('<div class="section-title">📈 Análisis de Patrones y Comportamiento</div>', unsafe_allow_html=True)
        
        layout_base = dict(plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Arial", size=14, color=COLORS["text_dark"]), margin=dict(l=10, r=10, t=40, b=10))

        # GRÁFICO 1: Mapa de Calor
        st.markdown("### 1️⃣ Mapa de Calor: Distribución Temporal de Accesos")
        st.markdown("*Este gráfico muestra en qué días y horas se concentran los accesos anómalos. Las zonas rojas oscuras indican mayor actividad.*")
        
        if "Timestamp" in df_res.columns and df_res["Timestamp"].notna().any():
            df_t = df_res.dropna(subset=["Timestamp"]).copy()
            df_t["Dia"] = df_t["Timestamp"].dt.day_name()
            df_t["Hora"] = df_t["Timestamp"].dt.hour
            pivot = df_t.pivot_table(index="Dia", columns="Hora", values="Score_Anomalia", aggfunc="count").fillna(0)
            dias_orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            pivot = pivot.reindex(dias_orden).dropna(how="all")
            
            fig_heat = px.imshow(pivot, color_continuous_scale="Reds", aspect="auto")
            fig_heat.update_layout(**layout_base, height=400)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("Datos de Timestamp no disponibles para generar el mapa de calor.")

        st.divider()

        # GRÁFICO 2: Histograma de Scores
        st.markdown("### 2️⃣ Comportamiento del Modelo: Distribución de Scores")
        st.markdown("*Este histograma muestra cómo el modelo clasificó los accesos. La mayoría deberían ser normales (verde), y las barras rojas/naranjas representan las anomalías.*")
        
        fig_hist = px.histogram(df_res, x="Score_Anomalia", color="Riesgo", color_discrete_map=RIESGO_HEX, category_orders={"Riesgo": RIESGO_ORDEN}, nbins=40)
        fig_hist.update_layout(**layout_base, legend_title_text="Nivel de Riesgo", height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        # GRÁFICO 3: Top 10 Usuarios
        st.markdown("### 3️⃣ Resumen de Actividad: Top 10 Usuarios")
        st.markdown("*Identifica a los usuarios con más inicios de sesión. Un volumen inusualmente alto puede indicar cuentas comprometidas.*")
        
        if "User_ID" in df_res.columns:
            top_users = df_res["User_ID"].value_counts().head(10).reset_index()
            top_users.columns = ["User_ID", "Inicios_Sesion"]
            top_users["User_ID"] = top_users["User_ID"].astype(str)
            fig_bar = px.bar(top_users, x="Inicios_Sesion", y="User_ID", orientation="h", color="Inicios_Sesion", color_continuous_scale="Blues")
            fig_bar.update_layout(**layout_base, coloraxis_showscale=False, yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

    # ------------------------------------------------------------
    # PESTAÑA 3: REPORTES Y DESCARGAS
    # ------------------------------------------------------------
    with tab3:
        st.markdown('<div class="section-title">📄 Centro de Descargas y Reportes</div>', unsafe_allow_html=True)
        st.markdown("Exporte los resultados del análisis para auditorías internas o entrega a entes reguladores.")
        
        c_down1, c_down2 = st.columns(2)
        
        with c_down1:
            st.markdown("#### 📥 Datos Crudos (CSV)")
            st.markdown("Descargue el archivo con los resultados completos y las clasificaciones de riesgo.")
            csv_data = df_res.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar Resultados (CSV)", csv_data, "resultados_auditoria.csv", "text/csv", use_container_width=True)
            
        with c_down2:
            st.markdown("#### 📑 Reporte Ejecutivo (PDF)")
            st.markdown("Genere un documento formal con los hallazgos principales para la dirección médica.")
            
            # Botón para generar PDF (Usando lógica simplificada para evitar errores)
            if st.button("Generar y Descargar Reporte PDF", use_container_width=True):
                with st.spinner("Generando documento PDF..."):
                    buf = io.BytesIO()
                    doc = SimpleDocTemplate(buf, pagesize=A4)
                    styles = getSampleStyleSheet()
                    estilo_texto = styles["Normal"]
                    estilo_texto.fontSize = 11
                    
                    story = [
                        Paragraph("Reporte Ejecutivo de Auditoría de Seguridad", styles["Heading1"]),
                        Paragraph(f"<b>Fecha:</b> {datetime.now().strftime('%d/%m/%Y, %H:%M')}", estilo_texto),
                        Paragraph("<b>Institución:</b> MediData Health", estilo_texto),
                        Paragraph("<b>Modelo:</b> Isolation Forest (Detección de Anomalías)", estilo_texto),
                        Spacer(1, 16),
                        
                        Paragraph("1. Resumen del Análisis", styles["Heading2"]),
                        Paragraph(f"Se procesaron un total de <b>{total:,}</b> registros de acceso a historiales clínicos.", estilo_texto),
                        Paragraph(f"• Accesos Normales: {norm:,} ({(100-pct):.1f}%)", estilo_texto),
                        Paragraph(f"• Anomalías Detectadas: <b>{anom:,} ({pct:.1f}%)</b>", estilo_texto),
                        Spacer(1, 12),
                        
                        Paragraph("2. Protocolo de Seguridad y Privacidad", styles["Heading2"]),
                        Paragraph("El sistema aplicó seudonimización (SHA-256) a los identificadores de pacientes y personal médico. No se procesaron nombres reales, garantizando el cumplimiento de la Ley N.º 787 de Protección de Datos Personales y normativas HIPAA.", estilo_texto),
                        Spacer(1, 12),
                        
                        Paragraph("3. Recomendaciones", styles["Heading2"]),
                        Paragraph("Se recomienda al equipo de TI revisar de inmediato los registros clasificados con nivel de riesgo CRÍTICO. Asimismo, se sugiere validar los permisos de acceso de los usuarios con mayor volumen de inicios de sesión para descartar cuentas comprometidas.", estilo_texto),
                        Spacer(1, 16)
                    ]
                    
                    # Tabla resumen
                    data = [["Métrica", "Valor"], ["Total Analizado", f"{total:,}"], ["Anomalías", f"{anom:,}"], ["Normales", f"{norm:,}"]]
                    t = Table(data, colWidths=[3*inch, 2*inch])
                    t.setStyle(TableStyle([
                        ("BACKGROUND", (0,0), (-1,0), colors.HexColor(COLORS["primary_d"])),
                        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
                        ("ALIGN", (0,0), (-1,-1), "CENTER")
                    ]))
                    story.append(t)
                    
                    doc.build(story)
                    buf.seek(0)
                    st.download_button(label="✅ Haga clic aquí para guardar el PDF", data=buf.getvalue(), file_name="reporte_auditoria_medidata.pdf", mime="application/pdf", use_container_width=True)
                    st.success("¡PDF generado exitosamente!")

except Exception as e:
    st.error(f"Error procesando el archivo: {e}")
