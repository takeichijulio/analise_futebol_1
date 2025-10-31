import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Análise de Futebol",page_icon="⚽🔥⭐" , layout="wide")
st.title("Análise De Futebol⚽🔥⭐")

@st.cache_data(show_spinner=False)
def load_df(path):
    return pd.read_csv(path)

# Carregar CSVs limpos (exportados anteriormente)
df1_s = load_df("df1_s_clean.csv")
df1_t = load_df("df1_t_clean.csv")
df1_j = load_df("df1_j_clean.csv")

# Garantir tipos
for _df in [df1_s, df1_t, df1_j]:
    # datas/semana podem não existir em todos
    if "data" in _df.columns:
        _df["data"] = pd.to_datetime(_df["data"], errors="coerce")
    if "semana" in _df.columns:
        try:
            _df["semana"] = _df["semana"].astype("Int64")
        except Exception:
            pass

def filtrar_sidebar(df: pd.DataFrame, *, key_prefix: str, incluir_semana=False, incluir_presenca=False):
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Filtros Interativos")
    st.sidebar.markdown("Use os controles abaixo para ajustar sua análise.")
    st.sidebar.markdown("---")
    view = df.copy()

    if "nome" in view.columns:
        atletas = sorted(view["nome"].dropna().unique().tolist())
        sel = st.sidebar.multiselect(f"{key_prefix}: Atletas", atletas, default=atletas, key=f"{key_prefix}_atletas")
        view = view[view["nome"].isin(sel)]

    if incluir_semana and "semana" in view.columns and view["semana"].notna().any():
        semanas = sorted(view["semana"].dropna().unique().tolist())
        sel_sem = st.sidebar.multiselect(f"{key_prefix}: Semanas", semanas, default=semanas, key=f"{key_prefix}_semanas")
        view = view[view["semana"].isin(sel_sem)]

    if incluir_presenca and "presenca" in view.columns:
        only_present = st.sidebar.checkbox(f"{key_prefix}: Somente presentes", value=False, key=f"{key_prefix}_presenca")
        if only_present:
            view = view[view["presenca"] == 1]

    def _range_slider(col, label):
        if col in view.columns and view[col].notna().any():
            cmin = float(view[col].min(skipna=True))
            cmax = float(view[col].max(skipna=True))
            if np.isfinite(cmin) and np.isfinite(cmax) and cmin != cmax:
                vmin, vmax = st.sidebar.slider(f"{key_prefix}: {label}", min_value=float(cmin), max_value=float(cmax),
                                               value=(float(cmin), float(cmax)), key=f"{key_prefix}_{col}")
                return view[(view[col] >= vmin) & (view[col] <= vmax)]
        return view

    for col, label in [("minutos","Minutos"), ("m_min","m/min"), ("distancia_total","Distância total")]:
        view = _range_slider(col, label)

    return view

aba_s, aba_t, aba_j = st.tabs(["Treino — 4 semanas", "Sessão de treino", "Jogo"])

with aba_s:
    df = filtrar_sidebar(df1_s, key_prefix="S", incluir_semana=True, incluir_presenca=True)
    st.dataframe(df.head(50), use_container_width=True)
    if {"semana","distancia_total"}.issubset(df.columns):
        evol = df.groupby("semana", as_index=False)["distancia_total"].mean().rename(columns={"distancia_total":"média_dist"})
        fig = px.line(evol, x="semana", y="média_dist", markers=True, title="Evolução semanal — Distância média")
        st.plotly_chart(fig, use_container_width=True)

with aba_t:
    df = filtrar_sidebar(df1_t, key_prefix="T")
    st.dataframe(df.head(50), use_container_width=True)
    if {"nome","distancia_total"}.issubset(df.columns):
        rank = df[["nome","distancia_total"]].sort_values("distancia_total", ascending=False)
        fig = px.bar(rank, x="nome", y="distancia_total", title="Ranking — Distância total (sessão)")
        st.plotly_chart(fig, use_container_width=True)

with aba_j:
    df = filtrar_sidebar(df1_j, key_prefix="J")
    st.dataframe(df.head(50), use_container_width=True)
    if {"nome","distancia_total"}.issubset(df.columns):
        rank = df[["nome","distancia_total"]].sort_values("distancia_total", ascending=False)
        fig = px.bar(rank, x="nome", y="distancia_total", title="Distância total por atleta — jogo")
        st.plotly_chart(fig, use_container_width=True)



# =======================
# Nova aba: Comparativo — Volume (Treino vs Jogo) — barras + linha (eixo secundário)
# - Barras: distancia_total (eixo primário)
# - Linha: vol_mecanico (eixo secundário)
# - Ordenado do maior para o menor por contexto
# - Opção "Ver por jogador"
# =======================
from plotly.subplots import make_subplots
import plotly.graph_objects as go

aba_comp, = st.tabs(["Comparativo — Volume (Treino vs Jogo)"])

with aba_comp:
    st.subheader("Distância (barras) + Volume mecânico (linha, eixo secundário) — Treino × Jogo")

    # Garantir colunas de volume mecânico
    for _df in (df1_t, df1_j):
        if "vol_mecanico" not in _df.columns:
            if {"dist_acc","dist_dcc"}.issubset(_df.columns):
                _df["vol_mecanico"] = _df["dist_acc"].fillna(0) + _df["dist_dcc"].fillna(0)
            else:
                _df["vol_mecanico"] = 0.0

    ver_por_jogador = st.checkbox("Ver por jogador", value=True, key="comp_por_jogador_v3")

    if ver_por_jogador and "nome" in df1_t.columns and "nome" in df1_j.columns:
        # Seleção de atletas
        atletas_all = sorted(pd.Index(df1_t["nome"].dropna().unique()).union(pd.Index(df1_j["nome"].dropna().unique())).tolist())
        sel_atletas = st.multiselect("Jogadores", atletas_all, default=atletas_all, key="comp_atletas_v3")

        def prep(df, contexto):
            cols = ["nome", "distancia_total", "vol_mecanico"]
            d = df.loc[df["nome"].isin(sel_atletas), [c for c in cols if c in df.columns]].copy()
            if "distancia_total" not in d.columns:
                d["distancia_total"] = 0.0
            if "vol_mecanico" not in d.columns:
                d["vol_mecanico"] = 0.0
            d = d.groupby("nome", as_index=False).sum(numeric_only=True)
            d["contexto"] = contexto
            # Ordena decrescente por distancia_total
            d = d.sort_values("distancia_total", ascending=False)
            return d

        t = prep(df1_t, "Treino")
        j = prep(df1_j, "Jogo")

        # Subplots com eixo secundário em cada linha
        fig = make_subplots(
            rows=2, cols=1, shared_yaxes=False, shared_xaxes=False,
            subplot_titles=("Treino", "Jogo"),
            specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
        )

        # Treino
        fig.add_trace(
            go.Bar(x=t["nome"], y=t["distancia_total"], name="Distância total (Treino)"),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=t["nome"], y=t["vol_mecanico"], name="Vol. mecânico (Treino)", mode="lines+markers"),
            row=1, col=1, secondary_y=True
        )

        # Jogo
        fig.add_trace(
            go.Bar(x=j["nome"], y=j["distancia_total"], name="Distância total (Jogo)"),
            row=2, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=j["nome"], y=j["vol_mecanico"], name="Vol. mecânico (Jogo)", mode="lines+markers"),
            row=2, col=1, secondary_y=True
        )

        fig.update_yaxes(title_text="Distância (m)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Vol. mecânico (m)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Distância (m)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Vol. mecânico (m)", row=2, col=1, secondary_y=True)

        fig.update_layout(height=700, title_text="Comparativo por jogador — Distância (barras) + Vol. mecânico (linha, eixo secundário)")
        st.plotly_chart(fig, use_container_width=True)

        # Tabela consolidada (ordenada por contexto/distância)
        t_tbl = t.assign(contexto="Treino")[["contexto","nome","distancia_total","vol_mecanico"]]
        j_tbl = j.assign(contexto="Jogo")[["contexto","nome","distancia_total","vol_mecanico"]]
        tbl = pd.concat([t_tbl, j_tbl], ignore_index=True)
        st.dataframe(tbl, use_container_width=True)

    else:
        # Agregado por time com eixo secundário
        def agg_team(df):
            dist_total = df["distancia_total"].sum(skipna=True) if "distancia_total" in df.columns else 0.0
            vol_mec = (df["dist_acc"].fillna(0).sum(skipna=True) + df["dist_dcc"].fillna(0).sum(skipna=True)) if {"dist_acc","dist_dcc"}.issubset(df.columns) else 0.0
            return dist_total, vol_mec

        dist_t, vm_t = agg_team(df1_t)
        dist_j, vm_j = agg_team(df1_j)

        ctx = ["Treino", "Jogo"]
        distancias = [dist_t, dist_j]
        vols = [vm_t, vm_j]

        # Ordena ambos pelo valor de distância (maior->menor)
        order = sorted(range(len(ctx)), key=lambda i: distancias[i], reverse=True)
        ctx = [ctx[i] for i in order]
        distancias = [distancias[i] for i in order]
        vols = [vols[i] for i in order]

        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=ctx, y=distancias, name="Distância total"), secondary_y=False)
        fig.add_trace(go.Scatter(x=ctx, y=vols, name="Vol. mecânico", mode="lines+markers"), secondary_y=True)
        fig.update_yaxes(title_text="Distância (m)", secondary_y=False)
        fig.update_yaxes(title_text="Vol. mecânico (m)", secondary_y=True)
        fig.update_layout(title="Time — Distância (barras) + Vol. mecânico (linha, eixo secundário)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            pd.DataFrame({"contexto": ctx, "distancia_total": distancias, "vol_mecanico": vols}),
            use_container_width=True
        )
