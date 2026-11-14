import streamlit as st
import pandas as pd
import heapq
import graphviz as gv
import itertools

# ==========================================
# BLOQUE 1: CONFIGURACIÓN GENERAL Y DISEÑO WEB
# ==========================================
st.set_page_config(page_title="Optimizador de Rutas Cripto (A* & Kruskal)", layout="wide")

st.title("🚀 Sistema de Optimización de Rutas en Grafos Multiexchange")
st.markdown("**Curso:** Complejidad Algorítmica | **Proyecto TB1** — Algoritmo $A^*$ y Análisis Topológico con Kruskal")

# Barra lateral para parámetros interactivos del usuario
st.sidebar.header("🎛️ Parámetros de Simulación")
monto_inicial = st.sidebar.number_input("Monto inicial (USD)", min_value=1.0, value=1000.0, step=50.0)
exchange_origen = st.sidebar.text_input("Exchange de origen", value="bitget").strip().lower()
moneda_origen = st.sidebar.text_input("Moneda que posee", value="BTC").strip().upper()
moneda_destino = st.sidebar.text_input("Moneda que desea obtener", value="ETH").strip().upper()

ejecutar = st.sidebar.button("Calcular Ruta Óptima", type="primary")


# ==========================================
# BLOQUE 2 & 3 OPTIMIZADO: ETL Y GRAFO DINÁMICO EN MEMORIA
# ==========================================

# 1. Carga estática en caché (Se ejecuta una sola vez y se queda guardada en memoria RAM)
@st.cache_data
def cargar_datos_estaticos():
    df_n = pd.read_csv("nodos.csv")
    df_a = pd.read_csv("aristas.csv")
    df_c = pd.read_csv("dataset_ccxt_completo.csv")
    
    ccxt_lookup = {}
    for _, fila in df_c.iterrows():
        ex_id = str(fila["id"]).strip().lower()
        ccxt_lookup[ex_id] = {
            "trans_ms": float(fila["transferencia_peso_total_min"]) * 60000.0,
            "trade_ms": float(fila["trade_peso_total_ms"])
        }
    
    taker_fees = df_a.groupby("Exchange")["Taker Fee"].mean().to_dict()
    nodos_dict = df_n.set_index("node_id").to_dict(orient="index")
    
    return df_n, df_a, ccxt_lookup, taker_fees, nodos_dict

df_nodos, df_aristas, ccxt_lookup, taker_fees_dict, nodos_dict = cargar_datos_estaticos()

FALLBACK_TRANS_MS = 15.0 * 60000.0
ALPHA_TIEMPO = 0.000005

def obtener_tiempo_trans(ex):
    return ccxt_lookup.get(str(ex).strip().lower(), {}).get("trans_ms", FALLBACK_TRANS_MS)

def obtener_tiempo_trade(ex):
    return ccxt_lookup.get(str(ex).strip().lower(), {}).get("trade_ms", 150.0)

# 2. Función dinámica: Construye el grafo al vuelo usando el monto actual (súper rápido)
@st.cache_data
def construir_grafo_dinamico(monto_actual):
    adj = {nid: [] for nid in nodos_dict}
    edges_kruskal = []

    # 1. Aristas Inter-bolsa (Transferencias) optimizadas con permutations
    for (token, red), grupo in df_aristas.groupby(["Criptomoneda", "Red"]):
        filas = grupo[["nodo_id", "Exchange", "Withdrawal Fee USD"]].values
        for fila_u, fila_v in itertools.permutations(filas, 2):
            u, ex_u, _ = int(fila_u[0]), str(fila_u[1]).lower(), fila_u[2]
            v, _, fee = int(fila_v[0]), str(fila_v[1]).lower(), float(fila_v[2])
            
            t_ms = obtener_tiempo_trans(ex_u)
            costo_comb = fee + (ALPHA_TIEMPO * t_ms)
            det = {"tipo": "Transferencia", "fee": fee, "tiempo_ms": t_ms, "red": red, "token": token}
            adj[u].append((v, fee, costo_comb, det))
            edges_kruskal.append((u, v, fee, t_ms))

    # 2. Aristas Intra-bolsa (Trades) optimizadas con permutations
    for exchange, grupo in df_nodos.groupby("Exchange"):
        ex_nom = str(exchange).lower()
        t_trade_ms = obtener_tiempo_trade(ex_nom)
        nids = grupo["node_id"].tolist()
        pct_taker = taker_fees_dict.get(exchange, 0.001)

        for u, v in itertools.permutations(nids, 2):
            tok_u, red_u = nodos_dict[u]["Criptomoneda"], nodos_dict[u]["Red"]
            tok_v, red_v = nodos_dict[v]["Criptomoneda"], nodos_dict[v]["Red"]

            if (tok_u == tok_v and red_u != red_v) or (tok_u != tok_v):
                fee_swap = monto_actual * pct_taker
                costo_comb = fee_swap + (ALPHA_TIEMPO * t_trade_ms)
                desc_op = f"Trade {tok_u} -> {tok_v}" if tok_u != tok_v else f"Cambio red {tok_u}"
                det = {"tipo": "Trade", "fee": fee_swap, "tiempo_ms": t_trade_ms, "desc": desc_op}
                adj[u].append((v, fee_swap, costo_comb, det))
                edges_kruskal.append((u, v, fee_swap, t_trade_ms))
                        
    return adj, edges_kruskal
# ==========================================
# BLOQUE 4: ANÁLISIS TOPOLÓGICO CON KRUSKAL (DSU)
# ==========================================
def find(parent, i):
    while i != parent[i]:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

def union(parent, u, v):
    ru, rv = find(parent, u), find(parent, v)
    if ru != rv:
        parent[ru] = rv
        return True
    return False

def kruskal_mst(nodos_ids, lista_aristas, criterio="precio"):
    parent = {nid: nid for nid in nodos_ids}
    aristas_ord = sorted(lista_aristas, key=lambda x: x[2] if criterio == "precio" else x[3])

    mst = []
    total_fee, total_tiempo_ms = 0.0, 0.0
    for u, v, fee, t_ms in aristas_ord:
        if union(parent, u, v):
            mst.append((u, v, fee, t_ms))
            total_fee += fee
            total_tiempo_ms += t_ms

    return parent, mst, total_fee, total_tiempo_ms


# ==========================================
# BLOQUE 5: MOTOR DE BÚSQUEDA HEURÍSTICA (A*)
# ==========================================
def heuristica(u, v):
    h = 0.0
    if nodos_dict[u]["Exchange"] != nodos_dict[v]["Exchange"]:
        h += 0.80
    if nodos_dict[u]["Criptomoneda"] != nodos_dict[v]["Criptomoneda"]:
        h += 1.00
    return h

def a_star_routing(adj, origen, destinos_posibles, aristas_bloqueadas=set()):
    pq = [(0.0, 0.0, 0.0, origen, [origen], [])]
    distancias = {}

    while pq:
        f, g_costo, g_tiempo_ms, u, camino, aristas_det = heapq.heappop(pq)
        if u in destinos_posibles:
            return camino, aristas_det, g_costo, g_tiempo_ms

        if u in distancias and distancias[u] <= f:
            continue
        distancias[u] = f

        for v, fee, comb, det in adj.get(u, []):
            if (u, v) in aristas_bloqueadas:
                continue

            nuevo_g_costo = g_costo + det["fee"]
            nuevo_g_tiempo_ms = g_tiempo_ms + det["tiempo_ms"]
            h = min(heuristica(v, d) for d in destinos_posibles)
            f_score = nuevo_g_costo + (ALPHA_TIEMPO * nuevo_g_tiempo_ms) + h

            heapq.heappush(pq, (f_score, nuevo_g_costo, nuevo_g_tiempo_ms, v, camino + [v], aristas_det + [det]))
            
    return None, None, float("inf"), float("inf")


# ==========================================
# BLOQUE 6: EJECUCIÓN Y RENDERIZADO WEB
# ==========================================
if ejecutar:

    adj, edges_kruskal = construir_grafo_dinamico(monto_inicial)

    nodos_inicio = df_nodos[
        (df_nodos["Exchange"].str.lower() == exchange_origen) &
        (df_nodos["Criptomoneda"].str.upper() == moneda_origen)
    ]["node_id"].tolist()

    nodos_meta_externos = df_nodos[
        (df_nodos["Criptomoneda"].str.upper() == moneda_destino) &
        (df_nodos["Exchange"].str.lower() != exchange_origen.lower())
    ]["node_id"].tolist()

    nodos_meta = nodos_meta_externos if nodos_meta_externos else df_nodos[
        df_nodos["Criptomoneda"].str.upper() == moneda_destino
    ]["node_id"].tolist()

    if not nodos_inicio:
        st.error(f"No se encontraron nodos para el origen [{exchange_origen.upper()}:{moneda_origen}]")
    elif not nodos_meta:
        st.error(f"No existen nodos con la moneda destino [{moneda_destino}] en el dataset.")
    else:
        origen_nodo = nodos_inicio[0]

        # 1. Ejecutar análisis de Kruskal
        nodos_totales = list(nodos_dict.keys())
        parent_p, mst_p, fee_p, t_p = kruskal_mst(nodos_totales, edges_kruskal, criterio="precio")
        parent_t, mst_t, fee_t, t_t = kruskal_mst(nodos_totales, edges_kruskal, criterio="tiempo")

        st.subheader("📊 Comparativa Topológica (Kruskal: Precio vs Tiempo)")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("MST por Precio", f"${fee_p:,.2f} USD", f"{t_p / 60000:.2f} min")
        col_m2.metric("MST por Tiempo", f"${fee_t:,.2f} USD", f"{t_t / 60000:.2f} min (Más rápido)")

        conectado = any(find(parent_p, origen_nodo) == find(parent_p, m) for m in nodos_meta)
        if not conectado:
            st.warning("⚠️ Validación Kruskal: No existe conectividad estructural hacia el destino.")
        else:
            # 2. Ejecutar Algoritmo A* (Ruta Principal y Alternativa)
            rutas = []
            camino_1, det_1, costo_1, t_ms_1 = a_star_routing(adj, origen_nodo, set(nodos_meta))
            if camino_1:
                rutas.append({
                    "nombre": "Ruta Óptima",
                    "camino": camino_1,
                    "detalles": det_1,
                    "costo_total": costo_1,
                    "tiempo_total_ms": t_ms_1
                })
                aristas_principales = {(camino_1[i], camino_1[i + 1]) for i in range(len(camino_1) - 1)}
                camino_2, det_2, costo_2, t_ms_2 = a_star_routing(
                    adj, origen_nodo, set(nodos_meta), aristas_bloqueadas=aristas_principales
                )
                if camino_2:
                    rutas.append({
                        "nombre": "Ruta Alternativa",
                        "camino": camino_2,
                        "detalles": det_2,
                        "costo_total": costo_2,
                        "tiempo_total_ms": t_ms_2
                    })

            # Mostrar Desglose de Rutas en Tablas
            for idx_r, r in enumerate(rutas):
                st.markdown(f"### 🛣️ {r['nombre']} (A*)")
                datos_tabla = []
                for idx_p, det in enumerate(r["detalles"], start=1):
                    u = r["camino"][idx_p - 1]
                    v = r["camino"][idx_p]
                    op = f"{nodos_dict[u]['Exchange'].upper()} ➔ {nodos_dict[v]['Exchange'].upper()}"
                    t_min_paso = det["tiempo_ms"] / 60000.0
                    datos_tabla.append({
                        "Paso": idx_p,
                        "Tipo": det["tipo"],
                        "Operación": op,
                        "Fee ($)": round(det["fee"], 4),
                        "Tiempo (min)": round(t_min_paso, 4)
                    })
                st.dataframe(pd.DataFrame(datos_tabla), width='stretch')
                t_total_min = r["tiempo_total_ms"] / 60000.0
                st.info(f"**TOTAL:** Costo: ${r['costo_total']:.2f} USD | Tiempo: {t_total_min:.2f} min")

            # 3. Visualización con Graphviz (Ruta Principal + Alternativa unidas)
            st.subheader("🕸️ Topología y Grafo de la Ruta")
            dot = gv.Digraph(
                comment="Ruta_Optima_A_Star",
                graph_attr={"rankdir": "LR", "bgcolor": "#FFFFFF", "splines": "true", "nodesep": "0.6", "ranksep": "1.0"},
                node_attr={"shape": "rect", "style": "rounded,filled", "fontname": "Helvetica", "fontsize": "9"},
                edge_attr={"fontname": "Helvetica", "fontsize": "8", "arrowsize": "0.7"}
            )

            todos_nodos = set()
            for r in rutas:
                todos_nodos.update(r["camino"])

            df_ruta = df_nodos[df_nodos["node_id"].isin(todos_nodos)]
            for ex in df_ruta["Exchange"].unique():
                with dot.subgraph(name=f"cluster_{ex}") as c:
                    c.attr(label=f"EXCHANGE: {ex.upper()}", style="dashed", color="#546E7A", bgcolor="#F8F9FA", fontcolor="#263238", fontsize="10")
                    for _, fila in df_ruta[df_ruta["Exchange"] == ex].iterrows():
                        nid = str(fila["node_id"])
                        label_nodo = f"{fila['Criptomoneda']}\n({fila['Red']})"
                        color = "#FFF2CC" if fila["Criptomoneda"] == moneda_origen else "#C8E6C9"
                        c.node(nid, label=label_nodo, fillcolor=color, color="#455A64")

            # Dibujar Ruta Óptima [0] (Líneas sólidas verdes/naranjas)
            if len(rutas) > 0:
                r0 = rutas[0]
                for i in range(len(r0["camino"]) - 1):
                    u, v = str(r0["camino"][i]), str(r0["camino"][i + 1])
                    det = r0["detalles"][i]
                    t_min = det["tiempo_ms"] / 60000.0
                    formato_t = f"{t_min:.2f} min" if t_min >= 0.01 else f"{t_min:.4f} min"

                    if det["tipo"] == "Transferencia":
                        etiqueta = f"Transferencia: {det['red']}\n${det['fee']:.2f} | {formato_t}"
                        color_l = "#E65100"
                    else:
                        etiqueta = f"Trade\n${det['fee']:.2f} | {formato_t}"
                        color_l = "#2E7D32"
                    dot.edge(u, v, label=etiqueta, color=color_l, penwidth="2.5", fontcolor="#263238")

            # Dibujar Ruta Alternativa [1] (Líneas punteadas azules)
            if len(rutas) > 1:
                r1 = rutas[1]
                for i in range(len(r1["camino"]) - 1):
                    u, v = str(r1["camino"][i]), str(r1["camino"][i + 1])
                    det = r1["detalles"][i]
                    t_min = det["tiempo_ms"] / 60000.0
                    formato_t = f"{t_min:.2f} min" if t_min >= 0.01 else f"{t_min:.4f} min"

                    if det["tipo"] == "Transferencia":
                        etiqueta = f"[Alt] Transferencia: {det['red']}\n${det['fee']:.2f} | {formato_t}"
                    else:
                        etiqueta = f"[Alt] Trade\n${det['fee']:.2f} | {formato_t}"
                    dot.edge(u, v, label=etiqueta, color="#1565C0", style="dashed", penwidth="1.8", fontcolor="#1565C0")

            st.graphviz_chart(dot, width='stretch')