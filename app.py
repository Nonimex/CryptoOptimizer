import heapq
import itertools
import os
import re
import sys
import time
from collections import deque
from math import inf
import graphviz as gv

import networkx as nx
import pandas as pd
import streamlit as st

# =============================================================================
# CONFIGURACIÓN (Optimizado para Streamlit y Parquet)
# =============================================================================
PARQUET_NODOS = "nodos.parquet"
PARQUET_ARISTAS = "aristas.parquet"
PARQUET_CCXT = "dataset_ccxt_completo.parquet"

# Fallbacks automáticos a CSV si los archivos parquet no están presentes
CSV_NODOS = PARQUET_NODOS if os.path.exists(PARQUET_NODOS) else "nodos.csv"
CSV_ARISTAS = PARQUET_ARISTAS if os.path.exists(PARQUET_ARISTAS) else "aristas.csv"
CSV_CCXT = PARQUET_CCXT if os.path.exists(PARQUET_CCXT) else "dataset_ccxt_completo.csv"

ALPHA_TIEMPO = 0.000005                 # USD por ms (pasa el tiempo a dólares en f(n))
FALLBACK_TRANS_MS = 15.0 * 60000.0   # exchange que no está en el CSV CCXT: 15 min
FALLBACK_TRADE_MS = 150.0            # exchange que no está en el CSV CCXT: 150 ms
TAKER_FEE_DEFECTO = 0.001            # exchange sin taker fee en aristas: 0.1 %
H_EXCHANGE = 0.10                    # heurística: otro exchange
H_TOKEN = 0.00                       # heurística: otra moneda
VIGENCIA_DATOS_SEG = 28800             # pasado este tiempo se avisa que los datos son viejos


# =============================================================================
# 1) FORMATO DE TIEMPOS (los datos van en ms; se muestran en MINUTOS)
# =============================================================================
def formatear_min(ms):
    """Duracion de una ruta en minutos."""
    if ms <= 0:
        return "0.00 min"
    minutos = ms / 60_000
    return "<0.01 min" if minutos < 0.01 else f"{minutos:,.2f} min"


def formatear_antiguedad(seg):
    """Antiguedad de los datos en la unidad mas legible (min, h o dias)."""
    if seg < 7_200:
        return f"{seg / 60:,.0f} min"
    if seg < 172_800:
        return f"{seg / 3_600:,.1f} h"
    return f"{seg / 86_400:,.1f} dias"


# =============================================================================
# 2) GRAFO (Cacheado para Streamlit)
# =============================================================================
@st.cache_data(show_spinner=False)
def cargar_tiempos_ccxt(csv=CSV_CCXT):
    """{id_exchange: {"trans_ms", "trade_ms"}} del dataset CCXT ({} si no esta o no se lee)."""
    if not os.path.exists(csv):
        return {}
    try:
        df = pd.read_parquet(csv) if csv.endswith('.parquet') else pd.read_csv(csv)
        return {
            str(f["id"]).strip().lower(): {
                "trans_ms": float(f["transferencia_peso_total_min"]) * 60000.0,
                "trade_ms": float(f["trade_peso_total_ms"])
            }
            for _, f in df.iterrows()
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=True)
def cargar_grafo(nodos_csv=CSV_NODOS, aristas_csv=CSV_ARISTAS, ccxt_csv=CSV_CCXT):
    nodos = pd.read_parquet(nodos_csv) if nodos_csv.endswith('.parquet') else pd.read_csv(nodos_csv)
    ar = pd.read_parquet(aristas_csv) if aristas_csv.endswith('.parquet') else pd.read_csv(aristas_csv)
    lookup = cargar_tiempos_ccxt(ccxt_csv)

    def t_trans(ex):
        return lookup.get(str(ex).strip().lower(), {}).get("trans_ms", FALLBACK_TRANS_MS)

    def t_trade(ex):
        return lookup.get(str(ex).strip().lower(), {}).get("trade_ms", FALLBACK_TRADE_MS)

    G = nx.DiGraph(archivo=aristas_csv, tiempos_csv=ccxt_csv if lookup else None, orden=[])
    orden = G.graph["orden"]
    precios = ar.groupby("nodo_id")["Precio Last"].first()
    for r in nodos.itertuples(index=False):
        nid = int(r.node_id)
        G.add_node(nid, exchange=r.Exchange, cripto=r.Criptomoneda, red=r.Red,
                   precio=float(precios.loc[nid]) if nid in precios.index else 1.0)

    def agregar(u, v, **datos):
        """Agrega la arista; si hay duplicados, conserva la de menor costo fijo."""
        if u == v or u not in G or v not in G:
            return
        if G.has_edge(u, v):
            if G[u][v]["fijo_usd"] <= datos["fijo_usd"]:
                return
        else:
            orden.append((u, v))
        G.add_edge(u, v, **datos)

    # A. TRANSFERENCIAS entre exchanges (misma moneda y red)
    for (token, red), grupo in ar.groupby(["Criptomoneda", "Red"]):
        filas = grupo[["nodo_id", "Exchange", "Withdrawal Fee USD"]].values
        for i in range(len(filas)):
            for j in range(len(filas)):
                if i != j:
                    u, ex_u = int(filas[i][0]), str(filas[i][1]).lower()
                    v, fee = int(filas[j][0]), float(filas[j][2])
                    if fee != fee:  # NaN -> 0
                        fee = 0.0
                    agregar(u, v, tipo="inter", fee_pct=0.0, fijo_usd=fee,
                            tiempo_ms=t_trans(ex_u), red=red, token=token)

    # B. TRADES dentro de cada exchange (cambio de moneda o de red)
    taker = ar.groupby("Exchange")["Taker Fee"].mean().to_dict()
    for exchange, grupo in nodos.groupby("Exchange"):
        t_ms = t_trade(str(exchange).lower())
        pct = taker.get(exchange, TAKER_FEE_DEFECTO)
        if not pct or pct <= 0.0 or pct != pct:  # NaN, 0.0 o negativo -> valor por defecto
            pct = TAKER_FEE_DEFECTO
        ids = grupo["node_id"].tolist()
        for u in ids:
            for v in ids:
                if u != v and (G.nodes[u]["cripto"] != G.nodes[v]["cripto"]
                                or G.nodes[u]["red"] != G.nodes[v]["red"]):
                    agregar(u, v, tipo="intra", fee_pct=pct, fijo_usd=0.0, tiempo_ms=t_ms)

    exchanges = {str(e).strip().lower() for e in nodos["Exchange"]}
    G.graph["sin_tiempo"] = sorted(e for e in exchanges if e not in lookup)
    G.graph["n_exchanges"] = len(exchanges)
    return G


def buscar_nodos(G, exchange=None, cripto=None):
    """Nodos que cumplen los filtros (exchange y/o moneda)."""
    return [n for n, d in G.nodes(data=True)
            if (not exchange or d["exchange"].lower() == exchange.lower())
            and (not cripto or d["cripto"].upper() == cripto.upper())]


def seleccionar_destinos(G, ex_o, cr_d):
    """Nodos destino: la moneda pedida en OTRO exchange (si no hay, en cualquiera)."""
    todos = buscar_nodos(G, None, cr_d)
    externos = [n for n in todos if G.nodes[n]["exchange"].lower() != ex_o.lower()]
    return externos if externos else todos


# =============================================================================
# 3) ALGORITMOS
# =============================================================================
def medir(func, *args, **kwargs):
    """Ejecuta func y devuelve (resultado, tiempo_en_ms)."""
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    return res, (time.perf_counter() - t0) * 1000


def costo_arista(d, monto):
    return monto * d["fee_pct"] + d["fijo_usd"]


def peso_arista(d, monto):
    return costo_arista(d, monto) + ALPHA_TIEMPO * d["tiempo_ms"]


def _reconstruir(padre, fin):
    ruta = []
    while fin is not None:
        ruta.append(fin)
        fin = padre[fin]
    return ruta[::-1]


def _resultado(G, ruta, monto, expandidos, peso):
    costo = sum(costo_arista(G[u][v], monto) for u, v in zip(ruta, ruta[1:]))
    tiempo = sum(G[u][v]["tiempo_ms"] for u, v in zip(ruta, ruta[1:]))
    return {"ruta": ruta, "costo": costo, "tiempo_ms": tiempo, "peso": peso,
            "expandidos": expandidos}


# ---------------- Kruskal + Union-Find ----------------
def _find(parent, i):
    while i != parent[i]:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _union(parent, u, v):
    ru, rv = _find(parent, u), _find(parent, v)
    if ru != rv:
        parent[ru] = rv
        return True
    return False


def kruskal_mst(nodos, aristas, criterio="precio"):
    """aristas = [(u, v, fee, tiempo_ms)]. Devuelve (parent, mst, total_fee, total_ms)."""
    parent = {n: n for n in nodos}
    ordenadas = sorted(aristas, key=lambda x: x[2] if criterio == "precio" else x[3])
    mst, total_fee, total_ms = [], 0.0, 0.0
    for u, v, fee, t_ms in ordenadas:
        if _union(parent, u, v):
            mst.append((u, v, fee, t_ms))
            total_fee += fee
            total_ms += t_ms
    return parent, mst, total_fee, total_ms


def aristas_kruskal(G, monto):
    return [(u, v, costo_arista(G[u][v], monto), G[u][v]["tiempo_ms"])
            for u, v in G.graph["orden"] if G.has_edge(u, v)]


def hay_conexion(parent, origenes, destinos):
    return any(_find(parent, o) == _find(parent, d) for o in origenes for d in destinos)


# ---------------- A* ----------------
def astar_grupo(G, origenes, destinos, monto, bloqueadas=frozenset()):
    origen = origenes[0]
    destinos = set(destinos)
    pares_destino = {(G.nodes[d]["exchange"], G.nodes[d]["cripto"]) for d in destinos}
    cache_h = {}

    def h(v):
        clave = (G.nodes[v]["exchange"], G.nodes[v]["cripto"])
        if clave not in cache_h:
            mejor = inf
            for ex_d, tok_d in pares_destino:
                x = 0.0
                if clave[0] != ex_d:
                    x += H_EXCHANGE
                if clave[1] != tok_d:
                    x += H_TOKEN
                mejor = min(mejor, x)
            cache_h[clave] = mejor
        return cache_h[clave]

    pq = [(0.0, 0.0, 0.0, origen, [origen])]
    distancias, expandidos = {}, 0
    while pq:
        f, g_costo, g_tiempo, u, camino = heapq.heappop(pq)
        if u in destinos:
            return _resultado(G, camino, monto, expandidos, g_costo + ALPHA_TIEMPO * g_tiempo)
        if u in distancias and distancias[u] <= f:
            continue
        distancias[u] = f
        expandidos += 1
        for v, d in G[u].items():
            if (u, v) in bloqueadas:
                continue
            nuevo_g_costo = g_costo + costo_arista(d, monto)
            nuevo_g_tiempo = g_tiempo + d["tiempo_ms"]
            f_score = nuevo_g_costo + (ALPHA_TIEMPO * nuevo_g_tiempo) + h(v)
            heapq.heappush(pq, (f_score, nuevo_g_costo, nuevo_g_tiempo, v, camino + [v]))
    return None


def ruta_directa(G, origenes, destinos):
    """Ruta 'sin A*': la de MENOS pasos (BFS), sin mirar los costos."""
    destinos = set(destinos)
    padre = {o: None for o in origenes}
    cola = deque(origenes)
    while cola:
        u = cola.popleft()
        if u in destinos:
            return _reconstruir(padre, u)
        for v in G[u]:
            if v not in padre:
                padre[v] = u
                cola.append(v)
    return None


def buscar_ruta(G, origenes, destinos, monto):
    aristas = aristas_kruskal(G, monto)
    nodos = list(G.nodes)
    parent, _, fee_p, t_p = kruskal_mst(nodos, aristas, "precio")
    _, _, fee_t, t_t = kruskal_mst(nodos, aristas, "tiempo")
    r = {"conectado": hay_conexion(parent, origenes[:1], destinos), "ruta": None,
         "alternativa": None, "topologia": (fee_p, t_p, fee_t, t_t)}
    if r["conectado"]:
        r["ruta"] = astar_grupo(G, origenes, destinos, monto)
        if r["ruta"] is not None:
            ruta = r["ruta"]["ruta"]
            bloqueadas = {(ruta[i], ruta[i + 1]) for i in range(len(ruta) - 1)}
            r["alternativa"] = astar_grupo(G, origenes, destinos, monto, bloqueadas)
    return r

# =============================================================================
# 4) CONSULTA: validaciones (camino feliz e infelices)
# =============================================================================
def obtener_ts_cotizacion(archivo_datos, archivo_fecha="fecha_extraccion.txt"):
    """Momento de la cotizacion: fecha_extraccion.txt (epoch o ISO) o fecha del archivo."""
    if os.path.exists(archivo_fecha):
        try:
            with open(archivo_fecha, encoding="utf-8") as f:
                txt = f.read().strip()
            return float(txt)
        except ValueError:
            try:
                from datetime import datetime
                return datetime.fromisoformat(txt).timestamp()
            except ValueError:
                pass
    return os.path.getmtime(archivo_datos)


ESTADOS = {
    "OK": "Ruta encontrada",
    "ENTRADA_INVALIDA": "Monto o campos invalidos",
    "ORIGEN_INEXISTENTE": "El exchange de origen no existe en el dataset",
    "MONEDA_ORIGEN_INEXISTENTE": "El exchange no tiene la moneda de origen",
    "DESTINO_INEXISTENTE": "La moneda de destino no existe en el dataset",
    "YA_EN_DESTINO": "Ya posee esa moneda en el exchange de destino",
    "SIN_CONEXION": "No hay conexion entre origen y destino (Kruskal)",
    "SIN_RUTA": "Hay conexion pero no existe ruta valida",
}


def evaluar_consulta(G, p):
    """
    Camino completo de UNA consulta, sin pedir datos ni imprimir: normaliza, valida,
    verifica conectividad (Kruskal) y busca la ruta (A*). Nunca lanza excepciones por
    datos de entrada: siempre devuelve un 'estado' (ver ESTADOS).
    """
    res = {"estado": None, "origenes": [], "destinos": [], "r": None, "t_ms": 0.0,
           "vencidos": False, "edad_seg": 0.0, "p": p}
    try:
        monto = float(p.get("monto"))
    except (TypeError, ValueError):
        monto = None
    ex_o = str(p.get("ex_o") or "").strip().lower()
    cr_o = str(p.get("cr_o") or "").strip().upper()
    cr_d = str(p.get("cr_d") or "").strip().upper()
    if monto is None or not (0 < monto < inf) or not ex_o or not cr_o or not cr_d:
        res["estado"] = "ENTRADA_INVALIDA"
        return res
    res["p"] = p = {"monto": monto, "ex_o": ex_o, "cr_o": cr_o, "cr_d": cr_d}
    if ex_o not in {d["exchange"].lower() for _, d in G.nodes(data=True)}:
        res["estado"] = "ORIGEN_INEXISTENTE"
        return res
    inicio = buscar_nodos(G, ex_o, cr_o)
    if not inicio:
        res["estado"] = "MONEDA_ORIGEN_INEXISTENTE"
        return res
    destinos = seleccionar_destinos(G, ex_o, cr_d)
    res["origenes"], res["destinos"] = inicio[:1], destinos
    if not destinos:
        res["estado"] = "DESTINO_INEXISTENTE"
        return res
    if inicio[0] in destinos:
        res["estado"] = "YA_EN_DESTINO"
        return res
    try:
        archivo_ref = G.graph.get("archivo", CSV_NODOS)
        if os.path.exists(archivo_ref):
            edad = time.time() - obtener_ts_cotizacion(archivo_ref)
            res["vencidos"], res["edad_seg"] = edad > VIGENCIA_DATOS_SEG, edad
    except OSError:
        pass
    r, res["t_ms"] = medir(buscar_ruta, G, res["origenes"], destinos, monto)
    res["r"] = r
    if not r["conectado"]:
        res["estado"] = "SIN_CONEXION"
    elif r["ruta"] is None:
        res["estado"] = "SIN_RUTA"
    else:
        res["estado"] = "OK"
    return res


# =============================================================================
# 5) FUNCIONES DE AUXILIARES Y FORMATO NUMÉRICO
# =============================================================================
def _cant(x):
    if x >= 1000:
        return f"{x:,.2f}"
    return f"{x:.4f}" if x >= 1 else f"{x:.6f}"


def _usd(x):
    return f"${x:,.2f} USD" if (x >= 0.01 or x == 0) else f"${x:.4f} USD"


def _pasos_ruta(G, res, monto):
    """Convierte la ruta en pasos (con cantidades)."""
    ruta = res["ruta"]
    pasos, restante = [], monto
    for u, v in zip(ruta, ruta[1:]):
        d = G[u][v]
        nu, nv = G.nodes[u], G.nodes[v]
        costo = costo_arista(d, monto)
        cant = restante / (nu.get("precio") or 1.0)
        ex_u, ex_v = nu["exchange"].capitalize(), nv["exchange"].capitalize()
        if d["tipo"] == "intra":
            tipo, medio, red = "Trade", "Plataforma", None
            if nu["cripto"] == nv["cripto"]:
                detalle = (f"{ex_u}: Cambiar red de {_cant(cant)} {nu['cripto']} "
                           f"({nu['red']} -> {nv['red']})")
            else:
                detalle = f"{ex_u}: Intercambiar {_cant(cant)} {nu['cripto']} -> {nv['cripto']}"
        else:
            tipo, medio, red = "Transferencia", "Red de transferencia", str(d["red"])
            detalle = f"Transferir de {ex_u} -> {ex_v} ({_cant(cant)} {nu['cripto']})"
        pasos.append({"tipo": tipo, "detalle": detalle, "medio": medio, "red": red,
                      "costo": costo, "tiempo_ms": d["tiempo_ms"]})
        restante -= costo
    return pasos


def _resumen_ruta(G, res, monto):
    pasos = _pasos_ruta(G, res, monto)
    trading = sum(x["costo"] for x in pasos if x["tipo"] == "Trade")
    retiro = sum(x["costo"] for x in pasos if x["tipo"] == "Transferencia")
    total = trading + retiro
    fin = G.nodes[res["ruta"][-1]]
    final_txt = f"{_cant((monto - total) / (fin.get('precio') or 1.0))} {fin['cripto']}"
    return {"pasos": pasos, "trading": trading, "retiro": retiro, "total": total,
            "redes": list(dict.fromkeys(x["red"] for x in pasos if x["red"])),
            "tiempo_ms": sum(x["tiempo_ms"] for x in pasos), "final_txt": final_txt}


# =============================================================================
# 6) GENERADOR DE GRAFO VISUAL (GRAPHVIZ)
# =============================================================================
def generar_grafo_visual(G, r, monto, p):
    """Genera y devuelve el objeto dot de Graphviz combinando la ruta óptima y alternativa."""
    try:
        import graphviz as gv
    except ImportError:
        return None

    dot = gv.Digraph(
        comment="Ruta_Optima_A_Star", format="png",
        graph_attr={"rankdir": "LR", "bgcolor": "#FFFFFF", "splines": "true", "nodesep": "0.6", "ranksep": "1.0"},
        node_attr={"shape": "rect", "style": "rounded,filled", "fontname": "Helvetica", "fontsize": "9"},
        edge_attr={"fontname": "Helvetica", "fontsize": "8", "arrowsize": "0.7"}
    )

    rutas_a_dibujar = []
    if r.get("ruta") and r["ruta"].get("ruta"):
        rutas_a_dibujar.append(("óptima", r["ruta"]["ruta"]))
    if r.get("alternativa") and r["alternativa"].get("ruta"):
        rutas_a_dibujar.append(("alternativa", r["alternativa"]["ruta"]))

    todos_nodos = set()
    for _, camino in rutas_a_dibujar:
        todos_nodos.update(camino)

    if not todos_nodos:
        return None

    df_nodos = pd.read_parquet(CSV_NODOS) if CSV_NODOS.endswith('.parquet') else pd.read_csv(CSV_NODOS)
    df_ruta = df_nodos[df_nodos["node_id"].isin(todos_nodos)]

    moneda_origen_actual = p["cr_o"].upper()

    for ex in df_ruta["Exchange"].unique():
        with dot.subgraph(name=f"cluster_{ex}") as c:
            c.attr(label=f"EXCHANGE: {ex.upper()}", style="dashed", color="#546E7A", bgcolor="#F8F9FA", fontcolor="#263238", fontsize="10")
            for _, fila in df_ruta[df_ruta["Exchange"] == ex].iterrows():
                nid = str(fila["node_id"])
                label_nodo = f"{fila['Criptomoneda']}\n({fila['Red']})"
                color = "#FFF2CC" if str(fila['Criptomoneda']).upper() == moneda_origen_actual else "#C8E6C9"
                c.node(nid, label=label_nodo, fillcolor=color, color="#455A64")

    # Dibujar Ruta Óptima
    if r.get("ruta") and r["ruta"].get("ruta"):
        camino_0 = r["ruta"]["ruta"]
        for i in range(len(camino_0) - 1):
            u, v = camino_0[i], camino_0[i + 1]
            d = G[u][v]
            fee = costo_arista(d, monto)
            t_min = d["tiempo_ms"] / 60000.0
            formato_t = f"{t_min:.2f} min" if t_min >= 0.01 else f"{t_min:.4f} min"

            if d["tipo"] == "inter":
                etiqueta = f"Transferencia: {d.get('red', 'Principal')}\n${fee:.2f} | {formato_t}"
                color_l = "#E65100"
            else:
                etiqueta = f"Trade\n${fee:.2f} | {formato_t}"
                color_l = "#2E7D32"

            dot.edge(str(u), str(v), label=etiqueta, color=color_l, penwidth="2.5", fontcolor="#263238")

    # Dibujar Ruta Alternativa
    if r.get("alternativa") and r["alternativa"].get("ruta"):
        camino_1 = r["alternativa"]["ruta"]
        for i in range(len(camino_1) - 1):
            u, v = camino_1[i], camino_1[i + 1]
            d = G[u][v]
            fee = costo_arista(d, monto)
            t_min = d["tiempo_ms"] / 60000.0
            formato_t = f"{t_min:.2f} min" if t_min >= 0.01 else f"{t_min:.4f} min"

            if d["tipo"] == "inter":
                etiqueta = f"[Alt] Transferencia: {d.get('red', 'Principal')}\n${fee:.2f} | {formato_t}"
            else:
                etiqueta = f"[Alt] Trade\n${fee:.2f} | {formato_t}"

            dot.edge(str(u), str(v), label=etiqueta, color="#1565C0", style="dashed", penwidth="1.8", fontcolor="#1565C0")

    return dot


# =============================================================================
# 7) INTERFAZ GRÁFICA WEB DE STREAMLIT (Capa de Presentación - Estilo Dashboard Financiero)
# =============================================================================
st.set_page_config(
    page_title="Terminal de Enrutamiento Multiexchange | Optimización Cripto",
    layout="wide"
)

st.title("Terminal de Enrutamiento Multiexchange")
st.markdown("Sistema de optimización informada mediante **A*** y validación estructural por **Kruskal**.")

# Carga de grafo cacheada
with st.spinner("Cargando grafo y datasets optimizados..."):
    G = cargar_grafo()

if G is None:
    st.error("No se pudo cargar el grafo. Verifique que los archivos de datos estén presentes.")
    st.stop()

# Configuración en la barra lateral (Estilo panel de control)
st.sidebar.header("Parámetros de Consulta")
monto_input = st.sidebar.number_input("Monto Inicial (USD)", min_value=1.0, value=1000.0, step=100.0)

exchanges_disponibles = sorted(list({d["exchange"] for _, d in G.nodes(data=True)}))
ex_o_input = st.sidebar.selectbox("Exchange de Origen", exchanges_disponibles)

monedas_origen = sorted(list({d["cripto"] for _, d in G.nodes(data=True) if d["exchange"].lower() == ex_o_input.lower()}))
cr_o_input = st.sidebar.selectbox("Moneda de Origen", monedas_origen)

todas_monedas = sorted(list({d["cripto"] for _, d in G.nodes(data=True)}))
cr_d_input = st.sidebar.selectbox("Moneda de Destino", todas_monedas)

ejecutar = st.sidebar.button("Ejecutar Búsqueda", type="primary")

if ejecutar:
    params = {
        "monto": monto_input,
        "ex_o": ex_o_input,
        "cr_o": cr_o_input,
        "cr_d": cr_d_input
    }

    with st.spinner("Procesando enrutamiento y análisis topológico..."):
        ev = evaluar_consulta(G, params)

    estado = ev["estado"]
    p = ev["p"]

    if estado == "ENTRADA_INVALIDA":
        st.error("Monto o datos inválidos: ingrese un monto mayor que 0.")
    elif estado == "ORIGEN_INEXISTENTE":
        st.error(f"El exchange '{p['ex_o']}' no existe en el dataset.")
    elif estado == "MONEDA_ORIGEN_INEXISTENTE":
        st.error(f"El exchange '{p['ex_o']}' no tiene la moneda '{p['cr_o']}'.")
    elif estado == "DESTINO_INEXISTENTE":
        st.error(f"La moneda '{p['cr_d']}' no existe en el dataset.")
    elif estado == "YA_EN_DESTINO":
        st.warning(f"Posesión actual detectada: ya posee {p['cr_d']} en {p['ex_o']}.")
    elif estado == "SIN_CONEXION":
        st.error("Validación Kruskal: No existe conectividad estructural hacia el destino.")
    elif estado == "SIN_RUTA":
        st.error("Validación de red: Existe conexión estructural pero no se encontró ruta dirigida válida.")
    elif estado == "OK":
        if ev["vencidos"]:
            st.warning(f"Aviso de mercado: Los datos presentan {formatear_antiguedad(ev['edad_seg'])} de antigüedad; los costos pueden haber variado.")

        # Cabecera de ejecución
        st.markdown("---")
        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.markdown("### Panel de Enrutamiento Óptimo (A*)")
            st.caption(f"Latencia de procesamiento: {ev['t_ms']:.2f} ms")
        with col_h2:
            st.metric("Estado de Red", "OPTIMIZADO", "100% Conectado")

        r = ev["r"]
        fee_p, t_p, fee_t, t_t = r["topologia"]

        # Sección 1: Análisis Topológico (Tarjetas estilo Grid Financiero)
        st.markdown("#### Análisis Topológico Global (Kruskal: Precio vs Tiempo)")
        t_col1, t_col2, t_col3 = st.columns(3)
        
        with t_col1:
            with st.container(border=True):
                st.markdown("**MST por Precio (Fee Mínimo)**")
                st.metric(label="Tarifa Global", value=f"${fee_p:,.2f} USD")
                st.caption(f"Tiempo acumulado: {formatear_min(t_p)}")
                
        with t_col2:
            with st.container(border=True):
                st.markdown("**MST por Tiempo (Latencia)**")
                st.metric(label="Tarifa Global", value=f"${fee_t:,.2f} USD")
                st.caption(f"Tiempo acumulado: {formatear_min(t_t)}")
                
        with t_col3:
            with st.container(border=True):
                st.markdown("**Delta de Eficiencia**")
                ahorro_t = t_p - t_t
                st.metric(label="Ahorro Temporal", value=f"{formatear_min(abs(ahorro_t))}")
                st.caption("Trade-off Costo vs Velocidad")

        st.markdown("---")

        res_o = _resumen_ruta(G, r["ruta"], p["monto"])
        res_d = _resumen_ruta(G, _resultado(G, ruta_directa(G, ev["origenes"], ev["destinos"]), p["monto"], 0, 0.0), p["monto"])

        # Sección 2: Pasos de la Ruta Óptima
        st.markdown("#### Ejecución de Órdenes y Pasos (A*)")
        with st.container(border=True):
            df_pasos = pd.DataFrame(res_o["pasos"])
            st.dataframe(df_pasos[["tipo", "detalle", "medio", "costo", "tiempo_ms"]], width='stretch')

        res_a = None
        if r.get("alternativa") is not None:
            res_a = _resumen_ruta(G, r["alternativa"], p["monto"])
            with st.expander("Ver Ruta Alternativa de Respaldo (A*)"):
                df_pasos_alt = pd.DataFrame(res_a["pasos"])
                st.dataframe(df_pasos_alt[["tipo", "detalle", "medio", "costo", "tiempo_ms"]], width='stretch')

        # Sección 3: Resumen y Comparativa Financiera
        st.markdown("#### Resumen y Comparativa Financiera")
        comparativa_data = {
            "Métrica": [
                "Pasos / Saltos",
                "Red de Transferencia",
                "Comisiones de Trading",
                "Comisiones de Retiro (Fee)",
                "COSTO TOTAL EN FEES",
                "TIEMPO TOTAL ESTIMADO",
                "MONTO FINAL RECIBIDO"
            ],
            "Ruta Directa (Sin A*)": [
                f"{len(res_d['pasos'])} pasos",
                ", ".join(res_d["redes"]) if res_d["redes"] else "Sin transferencia",
                _usd(res_d["trading"]),
                _usd(res_d["retiro"]),
                _usd(res_d["total"]),
                formatear_min(res_d["tiempo_ms"]),
                res_d["final_txt"]
            ],
            "Ruta Óptima A*": [
                f"{len(res_o['pasos'])} pasos",
                ", ".join(res_o["redes"]) if res_o["redes"] else "Sin transferencia",
                _usd(res_o["trading"]),
                _usd(res_o["retiro"]),
                _usd(res_o["total"]),
                formatear_min(res_o["tiempo_ms"]),
                res_o["final_txt"]
            ]
        }

        if res_a is not None:
            comparativa_data["Ruta Alternativa A*"] = [
                f"{len(res_a['pasos'])} pasos",
                ", ".join(res_a["redes"]) if res_a["redes"] else "Sin transferencia",
                _usd(res_a["trading"]),
                _usd(res_a["retiro"]),
                _usd(res_a["total"]),
                formatear_min(res_a["tiempo_ms"]),
                res_a["final_txt"]
            ]

        with st.container(border=True):
            df_comp = pd.DataFrame(comparativa_data)
            st.table(df_comp)

        # Resultado financiero destacado
        ahorro = res_d["total"] - res_o["total"]
        if ahorro > 1e-9 and res_d["total"] > 0:
            porcentaje_ahorro = (ahorro / res_d['total']) * 100
            st.success(f"**Resultado Financiero:** Ahorro total de **{_usd(ahorro)}** ({porcentaje_ahorro:.1f}% de reducción en comisiones).")
        else:
            st.warning("**Resultado Financiero:** La ruta directa ya es la opción óptima (sin ahorro adicional en comisiones).")

        if res_a is None:
            st.info("Información: No existe una ruta alternativa disponible para esta consulta.")

        st.markdown("---")

        # Sección 4: Visualización Gráfica
        st.markdown("#### Visualización de Topología y Red")
        with st.container(border=True):
            dot_obj = generar_grafo_visual(G, r, p["monto"], p)
            if dot_obj:
                st.graphviz_chart(dot_obj.source)
            else:
                st.info("No se pudo generar el diagrama visual del grafo.")
else:
    st.info("Configure los parámetros en la barra lateral y haga clic en **Ejecutar Búsqueda** para iniciar el análisis.")