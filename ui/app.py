import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import networkx as nx
from pyvis.network import Network
import plotly.express as px
import pandas as pd
import os

API_BASE = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="Graph Transformer", layout="wide")

# Custom CSS for specific overrides (Pyvis iframe height, padding, etc)
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(to right, #ffffff, #a3a3a3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #a3a3a3;
        font-size: 1.25rem;
        margin-bottom: 2rem;
    }
    .badge {
        display: inline-block;
        background: #111;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #a3a3a3;
        margin-right: 1rem;
        margin-bottom: 1rem;
    }
    hr {
        border-color: rgba(255,255,255,0.1);
        margin: 3rem 0;
    }
</style>
""", unsafe_allow_html=True)


def generate_graph(num_nodes, seed=None):
    """Return (graph_data, error). Never silently returns None on failure."""
    params = {"num_nodes": num_nodes}
    if seed is not None:
        params["seed"] = seed
    try:
        res = requests.get(f"{API_BASE}/generate", params=params, timeout=30)
    except requests.exceptions.ConnectionError:
        return None, (
            "Cannot reach the model API at "
            f"{API_BASE}. Start it with "
            "`.venv\\Scripts\\python -m uvicorn src.api:app --port 8000`."
        )
    except requests.exceptions.Timeout:
        return None, "The model API timed out after 30s."
    if res.status_code != 200:
        try:
            detail = res.json().get("detail", res.text)
        except ValueError:
            detail = res.text
        return None, f"API returned HTTP {res.status_code}: {detail}"
    return res.json(), None


def predict_path(graph_data):
    """Return (prediction, error)."""
    try:
        res = requests.post(f"{API_BASE}/predict", json=graph_data, timeout=120)
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the model API at {API_BASE}."
    except requests.exceptions.Timeout:
        return None, "Inference timed out after 120s."
    if res.status_code != 200:
        try:
            detail = res.json().get("detail", res.text)
        except ValueError:
            detail = res.text
        return None, f"API returned HTTP {res.status_code}: {detail}"
    return res.json(), None


# Training band and measured valid-and-optimal rates. The per-size figures are
# transcribed from results/ood_diagnostics_run5.txt (n=200 graphs per size); the
# band figure is checkpoints_run5/results.json eval_summary (n=1000).
TRAIN_LO, TRAIN_HI = 5, 20
ID_BAND_RATE = 83.6
OOD_BAND_RATE = 0.0
MEASURED_VO = {21: 41.5, 22: 3.0, 23: 0.0, 24: 0.0, 25: 0.5, 26: 0.0, 28: 0.0, 30: 0.0}


def distribution_badge(num_nodes):
    """(streamlit_severity, label, detail) for the current graph size."""
    if TRAIN_LO <= num_nodes <= TRAIN_HI:
        return (
            "success",
            "In distribution",
            f"Trained on N={TRAIN_LO}\u2013{TRAIN_HI}. Measured valid & optimal on that "
            f"band: {ID_BAND_RATE}% (n=1000). The gate was 90%, so expect roughly "
            "one graph in six to be wrong even here.",
        )
    if num_nodes in MEASURED_VO:
        measured = f"Measured valid & optimal at N={num_nodes}: {MEASURED_VO[num_nodes]}% (n=200)."
    else:
        measured = (
            f"Not measured at N={num_nodes} individually; the N=25\u201350 band scores "
            f"{OOD_BAND_RATE}% (n=3000)."
        )
    return (
        "error",
        "Out of distribution",
        f"The model was trained on N={TRAIN_LO}\u2013{TRAIN_HI} only, so N={num_nodes} is "
        f"outside its training range. {measured} The Dijkstra path below stays exact "
        "at any size and is the answer to trust here.",
    )


def node_positions(graph_data, half_width=300.0, half_height=250.0):
    """Deterministic layout keyed only on the graph, never on the prediction.

    Physics was previously left enabled, so every re-render (e.g. clicking Run
    Inference) restarted the force simulation and the graph visibly rearranged.
    Coordinates are normalised into a box the 600px canvas shows at zoom 1, since
    pyvis never calls network.fit().
    """
    G = nx.Graph()
    G.add_nodes_from(graph_data["node_ids"])
    for u, v, _w in graph_data["edges"]:
        G.add_edge(u, v)
    pos = nx.spring_layout(G, seed=int(graph_data["seed"]) % (2**31))

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    scale = min(2 * half_width / (max(xs) - min(xs)),
                2 * half_height / (max(ys) - min(ys)))
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    # y is negated: networkx grows upward, the browser canvas grows downward.
    return {n: ((p[0] - cx) * scale, -(p[1] - cy) * scale) for n, p in pos.items()}


def draw_graph(graph_data, prediction=None):
    if not graph_data:
        return ""

    net = Network(height="600px", width="100%", bgcolor="#0a0a0a", font_color="#f5f5f5")
    net.set_options("""
    var options = {
      "physics": {"enabled": false},
      "interaction": {"dragNodes": true, "zoomView": true},
      "edges": {
        "color": {"inherit": false},
        "smooth": {"type": "continuous"}
      }
    }
    """)

    positions = node_positions(graph_data)

    source = graph_data.get("source")
    target = graph_data.get("target")
    
    pred_edges = set()
    pred_nodes = set()
    if prediction and "model_path" in prediction:
        path = prediction["model_path"]
        pred_nodes = set(path)
        for i in range(len(path) - 1):
            u = min(path[i], path[i+1])
            v = max(path[i], path[i+1])
            pred_edges.add(f"{u}-{v}")
    
    # Add Nodes
    for node_id in graph_data["node_ids"]:
        is_path = node_id in pred_nodes
        is_src = node_id == source
        is_tgt = node_id == target
        
        color = "#111111"
        border_color = "#525252"
        size = 15
        
        if is_src:
            border_color = "#10b981"  # Emerald
            size = 20
        elif is_tgt:
            border_color = "#f43f5e"  # Rose
            size = 20
        elif is_path:
            border_color = "#00f0ff"  # Cyan
            
        # Always emit a shadow so the bounding box is consistent, preventing Pyvis auto-fit shifts.
        shadow_color = border_color if (is_src or is_tgt or is_path) else "rgba(0,0,0,0)"
        shadow = {"enabled": True, "color": shadow_color, "size": 15, "x": 0, "y": 0}
            
        title = f"Node {node_id}"
        if is_src: title += " (START)"
        if is_tgt: title += " (TARGET)"

        net.add_node(
            node_id, 
            label=str(node_id),
            title=title,
            x=positions[node_id][0],
            y=positions[node_id][1],
            color={"background": color, "border": border_color},
            borderWidth=3,  # Constant width ensures Pyvis bounding box stays stable
            size=size,
            shadow=shadow,
            font={"color": "#fff" if is_path else "#a3a3a3"}
        )
        
    # Add Edges
    for u, v, w in graph_data["edges"]:
        u_min = min(u, v)
        v_max = max(u, v)
        is_pred_edge = f"{u_min}-{v_max}" in pred_edges
        
        color = "#00f0ff" if is_pred_edge else "rgba(255,255,255,0.1)"
        width = 3 if is_pred_edge else 1
        
        net.add_edge(
            u, v, 
            label=str(w),
            color=color,
            width=width,
            font={"color": "rgba(255,255,255,0.5)", "size": 10, "align": "middle"}
        )
        
    # In-memory: writing to a fixed "graph.html" in the CWD instead would race
    # between concurrent browser sessions, and the old code swallowed every
    # failure into a blank panel.
    return net.generate_html()


# Initialize session state
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "api_error" not in st.session_state:
    st.session_state.api_error = None
if "seed_display" not in st.session_state:
    st.session_state.seed_display = ""

# The first graph has to exist before the Controls column renders, otherwise the
# Start/Target selectboxes have no options and never appear on load.
if st.session_state.graph_data is None:
    _gd, _err = generate_graph(12)
    st.session_state.graph_data = _gd
    st.session_state.api_error = _err
    if _gd:
        st.session_state.seed_display = str(_gd["seed"])

# --- HERO SECTION ---
st.markdown('<div class="hero-title">GRAPH TRANSFORMER</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">A from-scratch encoder-decoder Transformer trained to learn shortest-path behavior from weighted graphs and evaluated on larger unseen graphs.</div>', unsafe_allow_html=True)
st.markdown("""
<span class="badge">Pure PyTorch</span>
<span class="badge">No nn.Transformer</span>
<span class="badge">5–20 Node Training</span>
<span class="badge">25–50 Node OOD</span>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- GRAPH LAB SECTION ---
st.header("Interactive Graph Lab")
st.markdown("Run inference in real-time against the trained checkpoint.")

if st.session_state.api_error:
    st.error(st.session_state.api_error)

col1, col2 = st.columns([2, 1], gap="large")

with col2:
    st.subheader("Controls")

    num_nodes = st.slider(
        "Node count",
        min_value=5,
        max_value=50,
        value=12,
        key="num_nodes",
        help=f"The model was trained on N={TRAIN_LO}\u2013{TRAIN_HI}. Larger graphs are "
             "out of distribution and it fails on them; Dijkstra stays exact.",
    )
    seed_input = st.text_input(
        "Seed",
        value=st.session_state.seed_display,
        key="seed_input",
        placeholder="blank = random",
    )

    if st.button("Generate Graph", use_container_width=True):
        raw = seed_input.strip()
        try:
            seed = int(raw) if raw else None
        except ValueError:
            st.session_state.api_error = f"Seed must be a whole number, got '{raw}'."
        else:
            gd, err = generate_graph(num_nodes, seed)
            st.session_state.graph_data = gd
            st.session_state.api_error = err
            st.session_state.prediction = None
            if gd:
                st.session_state.seed_display = str(gd["seed"])

    gd = st.session_state.graph_data
    if gd:
        st.caption(
            f"N = {gd['num_nodes']}  \u2022  E = {len(gd['edges'])}  \u2022  seed = {gd['seed']}"
        )

        severity, label, detail = distribution_badge(gd["num_nodes"])
        getattr(st, severity)(f"**{label}** \u2014 {detail}")

        node_ids = gd["node_ids"]
        for key, fallback in (("start_node", gd["source"]), ("target_node", gd["target"])):
            if st.session_state.get(key) not in node_ids:
                st.session_state[key] = fallback

        source = st.selectbox("Start node", node_ids, key="start_node")
        target = st.selectbox("Target node", node_ids, key="target_node")

        if source != gd["source"] or target != gd["target"]:
            gd["source"] = source
            gd["target"] = target
            st.session_state.prediction = None

        st.markdown("<br>", unsafe_allow_html=True)
        if source == target:
            st.warning("Start and target are the same node. Pick two different nodes.")
        elif st.button("Run Transformer Inference", type="primary", use_container_width=True):
            with st.spinner("Running inference..."):
                pred, err = predict_path(gd)
                st.session_state.prediction = pred
                st.session_state.api_error = err

        if st.session_state.prediction:
            pred = st.session_state.prediction
            st.markdown("### Inference Results")
            model_path = pred.get("model_path", [])
            dijkstra_path = pred.get("dijkstra_path", [])
            raw_tokens = pred.get("model_raw_tokens", [])

            st.markdown("**Transformer Raw Tokens**")
            st.code(" ".join(raw_tokens), language="text")

            st.markdown("**Transformer Parsed Path**")
            st.code(" \u2192 ".join(map(str, model_path)), language="text")

            st.markdown(f"**Dijkstra (Ground Truth)** \u2014 cost {pred.get('dijkstra_cost')}")
            st.code(" \u2192 ".join(map(str, dijkstra_path)), language="text")

            if pred.get("is_optimal"):
                st.success("MATCH: Path is valid and optimal.")
            elif pred.get("valid_path"):
                st.warning("SUBOPTIMAL: Path is valid but not optimal.")
            else:
                st.error("INVALID: Model produced an invalid or incomplete path.")
    else:
        st.info("No graph loaded. Fix the error above, then click 'Generate Graph'.")

with col1:
    gd = st.session_state.graph_data
    if gd:
        components.html(draw_graph(gd, st.session_state.prediction), height=620)
    else:
        st.warning(
            f"No graph to draw. Ensure the FastAPI backend is running at {API_BASE}."
        )


st.markdown("<hr>", unsafe_allow_html=True)

# --- HOW IT WORKS ---
st.header("Architecture Pipeline")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 1. Tokenization")
    st.markdown("The graph structure (nodes, edges, weights) is flattened into a sequential token representation suitable for transformer attention.")
with c2:
    st.markdown("### 2. Encoder")
    st.markdown("Bidirectional self-attention builds a rich, global understanding of the entire graph topology and edge weights.")
with c3:
    st.markdown("### 3. Decoder & Cross-Attention")
    st.markdown("An autoregressive decoder generates the optimal path step-by-step, constantly attending back to the encoder's graph representation via cross-attention.")

st.markdown("<hr>", unsafe_allow_html=True)

# --- RESULTS ---
st.header("Experiment Validation")

# Metrics come from the same checkpoint the API serves, so the panel and the
# live predictions below describe one model.
results_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints_run5", "results.json")
try:
    with open(results_path, "r") as f:
        results = json.load(f)
        
    colA, colB = st.columns(2)
    with colA:
        val_opt = results.get("eval_summary", {}).get("valid_and_optimal_fraction", 0)
        st.metric("In-Distribution (5-20 Nodes)", f"{(val_opt * 100):.1f}%", "Valid & Optimal")

        if results.get("gate_passed") is False:
            st.warning(
                "The 90% acceptance gate was **not met** at this score and was waived "
                "so out-of-distribution evaluation could proceed. Out of distribution "
                "(N=25-50) this model scores 0.0%. See `RESULTS.md`."
            )
        
        # Plot Loss Curve using Plotly
        if "val_loss" in results and "train_loss" in results:
            df = pd.DataFrame({
                "Epoch": range(1, len(results["train_loss"]) + 1),
                "Train Loss": results["train_loss"],
                "Val Loss": results["val_loss"]
            })
            fig = px.line(df, x="Epoch", y=["Train Loss", "Val Loss"], template="plotly_dark", title="Training vs Validation Loss")
            fig.update_layout(plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a")
            st.plotly_chart(fig, use_container_width=True)
            
    with colB:
        st.markdown("### Model Configuration")
        cfg = results.get("config", {})
        
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("d_model", cfg.get("d_model", "-"))
        m_c2.metric("Layers", cfg.get("n_layers", "-"))
        m_c1.metric("Attention Heads", cfg.get("n_heads", "-"))
        m_c2.metric("Feed Forward", cfg.get("d_ff", "-"))
        
except FileNotFoundError:
    st.info("Static results file (results.json) not found. Run training and evaluation to generate metrics.")

st.markdown("<hr>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<center>Built with PyTorch &bull; Verified against Dijkstra &bull; Graph Transformer Research Demo</center>", unsafe_allow_html=True)
