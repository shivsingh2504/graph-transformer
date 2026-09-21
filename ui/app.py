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


def generate_graph(num_nodes=15):
    try:
        res = requests.get(f"{API_BASE}/generate", params={"num_nodes": num_nodes})
        if res.status_code == 200:
            return res.json()
        return None
    except requests.exceptions.ConnectionError:
        return None

def predict_path(graph_data):
    try:
        res = requests.post(f"{API_BASE}/predict", json=graph_data)
        if res.status_code == 200:
            return res.json()
        return None
    except requests.exceptions.ConnectionError:
        return None

def draw_graph(graph_data, prediction=None):
    if not graph_data:
        return ""
        
    net = Network(height="600px", width="100%", bgcolor="#0a0a0a", font_color="#f5f5f5")
    
    # Physics settings for a smooth, stable layout
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -100,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      },
      "edges": {
        "color": {"inherit": false},
        "smooth": {"type": "continuous"}
      }
    }
    """)

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
            
        # Optional: Add a glow shadow via Pyvis kwargs
        shadow = False
        if is_src or is_tgt or is_path:
            shadow = {"enabled": True, "color": border_color, "size": 15, "x": 0, "y": 0}
            
        title = f"Node {node_id}"
        if is_src: title += " (START)"
        if is_tgt: title += " (TARGET)"

        net.add_node(
            node_id, 
            label=str(node_id),
            title=title,
            color={"background": color, "border": border_color},
            borderWidth=3 if (is_src or is_tgt or is_path) else 1,
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
        
    try:
        path = "graph.html"
        net.save_graph(path)
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        # Clean up the generated file to avoid littering
        os.remove(path)
        return html
    except Exception as e:
        return ""


# Initialize session state
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

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

col1, col2 = st.columns([2, 1], gap="large")

with col2:
    st.subheader("Controls")
    import random
    if st.button("Generate Random Graph", use_container_width=True):
        st.session_state.graph_data = generate_graph(random.randint(5, 20))
        st.session_state.prediction = None

    if st.session_state.graph_data:
        node_ids = st.session_state.graph_data["node_ids"]
        source = st.selectbox("Start Node", node_ids, index=node_ids.index(st.session_state.graph_data["source"]))
        target = st.selectbox("Target Node", node_ids, index=node_ids.index(st.session_state.graph_data["target"]))
        
        # Update graph_data if source/target changed
        if source != st.session_state.graph_data["source"] or target != st.session_state.graph_data["target"]:
            st.session_state.graph_data["source"] = source
            st.session_state.graph_data["target"] = target
            st.session_state.prediction = None
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Run Transformer Inference", type="primary", use_container_width=True):
            with st.spinner("Running inference..."):
                st.session_state.prediction = predict_path(st.session_state.graph_data)
                
        if st.session_state.prediction:
            pred = st.session_state.prediction
            st.markdown("### Inference Results")
            model_path = pred.get("model_path", [])
            dijkstra_path = pred.get("dijkstra_path", [])
            raw_tokens = pred.get("model_raw_tokens", [])
            
            st.markdown("**Transformer Raw Tokens**")
            st.code(" ".join(raw_tokens), language="text")
            
            st.markdown("**Transformer Parsed Path**")
            st.code(" → ".join(map(str, model_path)), language="text")
            
            st.markdown("**Dijkstra (Ground Truth)**")
            st.code(" → ".join(map(str, dijkstra_path)), language="text")
            
            if pred.get("is_optimal"):
                st.success("MATCH: Path is valid and optimal.")
            elif pred.get("valid_path"):
                st.warning("SUBOPTIMAL: Path is valid but not optimal.")
            else:
                st.error("INVALID: Model produced an invalid or incomplete path.")
    else:
        st.info("Click 'Generate Random Graph' to start.")

with col1:
    if not st.session_state.graph_data:
        # Load initial
        import random
        st.session_state.graph_data = generate_graph(random.randint(5, 20))
        
    html = draw_graph(st.session_state.graph_data, st.session_state.prediction)
    if html:
        components.html(html, height=620)
    else:
        st.warning("Ensure the FastAPI backend is running at http://127.0.0.1:8000")


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

# Check if results.json exists in the current directory or nearby
results_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints_run2", "results.json")
try:
    with open(results_path, "r") as f:
        results = json.load(f)
        
    colA, colB = st.columns(2)
    with colA:
        val_opt = results.get("eval_summary", {}).get("valid_and_optimal_fraction", 0)
        st.metric("Training Distribution (5-20 Nodes)", f"{(val_opt * 100):.1f}%", "Valid & Optimal")
        
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
