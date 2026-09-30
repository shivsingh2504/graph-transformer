import streamlit as st
import requests
import json
import networkx as nx
from pyvis.network import Network
import os

API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000/api").rstrip("/")
if not API_BASE.startswith(("http://", "https://")):
    API_BASE = f"http://{API_BASE}"

st.set_page_config(page_title="NexPath | Graph Intelligence", page_icon=":material/hub:", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 9% 8%, rgba(237, 112, 101, 0.14), transparent 34%),
            radial-gradient(ellipse at 91% 26%, rgba(151, 119, 194, 0.12), transparent 33%),
            radial-gradient(ellipse at 54% 96%, rgba(67, 181, 157, 0.07), transparent 35%),
            linear-gradient(135deg, #1c1923 0%, #211d29 52%, #191821 100%);
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background: rgba(28, 25, 35, 0.62); }
    [data-testid="stIFrame"] {
        overflow: hidden;
        border: 1px solid #665360;
        border-radius: 15px;
        background: #101b2c;
        box-shadow: 0 14px 34px rgba(5, 12, 23, 0.26);
    }
    [data-testid="stIFrame"] iframe { border: 0; border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

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


def path_cost(path, edges):
    """Return the summed edge cost for a predicted node sequence, if valid."""
    costs = {frozenset((u, v)): weight for u, v, weight in edges}
    total = 0
    for u, v in zip(path, path[1:]):
        weight = costs.get(frozenset((u, v)))
        if weight is None:
            return None
        total += weight
    return total if path else None


TRAIN_LO, TRAIN_HI = 5, 20


def node_positions(graph_data, half_width=380.0, half_height=195.0):
    """Deterministic layout keyed only on the graph, never on the prediction.

    Physics was previously left enabled, so every re-render (e.g. clicking Run
    Inference) restarted the force simulation and the graph visibly rearranged.
    A slightly stronger repulsion spreads dense graphs before positions are
    normalized into the canvas. The iframe fits the stable layout on first draw.
    """
    G = nx.Graph()
    G.add_nodes_from(graph_data["node_ids"])
    for u, v, _w in graph_data["edges"]:
        G.add_edge(u, v)
    pos = nx.spring_layout(
        G,
        seed=int(graph_data["seed"]) % (2**31),
        k=2.0 / (len(G) ** 0.5),
        iterations=200,
    )

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

    net = Network(height="490px", width="100%", bgcolor="#101b2c", font_color="#e7eef8")
    net.set_options("""
    var options = {
      "physics": {"enabled": false},
      "interaction": {"dragNodes": true, "zoomView": true},
      "edges": {
        "color": {"inherit": false},
        "smooth": {"type": "continuous"},
        "font": {"size": 14, "face": "Inter, sans-serif", "color": "#e7eef8",
                 "background": "#1b2c43", "strokeWidth": 0, "bold": true},
        "selectionWidth": 1.5,
        "hoverWidth": 1.2
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
        
        color = "#263b57"
        border_color = "#6c829f"
        size = 18
        
        if is_src:
            color = "#153e45"
            border_color = "#42d9bd"
            size = 20
        elif is_tgt:
            color = "#493526"
            border_color = "#ffb86b"
            size = 20
        elif is_path:
            color = "#203b59"
            border_color = "#78aaff"
            
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
            font={"color": "#f2f7ff", "size": 16, "face": "Inter, sans-serif", "bold": True}
        )
        
    # Add Edges
    for u, v, w in graph_data["edges"]:
        u_min = min(u, v)
        v_max = max(u, v)
        is_pred_edge = f"{u_min}-{v_max}" in pred_edges
        
        color = "#35d6bf" if is_pred_edge else "#536b89"
        width = 3 if is_pred_edge else 1.5
        
        net.add_edge(
            u, v, 
            label=str(w),
            color=color,
            width=width,
            font={"color": "#e7eef8", "size": 14, "align": "middle",
                  "background": "#1b2c43", "strokeWidth": 0, "bold": True}
        )
        
    # In-memory: writing to a fixed "graph.html" in the CWD instead would race
    # between concurrent browser sessions, and the old code swallowed every
    # failure into a blank panel.
    html = net.generate_html()
    fit_script = "network.once('afterDrawing', function () { network.fit({ animation: false }); });"
    return html.replace(
        "network = new vis.Network(container, data, options);",
        "network = new vis.Network(container, data, options);\n" + fit_script,
    )


SHORTEST_PATH_DIAGRAM = r"""
graph shortest_path {
  graph [rankdir=LR, bgcolor="transparent", pad="0.25", nodesep="0.55", ranksep="0.65",
         label="LOWEST TOTAL COST WINS", labelloc="b", fontname="Inter", fontsize=12, fontcolor="#9db3cf"];
  node [shape=circle, style="filled", fixedsize=true, width=0.58, height=0.58,
        fontname="Inter", fontsize=13, fontcolor="#edf4ff", color="#6e86a6", penwidth=1.5];
  edge [fontname="Inter", fontsize=12, fontcolor="#cfdaea", color="#536b89", penwidth=1.5];
  A [label="A\nSTART", fillcolor="#164148", color="#42d9bd"];
  B [label="B", fillcolor="#263b57"];
  C [label="C", fillcolor="#263b57"];
  D [label="D", fillcolor="#263b57"];
  T [label="T\nGOAL", fillcolor="#493526", color="#ffb86b"];
  A -- B [label="2", color="#35d6bf", fontcolor="#35d6bf", penwidth=3];
  A -- C [label="5"];
  B -- C [label="1", color="#35d6bf", fontcolor="#35d6bf", penwidth=3];
  B -- D [label="4"];
  C -- D [label="2", color="#35d6bf", fontcolor="#35d6bf", penwidth=3];
  C -- T [label="7"];
  D -- T [label="3", color="#35d6bf", fontcolor="#35d6bf", penwidth=3];
}
"""


# One connected, accessible SVG makes the flow visible without turning every
# Transformer operation into another Streamlit card or button.
TRANSFORMER_DIAGRAM = st.components.v2.component(
    "pathfinder_transformer_architecture",
    html="""
<section class="architecture" aria-label="Interactive Transformer architecture">
  <div class="diagram-scroll">
    <svg class="diagram" viewBox="0 0 920 930" role="group" aria-labelledby="diagram-title diagram-description">
      <title id="diagram-title">How NexPath turns a weighted graph into a route</title>
      <desc id="diagram-description">Graph tokens flow through a three-layer encoder. The decoder combines the graph context with the route generated so far and predicts the next node.</desc>
      <defs>
        <linearGradient id="encFill" x1="0" x2="1" y1="0" y2="1"><stop stop-color="#173c50"/><stop offset="1" stop-color="#142b43"/></linearGradient>
        <linearGradient id="decFill" x1="0" x2="1" y1="0" y2="1"><stop stop-color="#49344d"/><stop offset="1" stop-color="#282d4b"/></linearGradient>
        <linearGradient id="attnFill" x1="0" x2="1"><stop stop-color="#14665f"/><stop offset="1" stop-color="#1b5269"/></linearGradient>
        <linearGradient id="ffFill" x1="0" x2="1"><stop stop-color="#76502c"/><stop offset="1" stop-color="#64452f"/></linearGradient>
        <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8ca9c5"/></marker>
        <marker id="arrowMint" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#58dfc6"/></marker>
      </defs>
      <text x="40" y="42" class="eyebrow">THE ROUTE-FINDING PIPELINE</text>
      <text x="40" y="70" class="subline">Select a stage to see what the model computes.</text>

      <!-- Graph understanding / encoder stream -->
      <g class="stage" data-stage="graph" role="button" tabindex="0" aria-label="Graph tokens: select to learn about the graph input">
        <rect class="node input" x="54" y="105" width="330" height="88" rx="19"/>
        <text x="78" y="137" class="kicker">01 · INPUT GRAPH</text>
        <text x="78" y="166" class="label">Nodes · edges · costs · query</text>
        <text x="356" y="160" class="glyph">G</text>
      </g>
      <path class="flow" d="M219 194 V222" marker-end="url(#arrow)"/>
      <g class="stage" data-stage="embedding" role="button" tabindex="0" aria-label="Token and position embedding: select to learn about embeddings">
        <rect class="node embed" x="92" y="229" width="254" height="68" rx="16"/>
        <text x="219" y="257" text-anchor="middle" class="kicker">TOKEN + POSITION</text>
        <text x="219" y="281" text-anchor="middle" class="label">Embedding · 128 features</text>
      </g>
      <path class="flow" d="M219 299 V326" marker-end="url(#arrow)"/>
      <rect class="lane encoder-lane" x="40" y="337" width="358" height="294" rx="25"/>
      <text x="63" y="370" class="lane-title">ENCODER <tspan class="repeat">× 3 LAYERS</tspan></text>
      <text x="63" y="393" class="lane-note">Reads the whole graph at once</text>
      <g class="stage" data-stage="self_attention" role="button" tabindex="0" aria-label="Encoder self-attention: select to learn">
        <rect class="node attention" x="72" y="411" width="294" height="77" rx="17"/>
        <text x="219" y="440" text-anchor="middle" class="kicker">4 PARALLEL HEADS</text>
        <text x="219" y="466" text-anchor="middle" class="label">Self-attention</text>
        <circle cx="116" cy="451" r="4" class="dot"/><circle cx="128" cy="451" r="4" class="dot"/><circle cx="140" cy="451" r="4" class="dot"/><circle cx="152" cy="451" r="4" class="dot"/>
      </g>
      <path class="flow" d="M219 489 V506" marker-end="url(#arrow)"/>
      <g class="stage" data-stage="residual" role="button" tabindex="0" aria-label="Residual connection and layer normalization: select to learn">
        <rect class="node norm" x="112" y="512" width="214" height="39" rx="12"/>
        <text x="219" y="538" text-anchor="middle" class="small-label">Residual + LayerNorm</text>
      </g>
      <path class="flow" d="M219 552 V568" marker-end="url(#arrow)"/>
      <g class="stage" data-stage="feed_forward" role="button" tabindex="0" aria-label="Feed-forward network: select to learn">
        <rect class="node feedforward" x="72" y="574" width="294" height="58" rx="13"/>
        <text x="219" y="598" text-anchor="middle" class="small-label">Feed-forward network</text>
        <text x="219" y="619" text-anchor="middle" class="micro-label">128 → 512 → 128 · Add + LayerNorm</text>
      </g>
      <path class="residual-path" d="M369 454 C390 454 390 533 369 533"/>
      <path class="flow mint-flow" d="M219 647 V670" marker-end="url(#arrowMint)"/>
      <g class="stage" data-stage="context" role="button" tabindex="0" aria-label="Encoded graph context: select to learn">
        <rect class="node memory" x="73" y="677" width="292" height="67" rx="17"/>
        <text x="219" y="704" text-anchor="middle" class="kicker">ENCODER MEMORY</text>
        <text x="219" y="730" text-anchor="middle" class="label">Context for every route decision</text>
      </g>

      <!-- Route generation / decoder stream -->
      <g class="stage" data-stage="prefix" role="button" tabindex="0" aria-label="Route prefix: select to learn about generated route tokens">
        <rect class="node input route-input" x="536" y="105" width="330" height="88" rx="19"/>
        <text x="560" y="137" class="kicker">02 · ROUTE SO FAR</text>
        <text x="560" y="166" class="label">[BOS] → nodeₛ → …</text>
        <text x="836" y="160" class="glyph">↗</text>
      </g>
      <path class="flow" d="M701 194 V222" marker-end="url(#arrow)"/>
      <g class="stage" data-stage="embedding" role="button" tabindex="0" aria-label="Decoder token and position embedding: select to learn about embeddings">
        <rect class="node embed" x="574" y="229" width="254" height="68" rx="16"/>
        <text x="701" y="257" text-anchor="middle" class="kicker">TOKEN + POSITION</text>
        <text x="701" y="281" text-anchor="middle" class="label">Current route prefix</text>
      </g>
      <path class="flow" d="M701 299 V326" marker-end="url(#arrow)"/>
      <rect class="lane decoder-lane" x="522" y="337" width="358" height="367" rx="25"/>
      <text x="545" y="370" class="lane-title">DECODER <tspan class="repeat">× 3 LAYERS</tspan></text>
      <text x="545" y="393" class="lane-note">Builds one next-node decision at a time</text>
      <g class="stage" data-stage="masked_attention" role="button" tabindex="0" aria-label="Masked self-attention: select to learn">
        <rect class="node attention" x="554" y="411" width="294" height="69" rx="17"/>
        <text x="701" y="439" text-anchor="middle" class="kicker">CAUSAL · 4 HEADS</text>
        <text x="701" y="463" text-anchor="middle" class="label">Masked self-attention</text>
      </g>
      <path class="flow" d="M701 481 V495" marker-end="url(#arrow)"/>
      <g class="stage" data-stage="residual" role="button" tabindex="0" aria-label="Residual connection and layer normalization: select to learn">
        <rect class="node norm" x="594" y="500" width="214" height="34" rx="11"/>
        <text x="701" y="523" text-anchor="middle" class="small-label">Residual + LayerNorm</text>
      </g>
      <path class="flow mint-flow" d="M701 536 V548" marker-end="url(#arrowMint)"/>
      <g class="stage" data-stage="cross_attention" role="button" tabindex="0" aria-label="Cross-attention: select to learn how decoder reads graph context">
        <rect class="node cross" x="554" y="553" width="294" height="71" rx="17"/>
        <text x="701" y="581" text-anchor="middle" class="kicker">READS ENCODER MEMORY</text>
        <text x="701" y="607" text-anchor="middle" class="label">Cross-attention</text>
      </g>
      <path class="flow" d="M701 625 V638" marker-end="url(#arrow)"/>
      <g class="stage" data-stage="feed_forward" role="button" tabindex="0" aria-label="Feed-forward network: select to learn">
        <rect class="node feedforward" x="574" y="645" width="254" height="58" rx="13"/>
        <text x="701" y="670" text-anchor="middle" class="small-label">Feed-forward network</text>
        <text x="701" y="690" text-anchor="middle" class="micro-label">128 → 512 → 128 · Add + LayerNorm</text>
      </g>
      <!-- Context is routed across the diagram into the decoder's cross-attention. -->
      <path class="context-path" d="M365 710 H447 V589 H548" marker-end="url(#arrowMint)"/>
      <rect class="context-label-bg" x="388" y="642" width="111" height="27" rx="13"/>
      <text x="443" y="660" text-anchor="middle" class="context-label">GRAPH CONTEXT</text>

      <path class="flow" d="M701 706 V733" marker-end="url(#arrow)"/>
      <g class="stage" data-stage="projection" role="button" tabindex="0" aria-label="Output projection and softmax: select to learn">
        <rect class="node projection" x="586" y="740" width="230" height="65" rx="16"/>
        <text x="701" y="767" text-anchor="middle" class="kicker">LINEAR + SOFTMAX</text>
        <text x="701" y="790" text-anchor="middle" class="label">Score next-token options</text>
      </g>
      <path class="flow mint-flow" d="M701 807 V831" marker-end="url(#arrowMint)"/>
      <g class="stage" data-stage="next_node" role="button" tabindex="0" aria-label="Choose and append the next node: select to learn">
        <rect class="node output" x="596" y="838" width="210" height="54" rx="16"/>
        <text x="701" y="872" text-anchor="middle" class="label">Choose next node</text>
      </g>
      <path class="feedback-path" d="M808 865 H888 V149 H872" marker-end="url(#arrowMint)"/>
      <text x="902" y="524" class="feedback-label" transform="rotate(-90 902 524)">APPEND TOKEN · REPEAT UNTIL [EOS]</text>

      <text x="62" y="790" class="footnote">Graph tokens encode source, destination, and weighted edges.</text>
      <text x="62" y="811" class="footnote">The route prefix grows until the model emits [EOS].</text>
    </svg>
  </div>
  <div class="inspector" aria-live="polite" aria-atomic="true">
    <div class="inspector-index" id="stage-index">01</div>
    <div><div class="inspector-title" id="stage-title">Graph tokens</div><p class="inspector-copy" id="stage-copy">The input sequence lists the source and target first, followed by each edge as two node tokens and its weight token.</p></div>
  </div>
</section>
""",
    css="""
.architecture { --ink:#edf4ff; --muted:#9db3ca; --line:#58738e; color:var(--ink); font-family:inherit; }
.diagram-scroll { display:flex; justify-content:center; overflow-x:auto; border:1px solid #29415d; border-radius:22px; background:radial-gradient(ellipse at 50% 38%,#1a2a40 0%,#111c2c 72%); scrollbar-color:#48627e #142033; }
.diagram { display:block; width:min(100%, 760px); min-width:680px; height:auto; margin-inline:auto; padding:10px; box-sizing:border-box; }
.eyebrow { fill:#69ddc9; font-size:13px; font-weight:750; letter-spacing:2px; }
.subline,.lane-note { fill:#9db3ca; font-size:14px; }
.lane-title { fill:#eaf3ff; font-size:17px; font-weight:750; letter-spacing:1.1px; }
.repeat { fill:#a9bad0; font-size:13px; font-weight:600; }
.node { stroke-width:1.5; transition:filter .18s ease,stroke .18s ease,transform .18s ease; }
.input { fill:#1b3147; stroke:#6c91b4; }.route-input { fill:#352c49; stroke:#ab88c5; }
.embed { fill:#203e4b; stroke:#58bcb7; }.attention { fill:url(#attnFill); stroke:#59dcca; }
.norm { fill:#344055; stroke:#8295b0; }.feedforward { fill:url(#ffFill); stroke:#e6b66b; }
.memory { fill:#123d3a; stroke:#59dfc5; }.cross { fill:#254f58; stroke:#68e0cf; }
.projection { fill:#383653; stroke:#b29ce2; }.output { fill:#14534d; stroke:#58dfc6; }
.lane { fill:rgba(21,34,51,.7); stroke-width:1.3; stroke-dasharray:5 6; }.encoder-lane { stroke:#438a95; }.decoder-lane { stroke:#8970a7; }
.label { fill:#f1f6ff; font-size:16px; font-weight:680; }.small-label { fill:#e4edf8; font-size:14px; font-weight:630; }
.micro-label { fill:#d0d4c7; font-size:11px; font-weight:650; }
.kicker { fill:#c2d4e8; font-size:11px; font-weight:760; letter-spacing:1.2px; }.glyph { fill:#81d8cc; font-size:30px; font-weight:800; }
.flow,.context-path,.feedback-path,.residual-path { fill:none; stroke:#8ca9c5; stroke-width:2; }
.mint-flow,.context-path,.feedback-path { stroke:#58dfc6; }.residual-path { stroke:#8195ab; stroke-width:1.5; stroke-dasharray:4 4; }
.dot { fill:#c7fbef; }.context-label-bg { fill:#163a3c; stroke:#438e88; }.context-label { fill:#9bf0df; font-size:10px; font-weight:750; letter-spacing:.8px; }
.feedback-label { fill:#75d9c9; font-size:10px; font-weight:700; letter-spacing:1px; }.footnote { fill:#91a8c1; font-size:12px; }
.stage { cursor:pointer; outline:none; }.stage:hover .node,.stage:focus-visible .node,.stage.is-selected .node { stroke:#fff; stroke-width:2.7; filter:drop-shadow(0 0 8px rgba(88,223,198,.46)); }
.stage:focus-visible .node { stroke-dasharray:5 3; }.stage.is-selected .node { filter:drop-shadow(0 0 10px rgba(88,223,198,.4)); }
.inspector { display:grid; grid-template-columns:42px 1fr; align-items:start; gap:14px; margin-top:14px; padding:16px 18px; border:1px solid #29415d; border-radius:16px; background:linear-gradient(110deg,#17273a,#182337); }
.inspector-index { display:grid; place-items:center; width:36px; height:36px; border-radius:12px; background:#174b49; color:#8df0db; font-size:13px; font-weight:800; }
.inspector-title { margin:0 0 4px; color:#f1f6ff; font-size:16px; font-weight:750; }.inspector-copy { margin:0; color:#b5c6d9; font-size:14px; line-height:1.55; }
@media (max-width:700px) { .diagram { min-width:680px; padding:8px; }.inspector { padding:13px; } }
@media (prefers-reduced-motion: reduce) { .node { transition:none; } }
""",
    js="""
export default function (component) {
  const { data, parentElement } = component
  const root = parentElement.querySelector(".architecture")
  if (!root) return
  const details = data?.details ?? {}
  const title = root.querySelector("#stage-title")
  const copy = root.querySelector("#stage-copy")
  const index = root.querySelector("#stage-index")
  const stages = [...root.querySelectorAll(".stage")]
  const select = (stage) => {
    stages.forEach((item) => {
      item.classList.toggle("is-selected", item === stage)
      item.setAttribute("aria-pressed", item === stage ? "true" : "false")
    })
    const key = stage.dataset.stage
    const item = details[key] ?? details.graph
    title.textContent = item.title
    copy.textContent = item.description
    index.textContent = item.index
  }
  const handleClick = (event) => {
    const stage = event.target.closest(".stage")
    if (stage && root.contains(stage)) select(stage)
  }
  const handleKey = (event) => {
    if ((event.key === "Enter" || event.key === " ") && event.target.matches(".stage")) {
      event.preventDefault()
      select(event.target)
    }
  }
  root.addEventListener("click", handleClick)
  root.addEventListener("keydown", handleKey)
  const initial = root.querySelector('.stage[data-stage="graph"]')
  if (initial) {
    initial.classList.add("is-selected")
    initial.setAttribute("aria-pressed", "true")
  }
  return () => {
    root.removeEventListener("click", handleClick)
    root.removeEventListener("keydown", handleKey)
  }
}
""",
)

TRANSFORMER_DETAILS = {
    "graph": {"index": "01", "title": "Graph tokens", "description": "The input sequence lists the source and target first, followed by each edge as two node tokens and its weight token."},
    "embedding": {"index": "02", "title": "Token + position embedding", "description": "Each token becomes a 128-value vector. A sinusoidal position signal is added so sequence order is represented."},
    "self_attention": {"index": "03", "title": "Encoder self-attention", "description": "Four heads compare every graph token with the others, letting node, edge, and cost information build shared context."},
    "residual": {"index": "04", "title": "Residual + LayerNorm", "description": "The layer adds its input back to the transformed values, then normalizes them. This stabilizes information as layers are stacked."},
    "feed_forward": {"index": "05", "title": "Feed-forward network", "description": "A position-wise network expands each token from 128 to 512 features, applies ReLU, and projects it back to 128."},
    "context": {"index": "06", "title": "Encoder memory", "description": "After three encoder layers, this contextual representation is available to the decoder at every route-generation step."},
    "prefix": {"index": "07", "title": "Route prefix", "description": "The decoder sees the start token and only the nodes generated so far. Each prediction is appended before the next step."},
    "masked_attention": {"index": "08", "title": "Masked self-attention", "description": "A causal mask prevents the decoder from seeing future route tokens, preserving one-step-at-a-time generation."},
    "cross_attention": {"index": "09", "title": "Cross-attention", "description": "The current route representation queries the encoder memory. This is how each decision can consult the graph and its edge costs."},
    "projection": {"index": "10", "title": "Next-token scores", "description": "A linear projection and softmax produce a probability distribution over possible next tokens; greedy decoding chooses the highest score."},
    "next_node": {"index": "11", "title": "Append and repeat", "description": "The selected node joins the route prefix. Decoding repeats until the model emits [EOS], which ends the route."},
}


# Initialize session state
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "api_error" not in st.session_state:
    st.session_state.api_error = None
if "seed_display" not in st.session_state:
    st.session_state.seed_display = ""
# --- HERO SECTION ---
with st.container(border=True):

    st.title("NexPath", icon=":material/hub:")
    st.markdown("### A Transformer is a neural network that uses attention to learn relationships across a sequence of information.")
    st.caption("NexPath turns a graph’s nodes, links, and costs into tokens. Its encoder reads the whole graph; its decoder predicts a route one node at a time.")
    st.caption("We compare the predicted route’s total cost with Dijkstra’s exact shortest path to check whether it found the cheapest answer.")

st.markdown("<hr>", unsafe_allow_html=True)

st.header("Shortest Path, One Example", icon=":material/route:")
problem_col, example_col = st.columns([0.9, 1.1], gap="large")
with problem_col:
    st.markdown("A graph is **nodes joined by edges**. Each edge has a cost. The shortest path is the route with the **lowest total cost**—even if it uses more links.")
    st.info("Highlighted route: 2 + 1 + 2 + 3 = **8**.", icon=":material/lightbulb:")
with example_col:
    st.graphviz_chart(SHORTEST_PATH_DIAGRAM, width="stretch")

st.header("Inside the Transformer", icon=":material/schema:")
st.markdown("A route is generated one node at a time. Select a stage in the diagram for a concise explanation.")
TRANSFORMER_DIAGRAM(data={"details": TRANSFORMER_DETAILS}, key="transformer_architecture")

st.header("Research Outcomes", icon=":material/emoji_events:")
st.markdown("Run 8 was evaluated on 1,000 held-out graphs in the supported 5–20 node range.")
achievement_cols = st.columns(3)
achievement_cols[0].metric("91.1%", "Valid & Optimal", "911 / 1,000 · 90% Gate Passed")
achievement_cols[1].metric("98.7%", "Valid Route Rate", "987 / 1,000 follow graph edges")
achievement_cols[2].metric("500,000", "Training Examples", "Weighted graph examples")
st.caption("Verified final checkpoint · 5–20 node operating range")

st.header("Model Configuration", icon=":material/tune:")
st.caption("A little more detail on what each design choice does inside the network.")
results_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints_run8", "results.txt")
try:
    with open(results_path, "r") as results_file:
        model_config = json.load(results_file).get("config", {})
except (FileNotFoundError, json.JSONDecodeError):
    model_config = {}

with st.expander("What Each Setting Does", expanded=False):
    layers = model_config.get("n_layers", 3)
    d_model = model_config.get("d_model", 128)
    n_heads = model_config.get("n_heads", 4)
    d_ff = model_config.get("d_ff", 512)
    dropout = model_config.get("dropout", 0.0)
    config_col1, config_col2 = st.columns(2, gap="medium")
    with config_col1.container(border=True):
        st.markdown(f"### {layers} Encoder + Decoder Layers")
        st.write("Each layer repeats attention and feed-forward processing. Stacking them lets the network refine graph context before building the route.")
    with config_col2.container(border=True):
        st.markdown(f"### {d_model} Features Per Token")
        st.write(f"Each token is represented by a {d_model}-value vector. Attention and feed-forward blocks transform this shared representation.")
    with config_col1.container(border=True):
        st.markdown(f"### {n_heads} Attention Heads")
        st.write(f"Heads examine relationships in parallel, with {d_model // max(n_heads, 1)} features per head.")
    with config_col2.container(border=True):
        st.markdown(f"### {d_ff}-Unit Feed-Forward Block")
        st.write(f"After attention, each token expands from {d_model} to {d_ff} features, then returns to {d_model} for the next layer.")
    with config_col1.container(border=True):
        st.markdown(f"### {dropout:g} Dropout")
        st.write("Dropout randomly masks activations during training. The final Run 8 configuration uses none.")

st.markdown("<hr>", unsafe_allow_html=True)

# --- GRAPH LAB SECTION ---
st.header("Try it yourself", icon=":material/hub:")
st.markdown("Choose two nodes and explore a route through the graph.")
with st.expander("How to use the graph lab"):
    st.markdown("Choose a size from 5–20, optionally enter a whole-number seed, and generate a graph. Select different start and target nodes, then run inference. The result is marked optimal, valid but suboptimal, or invalid; Dijkstra’s path is the exact reference.")

# Load the initial sample only after the explainer is on screen, so API startup
# does not hold back the page's first useful content.
if st.session_state.graph_data is None:
    with st.spinner("Preparing your sample graph..."):
        _gd, _err = generate_graph(12)
    st.session_state.graph_data = _gd
    st.session_state.api_error = _err
    if _gd:
        st.session_state.seed_display = str(_gd["seed"])

if st.session_state.api_error:
    st.error(st.session_state.api_error)

col1, col2 = st.columns([2, 1], gap="large")
visible_prediction = st.session_state.prediction

with col2:
    st.subheader("Controls")

    num_nodes = st.slider(
        "Node count",
        min_value=5,
        max_value=20,
        value=12,
        key="num_nodes",
        help=f"The proven training range is N={TRAIN_LO}\u2013{TRAIN_HI}. The final model passed its 90% quality gate in this range.",
    )
    seed_input = st.text_input(
        "Seed",
        value=st.session_state.seed_display,
        key="seed_input",
        placeholder="blank = random",
    )

    if st.button("Generate graph", width="stretch", help="Create a connected weighted graph for the selected size and seed."):
        raw = seed_input.strip()
        try:
            seed = int(raw) if raw else None
        except ValueError:
            st.session_state.api_error = f"Seed must be a whole number, got '{raw}'. Leave it blank for a random graph or enter an integer."
        else:
            with st.spinner("Generating a connected weighted graph..."):
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
        elif st.button("Run transformer inference", type="primary", width="stretch", help="Predict a path and compare it with Dijkstra’s exact result."):
            with st.spinner("Running inference..."):
                pred, err = predict_path(gd)
                st.session_state.prediction = pred
                st.session_state.api_error = err

        if st.session_state.prediction:
            pred = st.session_state.prediction
            st.markdown("### Inference results")
            model_path = pred.get("model_path", [])
            dijkstra_path = pred.get("dijkstra_path", [])

            if pred.get("is_optimal"):
                st.badge("MATCH", icon=":material/check_circle:", color="green")
            elif pred.get("valid_path"):
                st.badge("VALID · SUBOPTIMAL", icon=":material/warning:", color="orange")
            else:
                st.badge("INVALID ROUTE", icon=":material/error:", color="red")

            model_cost = path_cost(model_path, gd["edges"])
            if model_cost is None:
                st.markdown("**Transformer route** · cost unavailable")
            else:
                st.markdown(f"**Transformer route** · total cost **{model_cost}**")
            st.code(" → ".join(map(str, model_path)) or "No complete route", language="text")
            st.markdown(f"**Shortest route** · exact cost **{pred.get('dijkstra_cost')}**")
            st.code(" → ".join(map(str, dijkstra_path)), language="text")

            with st.expander("Show model tokens"):
                st.code(" ".join(pred.get("model_raw_tokens", [])), language="text")
    else:
        st.info("No graph loaded. Fix the error above, then click 'Generate Graph'.")

with col1:
    gd = st.session_state.graph_data
    if gd:
        prediction = st.session_state.prediction
        model_path = prediction.get("model_path", []) if prediction else []
        if len(model_path) > 1:
            edge_count = len(model_path) - 1
            replay_key = f"route_replay_{gd['seed']}_{'_'.join(map(str, model_path))}"
            st.caption("Trace the prediction · adjust the slider and watch the graph update below.")
            revealed_hops = st.slider(
                "Route progress · hops revealed",
                min_value=0,
                max_value=edge_count,
                value=edge_count,
                key=replay_key,
                help="Move left to inspect earlier route decisions; move right to reveal the complete predicted path.",
            )
            route_prefix = model_path[:revealed_hops + 1]
            visible_prediction = {"model_path": route_prefix}
            if revealed_hops == 0:
                st.caption(f"Start token · the route begins at node {model_path[0]}.")
            else:
                previous_node = model_path[revealed_hops - 1]
                next_node = model_path[revealed_hops]
                edge_costs = {
                    frozenset((u, v)): weight for u, v, weight in gd["edges"]
                }
                hop_cost = edge_costs.get(frozenset((previous_node, next_node)))
                prefix_cost = path_cost(route_prefix, gd["edges"])
                hop_label = str(hop_cost) if hop_cost is not None else "not an edge"
                prefix_label = str(prefix_cost) if prefix_cost is not None else "unavailable"
                st.caption(
                    f"Hop {revealed_hops}/{edge_count} · {previous_node} → {next_node}"
                    f" · edge cost {hop_label} · route cost so far {prefix_label}"
                )
        st.caption("Drag nodes to explore · edge labels show cost · mint = start · amber = goal · teal = revealed route")
        st.iframe(draw_graph(gd, visible_prediction), height=520, tab_index=0)
    else:
        st.warning(
            f"No graph to draw. Ensure the FastAPI backend is running at {API_BASE}."
        )


# --- FOOTER ---
