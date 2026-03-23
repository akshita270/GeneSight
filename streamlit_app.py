import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Genomics AI", page_icon="🧬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #f4f6f9; }
.block-container { padding: 2rem 3rem; }

.hero { background: linear-gradient(135deg, #0F6E56 0%, #1D9E75 60%, #5DCAA5 100%); border-radius: 16px; padding: 40px 48px; margin-bottom: 24px; }
.hero h1 { font-size: 30px; font-weight: 600; color: white; margin: 0 0 6px; }
.hero p  { font-size: 14px; color: rgba(255,255,255,0.85); margin: 0 0 20px; }
.hero-stats { display: flex; gap: 36px; }
.hs-num { font-size: 26px; font-weight: 600; color: white; }
.hs-lbl { font-size: 11px; color: rgba(255,255,255,0.7); }

.card { background: white; border-radius: 14px; padding: 24px 28px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 20px; }

.summary-card { background: #f0fdf8; border-left: 4px solid #1D9E75; border-radius: 0 12px 12px 0; padding: 18px 22px; margin-bottom: 20px; font-size: 14px; color: #1a1a1a; line-height: 1.7; }
.summary-lbl { font-size: 11px; font-weight: 600; color: #1D9E75; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }

.metric-card { background: white; border-radius: 12px; padding: 18px 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-top: 3px solid #1D9E75; text-align: center; }
.metric-num { font-size: 28px; font-weight: 600; color: #1D9E75; }
.metric-lbl { font-size: 12px; color: #888; }

.paper-card { background: white; border-radius: 12px; padding: 18px 20px; margin-bottom: 12px; box-shadow: 0 1px 6px rgba(0,0,0,0.05); border-left: 3px solid #1D9E75; }
.paper-title { font-size: 14px; font-weight: 600; color: #1a1a1a; margin-bottom: 4px; line-height: 1.45; }
.paper-meta  { font-size: 12px; color: #999; margin-bottom: 8px; }
.paper-abs   { font-size: 13px; color: #555; line-height: 1.6; }
.rel-bar     { height: 4px; border-radius: 4px; background: linear-gradient(90deg,#1D9E75,#5DCAA5); margin-top: 10px; }

.hyp-card { background: white; border-radius: 14px; padding: 22px 24px; margin-bottom: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
.hyp-title { font-size: 15px; font-weight: 600; color: #1a1a1a; margin-bottom: 8px; }
.hyp-stmt  { font-size: 13px; color: #555; line-height: 1.65; margin-bottom: 12px; }
.tag { display:inline-block; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:500; margin:2px; }
.tag-g  { background:#E1F5EE; color:#085041; }
.tag-p  { background:#EEEDFE; color:#3C3489; }
.tag-e  { background:#FFF8E1; color:#633806; }
.badge  { display:inline-block; padding:3px 12px; border-radius:20px; font-size:11px; font-weight:600; }
.b-str  { background:#E1F5EE; color:#085041; }
.b-mod  { background:#FAEEDA; color:#633806; }
.b-exp  { background:#EEEDFE; color:#3C3489; }
.conf-bar { height:6px; border-radius:4px; }

.node-chip { display:inline-block; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:500; margin:3px; }
.nc-gene    { background:#E1F5EE; color:#085041; border:1px solid #A5D6C0; }
.nc-disease { background:#FAECE7; color:#993C1D; border:1px solid #F0997B; }
.nc-protein { background:#EEEDFE; color:#3C3489; border:1px solid #AFA9EC; }

div.stButton > button { background:#1D9E75 !important; color:white !important; border-radius:10px !important; border:none !important; padding:10px 28px !important; font-weight:600 !important; font-size:14px !important; width:100%; }
div.stButton > button:hover { background:#0F6E56 !important; }
.stTextInput input { border-radius:10px !important; border:1.5px solid #e0e0e0 !important; padding:12px 16px !important; font-size:14px !important; }
.stTextInput input:focus { border-color:#1D9E75 !important; }
</style>
""", unsafe_allow_html=True)

AGENTS = ["Task Planner","Literature Retrieval","Info Extraction",
          "Genomics DB","Knowledge Graph","Hypothesis Generation",
          "Evidence Validation","Report Generation"]

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧬 Genomics AI Research Assistant</h1>
  <p>Automated hypothesis generation · Multi-agent pipeline · PubMed + NCBI + UniProt + Neo4j</p>
  <div class="hero-stats">
    <div><div class="hs-num">8</div><div class="hs-lbl">AI Agents</div></div>
    <div><div class="hs-num">4</div><div class="hs-lbl">Data Sources</div></div>
    <div><div class="hs-num">GPT-4o</div><div class="hs-lbl">LLM Model</div></div>
    <div><div class="hs-num">Neo4j</div><div class="hs-lbl">Knowledge Graph</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Query Input ───────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 🔍 Research Query")
query = st.text_input("", placeholder="e.g. Identify potential new gene relationships related to Alzheimer's disease", label_visibility="collapsed")
st.markdown("**Try an example:**")
c1, c2, c3 = st.columns(3)
if c1.button("🧠 Alzheimer's gene relationships"):
    query = "Identify potential new gene relationships related to Alzheimer's disease"
if c2.button("🎗️ BRCA1 cancer pathways"):
    query = "Analyze BRCA1 gene relationships in breast cancer pathways"
if c3.button("🩸 Type 2 diabetes immune genes"):
    query = "Identify immune gene relationships in Type 2 diabetes"
_, bc, _ = st.columns([2,1,2])
run_btn = bc.button("🔬 Run Pipeline")
st.markdown('</div>', unsafe_allow_html=True)

# ── Pipeline Runner ───────────────────────────────────────
def run_pipeline(query):
    try:
        r = requests.post(f"{API_URL}/query", json={"query": query}, timeout=30)
        r.raise_for_status()
        job_id = r.json()["job_id"]
    except Exception as e:
        st.error(f"❌ Failed to start pipeline: {e}")
        return None

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### ⚙️ Pipeline Status")
    prog     = st.progress(0)
    stat_txt = st.empty()
    chips    = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    while True:
        try:
            s = requests.get(f"{API_URL}/status/{job_id}", timeout=10).json()
            agent  = s.get("current_agent", "")
            status = s.get("status", "")
            idx    = AGENTS.index(agent) if agent in AGENTS else 0

            prog.progress((idx + 1) / len(AGENTS))
            stat_txt.markdown(f"**Running agent:** `{agent}`")

            chip_html = "".join([
                f'<span class="tag" style="background:{"#1D9E75" if i==idx else "#E1F5EE" if i<idx else "#f5f5f5"};'
                f'color:{"white" if i==idx else "#085041" if i<idx else "#aaa"};margin:3px;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:500">'
                f'{"▶ " if i==idx else "✓ " if i<idx else ""}{a}</span>'
                for i, a in enumerate(AGENTS)
            ])
            chips.markdown(chip_html, unsafe_allow_html=True)

            if status == "done":
                prog.progress(1.0)
                stat_txt.markdown("**✅ Pipeline complete!**")
                break
            elif status == "error":
                st.error(f"Pipeline error: {s.get('error','Unknown')}")
                return None
            time.sleep(2)
        except Exception as e:
            st.error(f"Polling error: {e}")
            return None

    try:
        res = requests.get(f"{API_URL}/result/{job_id}", timeout=30)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch result: {e}")
        return None

# ── Results ───────────────────────────────────────────────
if run_btn and query:
    result = run_pipeline(query)

    if result:
        papers     = result.get("papers", [])
        hyps       = result.get("hypotheses", [])
        graph      = result.get("graph", {})
        nodes      = graph.get("nodes", [])
        edges      = graph.get("edges", [])

        # Summary
        if result.get("summary"):
            st.markdown(f'<div class="summary-card"><div class="summary-lbl">📋 Research Summary</div>{result["summary"]}</div>', unsafe_allow_html=True)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-num">{len(papers)}</div><div class="metric-lbl">Papers Retrieved</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-num">{len(hyps)}</div><div class="metric-lbl">Hypotheses Generated</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-num">{len(nodes)}</div><div class="metric-lbl">Graph Nodes</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-num">{len(edges)}</div><div class="metric-lbl">Relationships</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs
        tab1, tab2, tab3 = st.tabs([f"📄 Literature ({len(papers)})", "🔗 Knowledge Graph", f"💡 Hypotheses ({len(hyps)})"])

        # Literature
        with tab1:
            if not papers:
                st.warning("No papers retrieved.")
            for p in papers:
                st.markdown(f"""
                <div class="paper-card">
                  <div class="paper-title">{p.get('title','N/A')}</div>
                  <div class="paper-meta">
                    {', '.join(p.get('authors',[])[:3])} &nbsp;·&nbsp;
                    <b>{p.get('journal','')}</b> &nbsp;·&nbsp;
                    {p.get('year','')} &nbsp;·&nbsp;
                    PMID: <code>{p.get('pmid','')}</code>
                  </div>
                  <div class="paper-abs">{str(p.get('abstract',''))[:350]}...</div>
                  <div class="rel-bar" style="width:{int(p.get('relevance_score',0.9)*100)}%"></div>
                </div>""", unsafe_allow_html=True)

        # Knowledge Graph
        with tab2:
            if not nodes:
                st.warning("No graph data available.")
            else:
                left, right = st.columns([1, 2])
                with left:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("**📊 Graph Stats**")
                    genes_n    = [n for n in nodes if n.get("type") == "gene"]
                    diseases_n = [n for n in nodes if n.get("type") == "disease"]
                    proteins_n = [n for n in nodes if n.get("type") == "protein"]
                    st.metric("Genes",     len(genes_n))
                    st.metric("Diseases",  len(diseases_n))
                    st.metric("Proteins",  len(proteins_n))
                    st.metric("Edges",     len(edges))
                    st.markdown('</div>', unsafe_allow_html=True)

                with right:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("**🧬 Entities**")
                    chips_html = ""
                    for n in nodes:
                        cls = {"gene":"nc-gene","disease":"nc-disease","protein":"nc-protein"}.get(n.get("type",""),"nc-gene")
                        chips_html += f'<span class="node-chip {cls}">{n.get("label", n.get("id",""))}</span>'
                    st.markdown(chips_html, unsafe_allow_html=True)

                    st.markdown("<br>**🔗 Relationships**", unsafe_allow_html=True)
                    for e in edges[:12]:
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid #f5f5f5;font-size:13px">
                          <span class="node-chip nc-gene" style="margin:0">{e.get('source','')}</span>
                          <span style="color:#1D9E75;font-weight:600">→ {e.get('relation','')} →</span>
                          <span class="node-chip nc-disease" style="margin:0">{e.get('target','')}</span>
                        </div>""", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        # Hypotheses
        with tab3:
            if not hyps:
                st.warning("No hypotheses generated.")
            for h in hyps:
                status = h.get("status","Exploratory")
                badge_cls = {"Strong":"b-str","Moderate":"b-mod","Exploratory":"b-exp"}.get(status,"b-exp")
                conf = int(h.get("confidence", 50))
                conf_color = "#1D9E75" if conf >= 80 else "#BA7517" if conf >= 60 else "#993C1D"
                genes_tags = "".join(f'<span class="tag tag-g">{g}</span>' for g in h.get("genes",[]))
                pathway_tag = f'<span class="tag tag-p">{h.get("pathway","")}</span>'
                evidence_tag = f'<span class="tag tag-e">📄 {h.get("evidence_count",0)} papers</span>'

                st.markdown(f"""
                <div class="hyp-card">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap">
                    <span class="badge {badge_cls}">{status}</span>
                    <span style="background:#f5f5f5;padding:3px 12px;border-radius:20px;font-size:11px;font-weight:600;color:{conf_color}">{conf}% confidence</span>
                  </div>
                  <div class="hyp-title">{h.get('title','')}</div>
                  <div class="hyp-stmt">{h.get('statement','')}</div>
                  <div style="height:6px;border-radius:4px;background:#f0f0f0;margin-bottom:12px">
                    <div class="conf-bar" style="width:{conf}%;background:{conf_color}"></div>
                  </div>
                  <div>{genes_tags}{pathway_tag}{evidence_tag}</div>
                </div>""", unsafe_allow_html=True)

elif run_btn and not query:
    st.warning("⚠️ Please enter a research query first.")