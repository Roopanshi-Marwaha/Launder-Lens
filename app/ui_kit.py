"""Launder Lens design system: 'forensic ledger' - ink-black canvas, bone-white type, one signal colour."""
import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

ASSETS = Path(__file__).parent / "assets"
C = dict(ink="#070a0f", panel="#0e141c", panel2="#131b26", line="rgba(255,255,255,.08)", text="#e8e6df",
         muted="#8a94a3", signal="#ff5a3c", amber="#ffb347", teal="#3dd6c0", steel="#6f9bff", violet="#b58cff")
MODEL_STYLE = {   # name -> (label, colour)
    "rules": ("Rule baseline", "#8a94a3"), "rf_all": ("Random Forest", "#7d8fb3"),
    "xgb_amount": ("Amount-only XGBoost", C["amber"]), "xgb_structure": ("Structure-first XGBoost", C["teal"]),
    "xgb_all": ("Hybrid XGBoost", C["steel"]), "gnn_structure": ("Graph NN (structure)", C["violet"]),
    "gnn_all": ("Graph NN (all)", "#d6b8ff"), "xgb_all_hardened": ("Hybrid - hardened", "#9bb8ff"),
    "xgb_structure_hardened": ("Structure-first - hardened", "#7af0dc"),
    "gnn_structure_hardened": ("Graph NN - hardened", "#d9c2ff"),
    "struct": ("Structure-first", C["teal"]), "gnn": ("Graph NN", C["violet"]), "all": ("Hybrid (amounts)", C["steel"]),
    "hybrid_naive": ("Hybrid (naive training)", C["steel"]), "hybrid_hardened": ("Hybrid (hardened)", "#9bb8ff"),
    "structure_hardened": ("Structure-first (hardened)", C["teal"]),
}


def label(m):
    return MODEL_STYLE.get(m, (m, "#aaa"))[0]


def color(m):
    return MODEL_STYLE.get(m, (m, "#aaa"))[1]


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{--ink:#070a0f;--panel:#0e141c;--panel2:#131b26;--line:rgba(255,255,255,.08);--text:#e8e6df;--muted:#8a94a3;
--signal:#ff5a3c;--amber:#ffb347;--teal:#3dd6c0;--steel:#6f9bff;--violet:#b58cff;
--serif:'Fraunces',Georgia,'Times New Roman',serif;--sans:'IBM Plex Sans',system-ui,-apple-system,Segoe UI,sans-serif;--mono:'IBM Plex Mono',ui-monospace,Menlo,Consolas,monospace}
html,body,[class*="css"],.stApp{font-family:var(--sans)!important;color:var(--text)}
.stApp{background:radial-gradient(1200px 600px at 85% -10%,rgba(255,90,60,.10),transparent 60%),radial-gradient(900px 500px at -10% 10%,rgba(61,214,192,.07),transparent 60%),var(--ink)}
.stApp:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.35;
background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);background-size:48px 48px;
-webkit-mask-image:radial-gradient(ellipse at 50% 0%,#000 30%,transparent 75%);mask-image:radial-gradient(ellipse at 50% 0%,#000 30%,transparent 75%)}
header[data-testid="stHeader"]{background:transparent}#MainMenu,footer,[data-testid="stToolbar"]{visibility:hidden}
.block-container{padding-top:1.6rem;padding-bottom:4rem;max-width:1320px}
h1,h2,h3{font-family:var(--serif)!important;letter-spacing:-.01em;font-weight:600}
h3{font-size:1.35rem!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a0f16,#070a0f);border-right:1px solid var(--line)}
[data-testid="stSidebar"] .block-container{padding-top:1.2rem}
.brand{display:flex;align-items:center;gap:.7rem;margin:.2rem 0 1.4rem}
.brand .mark{width:38px;height:38px;border-radius:50%;border:2px solid var(--signal);position:relative;box-shadow:0 0 18px rgba(255,90,60,.45)}
.brand .mark:after{content:"";position:absolute;width:14px;height:3px;background:var(--signal);right:-10px;bottom:-3px;transform:rotate(45deg);border-radius:2px}
.brand .mark:before{content:"";position:absolute;inset:8px;border-radius:50%;background:radial-gradient(circle,var(--signal),transparent 70%);opacity:.8}
.brand .name{font-family:var(--serif);font-size:1.25rem;font-weight:800;line-height:1}.brand .tag{font:500 .62rem var(--mono);letter-spacing:.18em;color:var(--muted);text-transform:uppercase;margin-top:.25rem}
[data-testid="stSidebar"] div[role="radiogroup"]{gap:.15rem}
[data-testid="stSidebar"] div[role="radiogroup"]>label{padding:.62rem .85rem;border-radius:10px;border:1px solid transparent;transition:.15s;cursor:pointer;width:100%}
[data-testid="stSidebar"] div[role="radiogroup"]>label>div:first-child{display:none}
[data-testid="stSidebar"] div[role="radiogroup"]>label:hover{background:rgba(255,255,255,.04)}
[data-testid="stSidebar"] div[role="radiogroup"]>label:has(input:checked){background:linear-gradient(90deg,rgba(255,90,60,.16),rgba(255,90,60,.02));border-color:rgba(255,90,60,.35)}
[data-testid="stSidebar"] div[role="radiogroup"] p{font:500 .92rem var(--sans)!important}
.side-note{font:400 .72rem/1.5 var(--mono);color:var(--muted);border-top:1px solid var(--line);padding-top:.9rem;margin-top:1.4rem}
.kicker{font:500 .7rem var(--mono);letter-spacing:.22em;text-transform:uppercase;color:var(--signal)}
.hero{position:relative;border:1px solid var(--line);border-radius:22px;overflow:hidden;background:linear-gradient(120deg,rgba(14,20,28,.95),rgba(14,20,28,.55));padding:2.3rem 2.4rem 2rem;margin-bottom:1.2rem}
.hero h1{font-size:clamp(2rem,3.6vw,3.35rem)!important;line-height:1.04;margin:.5rem 0 .9rem;max-width:760px;font-weight:800}
.hero h1 em{font-style:italic;color:var(--signal);font-weight:600}
.hero p{max-width:640px;color:#b9c0cb;font-size:1.02rem;line-height:1.6}
.pagehead{margin:.2rem 0 1.2rem}.pagehead h1{font-size:2.2rem!important;margin:.25rem 0 .35rem}.pagehead p{color:#b0b8c4;max-width:760px;margin:0;line-height:1.55}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.9rem;margin:.6rem 0 1.2rem}
.kpi{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem;position:relative;overflow:hidden}
.kpi:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--accent,var(--steel))}
.kpi .l{font:500 .68rem var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.kpi .v{font:800 2rem var(--serif);margin:.3rem 0 .1rem;color:var(--accent,var(--text))}
.kpi .s{font-size:.8rem;color:#98a2b0;line-height:1.4}
.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.25rem;height:100%}
.card h4{font:600 1.05rem var(--serif);margin:0 0 .45rem}.card p,.card li{color:#aeb6c2;font-size:.9rem;line-height:1.55}
.act{display:flex;gap:.9rem}.act .n{font:800 2.4rem var(--serif);color:var(--signal);opacity:.9;line-height:1}
.chip{display:inline-block;font:500 .7rem var(--mono);letter-spacing:.06em;padding:.22rem .6rem;border-radius:99px;border:1px solid var(--line);color:#c4cbd6;margin:0 .3rem .3rem 0;background:rgba(255,255,255,.03)}
.chip.red{color:#ffb4a5;border-color:rgba(255,90,60,.5);background:rgba(255,90,60,.1)}.chip.teal{color:#8af0e0;border-color:rgba(61,214,192,.45);background:rgba(61,214,192,.08)}.chip.amber{color:#ffd599;border-color:rgba(255,179,71,.45);background:rgba(255,179,71,.08)}
.flow{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;margin:.4rem 0 1rem}.flow .st{padding:.5rem .9rem;border:1px solid var(--line);border-radius:10px;background:var(--panel);font:600 .85rem var(--sans)}.flow .ar{color:var(--signal);font-family:var(--mono)}
.legend{display:flex;flex-wrap:wrap;gap:1rem;font:400 .75rem var(--mono);color:#aab2be;margin:.3rem 0 .6rem}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:.4rem;vertical-align:-1px}
.insight{border-left:3px solid var(--signal);background:rgba(255,90,60,.06);padding:.8rem 1rem;border-radius:0 12px 12px 0;color:#d6dae1;font-size:.93rem;line-height:1.55;margin:.6rem 0 1rem}
.case{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:1.1rem 1.3rem;line-height:1.65;font-size:.95rem}
.stButton>button,.stFormSubmitButton>button{border-radius:11px;border:1px solid rgba(255,90,60,.55);background:linear-gradient(180deg,rgba(255,90,60,.25),rgba(255,90,60,.1));color:#fff;font-weight:600;transition:.15s}
.stButton>button:hover,.stFormSubmitButton>button:hover{border-color:var(--signal);box-shadow:0 0 22px rgba(255,90,60,.35);transform:translateY(-1px)}
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:.8rem 1rem}
.stTabs [data-baseweb="tab-list"]{gap:.3rem;border-bottom:1px solid var(--line)}.stTabs [data-baseweb="tab"]{padding:.55rem 1rem;border-radius:10px 10px 0 0;font-weight:500}
.stTabs [aria-selected="true"]{background:rgba(255,90,60,.1)}
div[data-testid="stForm"]{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.2rem}
.stDataFrame{border:1px solid var(--line);border-radius:12px;overflow:hidden}
hr{border-color:var(--line)}
@keyframes pulse{0%,100%{opacity:.55}50%{opacity:1}}.live{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--signal);margin-right:.5rem;animation:pulse 1.6s infinite;box-shadow:0 0 10px var(--signal)}
</style>
"""


CSS_FX = r"""
<style>
@property --ang{syntax:'<angle>';initial-value:0deg;inherits:false}
/* film grain + soft vignette */
.stApp:after{content:"";position:fixed;inset:0;pointer-events:none;z-index:1;opacity:.07;mix-blend-mode:overlay;
background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>")}
::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-track{background:var(--ink)}::-webkit-scrollbar-thumb{background:#1c2633;border-radius:8px;border:2px solid var(--ink)}::-webkit-scrollbar-thumb:hover{background:var(--signal)}
::selection{background:var(--signal);color:#fff}
/* staggered entrance */
@keyframes rise{from{opacity:0;transform:translateY(18px);filter:blur(6px)}to{opacity:1;transform:none;filter:none}}
.pagehead,.hero,.insight,.case,.card,.flow,.legend{animation:rise .75s cubic-bezier(.2,.7,.2,1) backwards}
.kpi{animation:rise .75s cubic-bezier(.2,.7,.2,1) backwards}
.kpi:nth-child(2){animation-delay:.08s}.kpi:nth-child(3){animation-delay:.16s}.kpi:nth-child(4){animation-delay:.24s}.kpi:nth-child(5){animation-delay:.32s}
/* hero: rotating border light + scanner sweep + glitch */
.hero:after{content:"";position:absolute;inset:0;border-radius:inherit;padding:1.5px;pointer-events:none;
background:conic-gradient(from var(--ang),transparent 0 62%,var(--signal) 78%,var(--teal) 92%,transparent);
-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude;animation:spin 7s linear infinite}
@keyframes spin{to{--ang:360deg}}
.hero:before{content:"";position:absolute;left:0;right:0;height:140px;top:-140px;pointer-events:none;background:linear-gradient(180deg,transparent,rgba(61,214,192,.07),transparent);animation:scan 8s linear infinite}
@keyframes scan{to{top:110%}}
.hero h1 em{display:inline-block;animation:glitch 6s infinite}
@keyframes glitch{0%,91%,100%{text-shadow:none;transform:none}92%{text-shadow:-3px 0 var(--teal),3px 0 var(--steel);transform:translateX(2px) skewX(-4deg)}94%{text-shadow:3px 0 var(--teal),-3px 0 var(--steel);transform:translateX(-2px)}96%{text-shadow:-2px 0 var(--amber);transform:none}}
/* cards: tilt + cursor spotlight */
.kpi,.card,.case{position:relative;transform:perspective(800px) rotateX(var(--rx,0deg)) rotateY(var(--ry,0deg));transition:transform .18s ease-out,border-color .25s,box-shadow .25s}
.kpi:after,.card:after,.case:after{content:"";position:absolute;inset:0;pointer-events:none;border-radius:inherit;opacity:0;transition:opacity .25s;background:radial-gradient(280px circle at var(--mx,50%) var(--my,50%),rgba(255,255,255,.08),transparent 60%)}
.kpi:hover:after,.card:hover:after,.case:hover:after{opacity:1}
.kpi:hover,.card:hover{border-color:rgba(255,90,60,.4);box-shadow:0 14px 40px -18px rgba(255,90,60,.55)}
.kpi .v{text-shadow:0 0 26px color-mix(in srgb,var(--accent,#6f9bff) 45%,transparent)}
/* buttons: shine sweep */
.stButton>button,.stFormSubmitButton>button{position:relative;overflow:hidden}
.stButton>button:after,.stFormSubmitButton>button:after{content:"";position:absolute;top:0;bottom:0;width:60%;left:-80%;background:linear-gradient(100deg,transparent,rgba(255,255,255,.28),transparent);transform:skewX(-20deg);transition:left .6s}
.stButton>button:hover:after,.stFormSubmitButton>button:hover:after{left:130%}
.stButton>button:active{transform:scale(.97)}
/* sidebar nav */
[data-testid="stSidebar"] div[role="radiogroup"]>label:hover{transform:translateX(4px)}
[data-testid="stSidebar"] div[role="radiogroup"]>label:has(input:checked){box-shadow:inset 3px 0 0 var(--signal),0 0 24px -8px rgba(255,90,60,.6)}
/* cursor glow */
.ll-orb{position:fixed;top:0;left:0;width:320px;height:320px;border-radius:50%;pointer-events:none;z-index:2;opacity:0;transition:opacity .5s;mix-blend-mode:screen;background:radial-gradient(circle,rgba(255,90,60,.14),rgba(61,214,192,.05) 45%,transparent 68%)}
.element-container:has(iframe[height="0"]){position:absolute;height:0;overflow:hidden;margin:0}
/* transaction tape */
.tape{position:fixed;left:0;right:0;bottom:0;height:30px;z-index:60;display:flex;flex-direction:row-reverse;align-items:center;overflow:hidden;pointer-events:none;border-top:1px solid var(--line);background:rgba(7,10,15,.88);backdrop-filter:blur(8px);font:500 .68rem var(--mono);color:var(--muted);letter-spacing:.05em}
.tape .lab{flex:none;height:100%;display:flex;align-items:center;gap:.45rem;padding:0 .9rem;border-left:1px solid var(--line);background:var(--ink);color:var(--signal);text-transform:uppercase;letter-spacing:.16em}
.tape-in{display:flex;gap:2.4rem;white-space:nowrap;padding-left:2rem;animation:tape 80s linear infinite}
@keyframes tape{to{transform:translateX(-50%)}}
.tape b.f{color:var(--signal)}.tape b.o{color:var(--teal)}
/* responsive */
@media (max-width:900px){
.block-container{padding:1rem .9rem 4.5rem!important}
.hero{padding:1.4rem 1.2rem 1.3rem;border-radius:16px}.hero p{font-size:.95rem}
.pagehead h1{font-size:1.7rem!important}
.kpis{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}.kpi .v{font-size:1.6rem}
.stTabs [data-baseweb="tab-list"]{overflow-x:auto;flex-wrap:nowrap}.stTabs [data-baseweb="tab"]{white-space:nowrap}
.flow .st{font-size:.78rem;padding:.4rem .65rem}}
@media (max-width:640px){
.kpis{grid-template-columns:1fr 1fr}.kpi{padding:.8rem .85rem}.kpi .v{font-size:1.4rem}
.act{flex-direction:column;gap:.3rem}.tape{display:none}
div[data-testid="stForm"]{padding:.8rem}}
@media (pointer:coarse){.kpi,.card,.case{transform:none!important}.ll-orb{display:none}}
@media (prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
</style>
"""


CSS_NAV = r"""
<style>
/* sidebar navigation: works with both the old and the new Streamlit radio markup */
[data-testid="stSidebar"] [role="radiogroup"]{gap:.25rem!important}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]{width:100%;padding:.62rem .9rem;border-radius:12px;border:1px solid transparent;
transition:background .15s,border-color .15s,transform .15s,box-shadow .15s;cursor:pointer;margin:0}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]>div>div:first-child{display:none!important}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover{background:rgba(255,255,255,.05);transform:translateX(4px)}
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"]{background:linear-gradient(90deg,rgba(255,90,60,.20),rgba(255,90,60,.03));
border-color:rgba(255,90,60,.4);box-shadow:inset 3px 0 0 var(--signal),0 0 24px -8px rgba(255,90,60,.6)}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] p{font:500 .95rem var(--sans)!important;margin:0}
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p{color:#fff;font-weight:600!important}
/* keep the "open menu" button visible (hiding the toolbar used to hide it too, trapping mobile users) */
[data-testid="stExpandSidebarButton"],[data-testid="collapsedControl"]{visibility:visible!important;opacity:1!important;z-index:100}
[data-testid="stExpandSidebarButton"],[data-testid="collapsedControl"] button{background:rgba(14,20,28,.92);border:1px solid rgba(255,90,60,.5);
border-radius:12px;color:#fff;box-shadow:0 0 18px -4px rgba(255,90,60,.55);backdrop-filter:blur(6px)}
[data-testid="stExpandSidebarButton"]:hover{border-color:var(--signal);box-shadow:0 0 24px rgba(255,90,60,.5)}
/* system status panel */
.sysmon{margin-top:1.3rem;border:1px solid var(--line);border-radius:14px;background:linear-gradient(180deg,var(--panel2),var(--panel));padding:.8rem .95rem}
.sysmon .h{font:500 .64rem var(--mono);letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-bottom:.55rem}
.sysmon .r{display:flex;align-items:center;justify-content:space-between;gap:.6rem;font:400 .78rem var(--sans);color:#c4cbd6;padding:.22rem 0}
.sysmon .r b{font:500 .68rem var(--mono);letter-spacing:.06em;padding:.12rem .5rem;border-radius:99px}
.sysmon .ok b{color:#8af0e0;background:rgba(61,214,192,.1);border:1px solid rgba(61,214,192,.4)}
.sysmon .no b{color:#ffb4a5;background:rgba(255,90,60,.1);border:1px solid rgba(255,90,60,.5)}
.sysmon .dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:.5rem;background:var(--teal);box-shadow:0 0 8px var(--teal)}
.sysmon .no .dot{background:var(--signal);box-shadow:0 0 8px var(--signal)}
/* small screens: tighter hero canvas / tables / buttons */
@media (max-width:900px){
[data-testid="stSidebar"] label[data-testid="stRadioOption"]{padding:.8rem .9rem}
.stButton>button{width:100%;min-height:2.8rem}
.legend{gap:.6rem .9rem}
.insight{font-size:.88rem}}
@media (max-width:640px){
h1{word-break:break-word}
.hero h1{font-size:1.9rem!important}
.kpi .l{font-size:.6rem}
.sysmon{margin-top:1rem}}
</style>
"""


def inject_css():
    st.markdown(CSS + CSS_FX + CSS_NAV, unsafe_allow_html=True)
    fx()
    tape()


def brand():
    st.markdown('<div class="brand"><div class="mark"></div><div><div class="name">Launder Lens</div>'
                '<div class="tag">Financial crime lab 2.0</div></div></div>', unsafe_allow_html=True)


def status_panel(items):
    """items: [(label, ok, detail)] -> small 'system status' card for the sidebar."""
    rows = "".join(f'<div class="r {"ok" if ok else "no"}"><span><span class="dot"></span>{l}</span><b>{d}</b></div>'
                   for l, ok, d in items)
    st.markdown(f'<div class="sysmon"><div class="h">System status</div>{rows}</div>', unsafe_allow_html=True)


def page_head(kicker, title, sub):
    st.markdown(f'<div class="pagehead"><div class="kicker">{kicker}</div><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def kpis(items):
    """items: [(label, value, sub, accent)]"""
    html = "".join(f'<div class="kpi" style="--accent:{a or C["steel"]}"><div class="l">{l}</div><div class="v">{v}</div>'
                   f'<div class="s">{s}</div></div>' for l, v, s, a in items)
    st.markdown(f'<div class="kpis">{html}</div>', unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def legend(items):
    st.markdown('<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{t}</span>' for t, c in items) + "</div>",
                unsafe_allow_html=True)


def plot(fig, h=380, key=None):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Sans, system-ui, sans-serif", color="#c9d1d9", size=12),
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2),
                      title_font=dict(family="Fraunces, Georgia, serif", size=16, color="#e8e6df"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.1)")
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def embed(html, height):
    """st.iframe on new Streamlit, components.html on older versions."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


_VIS = None


def _vis_js():
    global _VIS
    if _VIS is None:
        _VIS = (ASSETS / "vis-network.min.js").read_text(encoding="utf-8")
    return _VIS


def network(nodes, edges, height=560, hierarchical=False, physics=True):
    """Self-contained interactive network (vis-network bundled offline)."""
    opts = {
        "nodes": {"shape": "dot", "font": {"color": "#c9d1d9", "size": 11, "face": "IBM Plex Mono, monospace", "strokeWidth": 3,
                                            "strokeColor": "#070a0f"}, "borderWidth": 1.5},
        "edges": {"smooth": {"type": "continuous"}, "arrows": {"to": {"enabled": True, "scaleFactor": .55}}, "selectionWidth": 1.5},
        "physics": {"enabled": physics and not hierarchical, "barnesHut": {"gravitationalConstant": -4200, "springLength": 95,
                    "springConstant": .035, "damping": .35}, "stabilization": {"iterations": 160}},
        "interaction": {"hover": True, "tooltipDelay": 80, "navigationButtons": False, "zoomView": True},
    }
    if hierarchical:
        opts["layout"] = {"hierarchical": {"enabled": True, "direction": "LR", "sortMethod": "directed", "levelSeparation": 150, "nodeSpacing": 70}}
        opts["edges"]["smooth"] = {"type": "cubicBezier", "forceDirection": "horizontal"}
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;height:100%;background:transparent}}
#g{{height:{height - 4}px;border:1px solid rgba(255,255,255,.09);border-radius:16px;background:radial-gradient(circle at 50% 40%,#101826,#080c12 75%)}}
#tip{{position:absolute;right:14px;bottom:10px;font:11px monospace;color:#6c7685}}</style></head><body><div id="g"></div>
<div id="tip">scroll to zoom - drag to explore - hover a node</div><script>{_vis_js()}</script><script>
const nodes=new vis.DataSet({json.dumps(nodes)});const edges=new vis.DataSet({json.dumps(edges)});
const net=new vis.Network(document.getElementById('g'),{{nodes,edges}},{json.dumps(opts)});
net.once('stabilizationIterationsDone',()=>{{net.setOptions({{physics:{{enabled:false}}}});net.fit({{animation:true}})}});
</script></body></html>"""
    embed(html, height)


def hero_canvas(height=330):
    """Decorative animated 'money moving through a network' canvas for the landing hero."""
    html = """<!doctype html><html><body style="margin:0;background:transparent;overflow:hidden"><canvas id="c"></canvas><script>
const cv=document.getElementById('c'),x=cv.getContext('2d');let W,H,dpr=window.devicePixelRatio||1;
function rs(){W=cv.clientWidth=innerWidth;H=innerHeight;cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+'px';cv.style.height=H+'px';x.setTransform(dpr,0,0,dpr,0,0)}rs();onresize=rs;
const N=80,P=[];for(let i=0;i<N;i++)P.push({x:Math.random(),y:Math.random(),vx:(Math.random()-.5)*.00018,vy:(Math.random()-.5)*.00018});
const E=[];for(let i=0;i<N;i++){const d=P.map((q,j)=>[j,(q.x-P[i].x)**2+(q.y-P[i].y)**2]).filter(a=>a[0]!=i).sort((a,b)=>a[1]-b[1]).slice(0,2);d.forEach(a=>E.push([i,a[0]]))}
const R=7;let t=0;const parts=[];for(let i=0;i<R;i++)parts.push({k:i,u:Math.random()});
function ring(i,tt){const a=tt*.12+i/R*Math.PI*2;return[.64+.12*Math.cos(a)*(H/W)*1.9,.5+.3*Math.sin(a)]}
function frame(){t+=.016;x.clearRect(0,0,W,H);
P.forEach(p=>{p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>1)p.vx*=-1;if(p.y<0||p.y>1)p.vy*=-1});
x.lineWidth=1;E.forEach(e=>{const a=P[e[0]],b=P[e[1]];x.strokeStyle='rgba(140,155,175,.13)';x.beginPath();x.moveTo(a.x*W,a.y*H);x.lineTo(b.x*W,b.y*H);x.stroke()});
P.forEach(p=>{x.fillStyle='rgba(150,165,185,.45)';x.beginPath();x.arc(p.x*W,p.y*H,2.2,0,7);x.fill()});
const L=[];for(let i=0;i<R;i++)L.push(ring(i,t));
const lx=(.2+.65*(.5+.5*Math.sin(t*.35)))*W,ly=H*.5,lr=Math.min(H*.42,150);
x.strokeStyle='rgba(255,90,60,.55)';x.lineWidth=1.5;for(let i=0;i<R;i++){const a=L[i],b=L[(i+1)%R];x.beginPath();x.moveTo(a[0]*W,a[1]*H);x.lineTo(b[0]*W,b[1]*H);x.stroke()}
parts.forEach(p=>{p.u+=.006;if(p.u>1){p.u=0}const a=L[p.k],b=L[(p.k+1)%R];const px=(a[0]+(b[0]-a[0])*p.u)*W,py=(a[1]+(b[1]-a[1])*p.u)*H;
const g=x.createRadialGradient(px,py,0,px,py,9);g.addColorStop(0,'rgba(255,200,170,1)');g.addColorStop(1,'rgba(255,90,60,0)');x.fillStyle=g;x.beginPath();x.arc(px,py,9,0,7);x.fill()});
L.forEach(a=>{const g=x.createRadialGradient(a[0]*W,a[1]*H,0,a[0]*W,a[1]*H,16);g.addColorStop(0,'rgba(255,90,60,.9)');g.addColorStop(1,'rgba(255,90,60,0)');x.fillStyle=g;x.beginPath();x.arc(a[0]*W,a[1]*H,16,0,7);x.fill();x.fillStyle='#ff5a3c';x.beginPath();x.arc(a[0]*W,a[1]*H,4,0,7);x.fill()});
x.strokeStyle='rgba(61,214,192,.75)';x.lineWidth=2;x.beginPath();x.arc(lx,ly,lr,0,7);x.stroke();x.strokeStyle='rgba(61,214,192,.25)';x.lineWidth=8;x.beginPath();x.arc(lx,ly,lr,0,7);x.stroke();
x.strokeStyle='rgba(61,214,192,.75)';x.lineWidth=5;x.lineCap='round';x.beginPath();x.moveTo(lx+lr*.72,ly+lr*.72);x.lineTo(lx+lr*1.12,ly+lr*1.12);x.stroke();
requestAnimationFrame(frame)}frame();</script></body></html>"""
    embed(html, height)


FX_JS = r"""<script>
(function(){var P;try{P=window.parent;void P.document.body}catch(e){return}
if(P.__llfx)return;P.__llfx=1;
var D=P.document,fine=P.matchMedia('(pointer:fine)').matches,calm=P.matchMedia('(prefers-reduced-motion:reduce)').matches;
if(fine&&!calm){var o=D.createElement('div');o.className='ll-orb';D.body.appendChild(o);var tx=0,ty=0,cx=0,cy=0;
D.addEventListener('mousemove',function(e){tx=e.clientX;ty=e.clientY;o.style.opacity=1;
var el=e.target.closest&&e.target.closest('.kpi,.card,.case');if(!el)return;var r=el.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
el.style.setProperty('--mx',x+'px');el.style.setProperty('--my',y+'px');
if(el.classList.contains('kpi')){el.style.setProperty('--ry',((x/r.width-.5)*10)+'deg');el.style.setProperty('--rx',((.5-y/r.height)*10)+'deg')}});
D.addEventListener('mouseout',function(e){var el=e.target.closest&&e.target.closest('.kpi');
if(el&&!el.contains(e.relatedTarget)){el.style.setProperty('--rx','0deg');el.style.setProperty('--ry','0deg')}});
(function loop(){cx+=(tx-cx)*.12;cy+=(ty-cy)*.12;o.style.transform='translate('+(cx-160)+'px,'+(cy-160)+'px)';P.requestAnimationFrame(loop)})()}
function ease(t){return 1-Math.pow(1-t,3)}
function count(el){if(calm)return;var m=el.textContent.trim().match(/^([^\d-]*)(-?[\d,]*\.?\d+)([\s\S]*)$/);if(!m)return;
var raw=m[2],end=parseFloat(raw.replace(/,/g,'')),dec=(raw.split('.')[1]||'').length,comma=raw.indexOf(',')>-1,pre=m[1],suf=m[3],t0=null;
function fmt(v){var s=v.toFixed(dec);if(comma){var p=s.split('.');p[0]=p[0].replace(/\B(?=(\d{3})+(?!\d))/g,',');s=p.join('.')}return pre+s+suf}
function step(ts){if(t0===null)t0=ts;var k=Math.min((ts-t0)/900,1);el.textContent=fmt(end*ease(k));if(k<1)P.requestAnimationFrame(step);else el.textContent=m[0]}
P.requestAnimationFrame(step)}
D.addEventListener('click',function(e){var o=e.target.closest&&e.target.closest('[data-testid="stSidebar"] label[data-testid="stRadioOption"]');
if(!o||P.innerWidth>900)return;P.setTimeout(function(){var b=D.querySelector('[data-testid="stSidebarCollapseButton"] button');if(b)b.click()},450)},true);
function scan(){D.querySelectorAll('.kpi .v:not([data-cu])').forEach(function(el){el.setAttribute('data-cu','1');count(el)})}
new P.MutationObserver(scan).observe(D.body,{childList:true,subtree:true});scan()})();
</script>"""


def fx():
    """Cursor glow, card tilt/spotlight and KPI count-up. Runs once per browser tab (guarded)."""
    try:
        components.html(FX_JS, height=0)
    except Exception:
        embed(FX_JS, 1)


def tape():
    """Fixed bottom ticker of simulated transactions (decorative)."""
    import random
    r = random.Random(7)
    items = []
    for _ in range(18):
        a, b, amt = r.randint(1000, 9999), r.randint(1000, 9999), r.choice([240, 560, 1310, 2875, 4120, 9400, 9940, 18650])
        k = r.random()
        tag = ('<b class="f">FLAGGED</b>' if k < .22 else f'<b class="f">RING #{r.randint(1, 40)}</b>' if k < .3 else '<b class="o">CLEARED</b>')
        items.append(f'<span>TX {a} &rarr; {b} &middot; ${amt:,} &middot; {tag}</span>')
    row = "".join(items)
    st.markdown(f'<div class="tape"><div class="lab"><span class="live"></span>simulated tape</div>'
                f'<div class="tape-in">{row}{row}</div></div>', unsafe_allow_html=True)