"""
Hiking in Slovenia — RAG Knowledge Base
Streamlit + LangChain + ChromaDB
"""

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Hiking in Slovenia",
    page_icon="⛰️",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,700;0,900;1,700&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

/* ─── Palette
   --tq-dark   : #0a3d52   deep ocean
   --tq-mid    : #0097a7   turquoise
   --tq-bright : #00bcd4   bright cyan-turquoise
   --tq-light  : #b2ebf2   pale turquoise
   --tq-xlight : #e0f7fa   near-white turquoise
   --tq-ink    : #062030   text on light
─── */

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #f5fcfe;
}
.block-container {
    padding: 1.2rem 2.5rem 2rem;
    max-width: 1100px;
}

/* scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #e0f7fa; }
::-webkit-scrollbar-thumb { background: #0097a7; border-radius: 4px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #b2ebf2 !important;
}
[data-testid="stSidebar"] * { color: #0a3d52 !important; }
[data-testid="stSidebar"] .sidebar-brand {
    font-family: 'Fraunces', serif;
    font-size: 1.2rem;
    color: #062030 !important;
    letter-spacing: 0.01em;
}
[data-testid="stSidebar"] hr { border-color: #b2ebf2 !important; }

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #e0f7fa 0%, #f0fcfe 100%);
    border: 1px solid #b2ebf2;
    border-left: 4px solid #00bcd4;
    padding: 14px 16px;
    border-radius: 12px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0,188,212,0.18);
}
div[data-testid="stMetric"] label {
    color: #0097a7 !important;
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
div[data-testid="stMetricValue"] > div {
    color: #062030 !important;
    font-family: 'Fraunces', serif;
    font-size: 1.9rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #0a3d52 !important;
    color: #e0f7fa !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #0097a7 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(0,151,167,0.35) !important;
}

/* ── Text input ── */
.stTextInput > div > div > input {
    border: 1.5px solid #b2ebf2 !important;
    border-radius: 12px !important;
    padding: 0.65rem 1rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #f5fcfe !important;
    color: #062030 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00bcd4 !important;
    box-shadow: 0 0 0 3px rgba(0,188,212,0.15) !important;
}

/* ── Expander ── */
details {
    border: 1px solid #b2ebf2 !important;
    border-radius: 12px !important;
    background: #f5fcfe !important;
    margin-bottom: 8px !important;
    transition: box-shadow 0.2s ease;
}
details:hover { box-shadow: 0 2px 12px rgba(0,188,212,0.12) !important; }
details summary {
    padding: 10px 14px !important;
    font-weight: 500 !important;
    color: #0a3d52 !important;
    cursor: pointer !important;
}

/* ── Info / success boxes ── */
[data-testid="stInfo"] {
    background: #e0f7fa !important;
    border-left: 4px solid #00bcd4 !important;
    border-radius: 10px !important;
    color: #062030 !important;
}
[data-testid="stSuccess"] {
    background: #ccf5f8 !important;
    border-left: 4px solid #0097a7 !important;
    border-radius: 10px !important;
    color: #062030 !important;
}

/* ── Divider ── */
hr { border-color: #b2ebf2 !important; margin: 1.5rem 0 !important; }

/* ── Page fade-in ── */
.main .block-container { animation: fadeUp 0.4s ease both; }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ─── HOME: hero image wrapper with overlay ─── */
.hero-wrapper {
    position: relative;
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 0;
}
.hero-wrapper img { display: block; width: 100%; }
.hero-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        135deg,
        rgba(6,32,48,0.82) 0%,
        rgba(0,97,107,0.55) 55%,
        rgba(0,188,212,0.15) 100%
    );
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 2.5rem 2.8rem;
}
.hero-chip {
    display: inline-block;
    background: rgba(0,188,212,0.28);
    border: 1px solid rgba(0,188,212,0.6);
    color: #b2ebf2;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 30px;
    margin-bottom: 0.75rem;
    backdrop-filter: blur(4px);
    width: fit-content;
}
.hero-h1 {
    font-family: 'Fraunces', serif;
    font-size: clamp(2.4rem, 5.5vw, 3.8rem);
    font-weight: 900;
    color: #ffffff;
    line-height: 1.05;
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem;
    text-shadow: 0 2px 20px rgba(0,0,0,0.4);
}
.hero-tagline {
    font-size: 1rem;
    color: #b2ebf2;
    font-weight: 300;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0 0 1.4rem;
}
.hero-cta {
    display: inline-block;
    background: #00bcd4;
    color: #062030;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 10px 22px;
    border-radius: 30px;
    width: fit-content;
    box-shadow: 0 4px 20px rgba(0,188,212,0.45);
}

/* ─── HOME: stats ribbon ─── */
.stats-ribbon {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    background: #0a3d52;
    border-radius: 0 0 20px 20px;
    overflow: hidden;
    margin-bottom: 2rem;
}
.stat-cell {
    text-align: center;
    padding: 16px 8px;
    border-right: 1px solid #0e5570;
}
.stat-cell:last-child { border-right: none; }
.stat-val {
    font-family: 'Fraunces', serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: #00bcd4;
    line-height: 1;
    margin-bottom: 2px;
}
.stat-lbl {
    font-size: 0.66rem;
    color: #80d0dc;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 500;
}

/* ─── HOME: feature cards ─── */
.feat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 1.4rem;
}
.feat-card {
    background: #ffffff;
    border: 1px solid #b2ebf2;
    border-radius: 14px;
    padding: 20px 18px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}
.feat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00bcd4, #0097a7);
}
.feat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 28px rgba(0,188,212,0.16);
}
.feat-icon {
    font-size: 1.6rem;
    margin-bottom: 10px;
    display: block;
}
.feat-title {
    font-family: 'Fraunces', serif;
    font-size: 1rem;
    font-weight: 700;
    color: #0a3d52;
    margin-bottom: 5px;
}
.feat-desc {
    font-size: 0.82rem;
    color: #4a7a8a;
    line-height: 1.55;
    margin: 0;
}

/* ─── HOME: CTA banner ─── */
.cta-banner {
    background: linear-gradient(135deg, #0a3d52 0%, #0097a7 100%);
    border-radius: 16px;
    padding: 22px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}
.cta-text {
    font-family: 'Fraunces', serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #e0f7fa;
    margin: 0;
}
.cta-sub {
    font-size: 0.82rem;
    color: #80d0dc;
    margin: 4px 0 0;
}
.cta-arrow {
    font-size: 1.8rem;
    color: #00bcd4;
    flex-shrink: 0;
}

/* ── Search result card ── */
.result-card {
    border: 1px solid #b2ebf2;
    border-left: 5px solid #00bcd4;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    background: linear-gradient(135deg, #f5fcfe 0%, #e8f9fc 100%);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    animation: fadeUp 0.3s ease both;
}
.result-card:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 20px rgba(0,188,212,0.14);
}
.result-num {
    font-family: 'Fraunces', serif;
    font-size: 0.78rem;
    color: #0097a7;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.result-score {
    display: inline-block;
    background: #e0f7fa;
    color: #0097a7;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 20px;
    margin-bottom: 10px;
    border: 1px solid #b2ebf2;
}
.result-text { color: #062030; font-size: 0.92rem; line-height: 1.7; margin: 0; }

/* ── Gallery ── */
.gallery-label {
    font-family: 'Fraunces', serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #0a3d52;
    padding: 10px 0 6px;
    border-bottom: 2px solid #00bcd4;
    margin-bottom: 8px;
}

/* ── Section heading ── */
.section-head {
    font-family: 'Fraunces', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #0a3d52;
    margin-bottom: 0.2rem;
}
.section-rule {
    height: 3px;
    background: linear-gradient(90deg, #00bcd4, #b2ebf2 80%, transparent);
    border: none; border-radius: 2px; margin-bottom: 1.2rem;
}

/* ── Pills ── */
.pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1rem; }
.pill {
    background: #e0f7fa; color: #0097a7;
    border: 1px solid #b2ebf2; border-radius: 20px;
    padding: 4px 12px; font-size: 0.8rem; font-weight: 500;
    cursor: pointer; white-space: nowrap;
    transition: background 0.15s, color 0.15s;
}
.pill:hover { background: #00bcd4; color: #fff; }

/* ── About steps ── */
.about-step {
    display: flex; align-items: flex-start; gap: 14px;
    margin-bottom: 12px; padding: 12px 16px;
    background: #f0fbfe; border-radius: 10px; border: 1px solid #b2ebf2;
}
.step-num {
    background: #0a3d52; color: #b2ebf2;
    font-family: 'Fraunces', serif; font-size: 1rem; font-weight: 700;
    min-width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
}
.step-txt { font-size: 0.9rem; color: #062030; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# DOCUMENTS
# ──────────────────────────────────────────────────────────────────────
DOCUMENTS = [

    """Slovenia packs more than 10,000 kilometers of marked hiking trails
into a remarkably compact country. Its location at the meeting point
of the south-eastern Alps, the Adriatic area, and the Pannonian Plain
creates an unusual variety of terrain: high alpine peaks, turquoise
rivers, dense forests, and rolling wine hills are all within a short
drive of each other. This geographic diversity makes Slovenia suitable
for beginners and experienced hikers alike. Trails sit close to towns
and accommodation, so the feeling of wilderness and easy access exist
side by side. Hiking is deeply embedded in Slovenian culture. The
national flag carries a mountain symbol, forests cover nearly 60
percent of the country, and outdoor recreation is part of everyday
life. Visiting hikers benefit from well-maintained paths, clear
signage, and a local population genuinely connected to its mountains.""",


    """The Julian Alps are Slovenia's most celebrated mountain range and
home to its highest peak, Mount Triglav at 2,864 meters. The range
is built almost entirely of ancient limestone and dolomite, and most
of it lies within Triglav National Park, a protected area established
in 1924 and recognized by UNESCO as a biosphere reserve. The Soča
and Sava rivers define the range's natural boundaries. Lake Bled,
Lake Bohinj, and Kranjska Gora are the most popular bases for day
hikes, while hut-to-hut routes traverse the range from late June
through September when mountain huts are open. Easier trails follow
the southern slopes around Bohinj; more demanding north-to-south
traversals include via ferrata sections. The Julian Alps offer the
widest choice of alpine experiences in Slovenia and are the first
destination most hikers visit.""",


    """The Karawanks stretch roughly 120 kilometers along the Slovenian-
Austrian border, making them the longest mountain range in Slovenia.
Their southern slopes are gentler and greener than the Julian Alps,
and routes to most summits are technically straightforward, requiring
no prior alpine experience. This makes the Karawanks ideal for day
hikes and families. Well-known destinations include Golica, famous for
its spectacular daffodil fields in late May, Mount Stol, the range's
highest point, and Begunjščica, which offers wide views toward Lake
Bled. Mountain huts are fewer here than in the higher ranges, since
most hikes can be completed in a single day. The Karawanks also form
part of the Slovene Mountain Trail and the international Via Alpina
long-distance routes.""",


    """The Kamnik-Savinja Alps run roughly 66 kilometers across central
northern Slovenia and offer the country's most demanding mountain
terrain. Few roads approach the higher trailheads, so many routes
require 1,200 to 1,500 meters of elevation gain in a single day on
rocky karst paths. Some sections include via ferrata equipment.
Despite the difficulty, the range contains genuinely rewarding
destinations at all levels. Logar Valley and Rinka Waterfall are
among the most beautiful glacial landscapes in Slovenia and can be
visited on an easy valley walk. Velika Planina, a vast highland
meadow with traditional wooden herdsman's huts, is popular with
families and casual hikers. The Kamnik-Savinja Alps suit those who
want a quieter, more remote experience than the Julian Alps provide.""",


    """Slovenia's best season for most hiking is late May through October,
but the ideal window depends heavily on elevation and route type.
May and June offer pleasant temperatures, blooming landscapes, and
fewer crowds than July and August. High-alpine hut-to-hut routes are
only practical from late June to September, when snow has cleared and
mountain huts are open. Wine-region hiking is best in spring and
autumn; September coincides with the grape harvest, adding a cultural
dimension to walks through the vineyards. October brings vivid autumn
color across the forests, including golden larches in the higher alpine
zones, and is a favorite month for experienced hikers. Above the
tree line, conditions can change quickly at any time of year, so
checking weather forecasts before setting out is always important.""",


    """Slovenia's main hiking risks are environmental rather than related
to crime. Mountain weather changes rapidly, trails can be physically
demanding, and mobile coverage is unreliable in remote areas. Choosing
routes that match your fitness level, carrying sufficient water and
food, and monitoring forecasts before departure are the most important
precautions. Ticks are common throughout forested areas and can
transmit tick-borne encephalitis and Lyme disease; applying repellent
and checking carefully after hikes reduces the risk. Travel insurance
covering mountain activities is strongly recommended. Slovenia's
mountain rescue service is professional and well-regarded, but it is
not a substitute for careful preparation. Petty theft from unattended
cars at popular trailheads has been reported; do not leave valuables
visible inside parked vehicles.""",


    """Slovenian trails are marked with the Knafelc waymark, a red circle
on a white background applied to rocks, trees, and posts. This system
is maintained by the Alpine Association of Slovenia and covers the
entire national network. For navigation, Sidarta's 1:25,000 printed
maps are the most reliable choice for high alpine terrain. Digital
apps such as AllTrails, Gaia GPS, and OutdoorActive are useful for
planning and live tracking but should supplement rather than replace
a printed map. Mountain hut keepers are an excellent local source of
current trail conditions. On the trail, greeting other hikers is
customary; a simple "Dober dan" (good day) or "živjo" (hello) is
always appreciated and reflects Slovenian mountain culture. English
is widely spoken in tourist areas.""",


    """Slovenia's wine-region hiking routes offer a completely different
experience from alpine trails. Instead of steep ascents, the focus
is on walking through vineyards and rural villages, visiting family
wine cellars, and tasting local food and wine along the way. Routes
are available across several regions including the Karst, Vipava
Valley, Goriška Brda, and Štajerska. Some sections follow quiet
asphalt or gravel farm roads rather than mountain trails. Wine-region
hikes are less physically demanding than alpine routes and are
especially enjoyable in spring and during the September harvest
period. They offer an authentic glimpse into rural Slovenian life
and show that the country is far more than alpine scenery alone.
Prior planning and booking at local wineries is recommended during
peak harvest weekends.""",


    """Slovenia has a well-developed long-distance hiking network. The
Slovene Mountain Trail, established in 1953, covers 617 kilometers
from Maribor through the Pohorje forests, across the Kamnik-Savinja
and Julian Alps, and down to the Adriatic coast. The Alpe Adria Trail
spans 750 kilometers linking Austria, Slovenia, and Italy, following
valleys and cultural landscapes rather than high ridges, making it
accessible to a wide range of hikers. The Juliana Trail is a circular
route of around 270 kilometers encircling Triglav National Park. The
Via Dinarica passes through Slovenia for approximately 160 kilometers
as part of a larger trans-Balkan route. Most long-distance routes can
be split into independent day sections, and luggage transfer services
are available on the more popular trails.""",


    """Water quality in Slovenia is excellent; tap water is safe everywhere
and many lower-altitude trails have drinking springs or water troughs.
At higher elevations, porous limestone means surface water is often
absent, so carrying enough water between huts is essential on alpine
routes. Mountain huts typically sell water and simple food, but
selection is limited on the most remote sections. For gear, sturdy
ankle-supporting boots with good grip are essential on rocky Slovenian
trails. Hiking poles are highly useful on longer descents. Clothing
should account for rapid weather changes at altitude: a waterproof
layer, insulation, and a sun hat are the core requirements regardless
of the season. Multi-day hikers often carry light sandals for comfort
around mountain huts after each day's stage.""",

]


# ──────────────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Building vector database…")
def build_vector_store(_documents: tuple):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in _documents:
        chunks.extend(splitter.split_text(doc))
    embeddings = load_embedding_model()
    vector_store = Chroma.from_texts(
        texts=chunks, embedding=embeddings, collection_name="knowledge_base",
    )
    return vector_store, chunks


# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
from streamlit_option_menu import option_menu

with st.sidebar:
    st.markdown('<div class="sidebar-brand">⛰️ Hiking in Slovenia</div>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=["Home", "Search", "Examples", "Gallery", "Explore Chunks", "About"],
        icons=["house-fill", "search", "lightning-fill", "images", "grid-3x3-gap", "info-circle-fill"],
        default_index=0,
        styles={
            "container": {"background-color": "transparent", "padding": "0"},
            "icon": {"color": "#0097a7", "font-size": "0.85rem"},
            "nav-link": {
                "color": "#0a3d52",
                "font-size": "0.88rem",
                "font-weight": "500",
                "padding": "8px 12px",
                "border-radius": "8px",
                "margin-bottom": "2px",
            },
            "nav-link-selected": {
                "background-color": "#e0f7fa",
                "color": "#0a3d52",
                "font-weight": "700",
            },
        },
    )
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.75rem; color:#0097a7; padding:4px 0;">'
        f'📚 {len(DOCUMENTS)} documents · semantic search</div>',
        unsafe_allow_html=True,
    )

page = selected


# ──────────────────────────────────────────────────────────────────────
# HOME
# ──────────────────────────────────────────────────────────────────────
if page == "Home":

    # ── Plain text header ──
    st.markdown('<div class="hero-chip" style="background:#e0f7fa; border:1px solid #b2ebf2; color:#0097a7;">⛰️ Semantic Search App</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-h1" style="color:#0a3d52; text-shadow:none; margin-bottom:0.3rem;">Hiking in Slovenia</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-tagline" style="color:#0097a7; margin-bottom:1rem;">10,000 km of trails &nbsp;·&nbsp; Alps &nbsp;·&nbsp; Forests &nbsp;·&nbsp; Wine hills</p>', unsafe_allow_html=True)

    st.image("image.jpg", use_container_width=True)

    # ── Stats ribbon ──
    st.markdown("""
    <div class="stats-ribbon">
        <div class="stat-cell">
            <div class="stat-val">10K+</div>
            <div class="stat-lbl">km of trails</div>
        </div>
        <div class="stat-cell">
            <div class="stat-val">3</div>
            <div class="stat-lbl">alpine ranges</div>
        </div>
        <div class="stat-cell">
            <div class="stat-val">2864m</div>
            <div class="stat-lbl">highest peak</div>
        </div>
        <div class="stat-cell">
            <div class="stat-val">60%</div>
            <div class="stat-lbl">forest cover</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature cards ──
    st.markdown("""
    <div class="feat-grid">
        <div class="feat-card">
            <span class="feat-icon">🏔️</span>
            <div class="feat-title">Mountain Ranges</div>
            <p class="feat-desc">Julian Alps, Karawanks, and the wild Kamnik-Savinja Alps — three distinct alpine worlds in one compact country.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">🗓️</span>
            <div class="feat-title">Seasons & Timing</div>
            <p class="feat-desc">Best windows for alpine huts, daffodil blooms, harvest walks, and golden autumn larches.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">🧭</span>
            <div class="feat-title">Navigation & Safety</div>
            <p class="feat-desc">Knafelc waymarks, Sidarta maps, digital tools, and mountain rescue essentials explained.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">🍷</span>
            <div class="feat-title">Beyond the Alps</div>
            <p class="feat-desc">Wine-region walks through Goriška Brda, Vipava, and Karst vineyards with tastings en route.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">🥾</span>
            <div class="feat-title">Long-Distance Trails</div>
            <p class="feat-desc">Slovene Mountain Trail, Alpe Adria, Via Dinarica, and the Juliana Trail around Triglav.</p>
        </div>
        <div class="feat-card">
            <span class="feat-icon">🍽️</span>
            <div class="feat-title">Food & Water</div>
            <p class="feat-desc">Mountain hut cuisine, tap water safety, spring sources, and logistics for hut-to-hut routes.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CTA banner ──
    st.markdown("""
    <div class="cta-banner">
        <div>
            <p class="cta-text">Ready to explore Slovenia's trails?</p>
            <p class="cta-sub">Open the Search page and ask anything about hiking in Slovenia.</p>
        </div>
        <div class="cta-arrow">→</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Built with Streamlit · LangChain · ChromaDB · all-MiniLM-L6-v2")


# ──────────────────────────────────────────────────────────────────────
# SEARCH
# ──────────────────────────────────────────────────────────────────────
elif page == "Search":
    st.markdown('<h2 class="section-head">🔎 Search</h2><div class="section-rule"></div>', unsafe_allow_html=True)

    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))

    # Quick-search pills
    st.markdown("""
    <div class="pill-row">
      <span class="pill" onclick="void(0)">Best time to hike</span>
      <span class="pill">Julian Alps vs Karawanks</span>
      <span class="pill">Safety risks</span>
      <span class="pill">Hut-to-hut routes</span>
      <span class="pill">Wine-region walks</span>
      <span class="pill">Knafelc waymark</span>
    </div>
    """, unsafe_allow_html=True)

    col_q, col_n = st.columns([4, 1])
    with col_q:
        query = st.text_input(
            "",
            placeholder="Ask anything about hiking in Slovenia…",
            label_visibility="collapsed",
        )
    with col_n:
        num_results = st.slider("Results", 1, 8, 3, label_visibility="collapsed")

    if query:
        with st.spinner("Searching knowledge base…"):
            results = vector_store.similarity_search_with_score(query, k=num_results)

        st.markdown(f"<p style='color:#0097a7; font-size:0.82rem; margin-bottom:1rem;'>Showing {len(results)} results for <strong>\"{query}\"</strong></p>", unsafe_allow_html=True)

        for i, (doc, score) in enumerate(results, 1):
            similarity = max(0, 1 - score)
            bar_w = int(similarity * 100)
            st.markdown(
                f"""<div class="result-card">
                    <div class="result-num">Result {i}</div>
                    <div class="result-score">⬆ {similarity:.0%} match</div>
                    <div style="height:3px; background:#e0f7fa; border-radius:2px; margin-bottom:12px;">
                        <div style="height:3px; width:{bar_w}%; background:linear-gradient(90deg,#0097a7,#00e5ff); border-radius:2px;"></div>
                    </div>
                    <p class="result-text">{doc.page_content}</p>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="padding:32px; text-align:center; color:#0097a7; font-size:0.9rem; '
            'background:#e0f7fa; border-radius:12px; border:1px dashed #b2ebf2;">'
            '⛰️ Type a question above to search the knowledge base</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(DOCUMENTS))
    c2.metric("Chunks", len(chunks))
    c3.metric("Query length", len(query) if query else 0)


# ──────────────────────────────────────────────────────────────────────
# EXAMPLES
# ──────────────────────────────────────────────────────────────────────
elif page == "Examples":
    st.markdown('<h2 class="section-head">⚡ Example Queries</h2><div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("Click any question to run it instantly against the knowledge base.")

    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))

    example_queries = [
        ("🗓️", "What is the best time for hiking in Slovenia?"),
        ("🏔️", "Why is Slovenia a good hiking destination?"),
        ("⚖️", "How do the Julian Alps differ from the Karawanks?"),
        ("⚠️", "What safety risks should hikers consider?"),
        ("🔴", "What does the Knafelc waymark mean?"),
    ]

    for icon, q in example_queries:
        if st.button(f"{icon}  {q}", use_container_width=True):
            results = vector_store.similarity_search_with_score(q, k=3)
            st.markdown(f"<p style='color:#0097a7; font-size:0.82rem; margin:0.5rem 0 1rem;'>Results for <strong>\"{q}\"</strong></p>", unsafe_allow_html=True)
            for i, (doc, score) in enumerate(results, 1):
                similarity = max(0, 1 - score)
                bar_w = int(similarity * 100)
                st.markdown(
                    f"""<div class="result-card">
                        <div class="result-num">Result {i}</div>
                        <div class="result-score">⬆ {similarity:.0%} match</div>
                        <div style="height:3px; background:#e0f7fa; border-radius:2px; margin-bottom:12px;">
                            <div style="height:3px; width:{bar_w}%; background:linear-gradient(90deg,#0097a7,#00e5ff); border-radius:2px;"></div>
                        </div>
                        <p class="result-text">{doc.page_content}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────────────────────────────
# GALLERY
# ──────────────────────────────────────────────────────────────────────
elif page == "Gallery":
    st.markdown('<h2 class="section-head">🖼️ Destination Gallery</h2><div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("A visual tour through Slovenia's most remarkable hiking landscapes.")

    gallery_items = [
        ("triglav.jpg",       "Mount Triglav",         "The highest peak in Slovenia and a symbol of national identity."),
        ("bohinj.jpg",        "Lake Bohinj",           "An alpine lake surrounded by the Julian Alps and forest trails."),
        ("soca.jpg",          "Soča River",            "Famous for its emerald color and scenic hiking paths."),
        ("velika_planina.jpg","Velika Planina",         "A vast alpine meadow with traditional shepherd huts."),
        ("logar_valley.jpg",  "Logar Valley",          "One of the most beautiful glacial valleys in Slovenia."),
        ("karawanks.jpg",     "Karawanks",             "A long range with accessible and scenic ridge trails."),
        ("kamnik_alps.jpg",   "Kamnik–Savinja Alps",   "A wilder and more demanding alpine hiking region."),
        ("pohorje.jpg",       "Pohorje",               "Forest-covered hills ideal for relaxed local hiking."),
        ("brda.jpg",          "Goriška Brda",          "Rolling wine hills combining hiking with food and wine culture."),
    ]

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for i, (filename, title, caption_text) in enumerate(gallery_items):
        with cols[i % 3]:
            st.markdown(f'<div class="gallery-label">📍 {title}</div>', unsafe_allow_html=True)
            st.image(filename, use_container_width=True)
            st.markdown(
                f'<p style="font-size:0.8rem; color:#5a7a5a; margin-top:4px; margin-bottom:18px;">{caption_text}</p>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption("All destinations are located in Slovenia.")


# ──────────────────────────────────────────────────────────────────────
# EXPLORE CHUNKS
# ──────────────────────────────────────────────────────────────────────
elif page == "Explore Chunks":
    st.markdown('<h2 class="section-head">🗂️ Explore Chunks</h2><div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("Inspect how documents are split before embedding.")

    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))
    lengths = [len(c) for c in chunks]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total chunks", len(chunks))
    c2.metric("Avg size", f"{np.mean(lengths):.0f} ch")
    c3.metric("Min size", f"{min(lengths)} ch")
    c4.metric("Max size", f"{max(lengths)} ch")

    st.markdown("**Chunk length distribution**")
    st.bar_chart(lengths, height=160)

    st.markdown("---")
    keyword = st.text_input("🔍 Filter chunks by keyword", placeholder="e.g. Julian Alps")
    filtered = [c for c in chunks if keyword.lower() in c.lower()] if keyword else chunks
    st.markdown(
        f'<p style="font-size:0.82rem; color:#0097a7; margin-bottom:0.8rem;">'
        f'Showing <strong>{len(filtered)}</strong> of {len(chunks)} chunks'
        f'{"  ·  filtered by "" + keyword + """ if keyword else ""}</p>',
        unsafe_allow_html=True,
    )
    for i, chunk in enumerate(filtered, 1):
        with st.expander(f"Chunk {i} · {len(chunk)} chars"):
            st.code(chunk, language=None)


# ──────────────────────────────────────────────────────────────────────
# ABOUT
# ──────────────────────────────────────────────────────────────────────
elif page == "About":
    st.markdown('<h2 class="section-head">ℹ️ About</h2><div class="section-rule"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("**What this app covers**")
        topics = [
            "Mountain regions: Julian Alps, Karawanks, Kamnik-Savinja Alps",
            "Hiking seasons and optimal timing by region",
            "Safety, navigation tools, and trail markings",
            "Long-distance trails and hut-to-hut routes",
            "Wine-region hiking and cultural landscapes",
            "Gear, water, food logistics, and local etiquette",
        ]
        for t in topics:
            st.markdown(f'<div style="padding:6px 0; border-bottom:1px solid #b2ebf2; font-size:0.88rem; color:#062030;">▸ {t}</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown("**How it works**")
        steps = [
            ("1", "Documents split into 500-char chunks"),
            ("2", "Each chunk embedded via all-MiniLM-L6-v2"),
            ("3", "Embeddings stored in ChromaDB"),
            ("4", "Query embedded at search time"),
            ("5", "Nearest chunks returned by cosine distance"),
        ]
        for num, txt in steps:
            st.markdown(
                f'<div class="about-step">'
                f'<div class="step-num">{num}</div>'
                f'<div class="step-txt">{txt}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.markdown("**Embedding**  \nall-MiniLM-L6-v2")
    col_t2.markdown("**Vector DB**  \nChromaDB")
    col_t3.markdown("**Chunking**  \nRecursiveCharacterTextSplitter")
    st.caption("Built with Streamlit · LangChain · ChromaDB")