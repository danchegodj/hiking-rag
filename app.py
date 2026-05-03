"""
Hiking in Slovenia — RAG Knowledge Base
Streamlit + LangChain + scikit-learn TF-IDF
"""

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Hiking in Slovenia",
    page_icon="⛰️",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────
# BASE STYLES  (sidebar, metrics, buttons, inputs — used on every page)
# No external font imports — system font stack only
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
    background-color: #f5fcfe;
}
.block-container { padding: 1.2rem 2.5rem 2rem; max-width: 1100px; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #e0f7fa; }
::-webkit-scrollbar-thumb { background: #0097a7; border-radius: 4px; }

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #b2ebf2 !important;
}
[data-testid="stSidebar"] * { color: #062030 !important; }
[data-testid="stSidebar"] hr { border-color: #b2ebf2 !important; }

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #e0f7fa 0%, #f0fcfe 100%);
    border: 1px solid #b2ebf2;
    border-left: 4px solid #00bcd4;
    padding: 14px 16px;
    border-radius: 12px;
}
div[data-testid="stMetric"] label { color: #0097a7 !important; font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; }
div[data-testid="stMetricValue"] > div { color: #062030 !important; font-size: 1.9rem !important; font-family: Georgia, serif; }

.stButton > button {
    background: #0a3d52 !important; color: #e0f7fa !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important; font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #0097a7 !important;
    box-shadow: 0 6px 16px rgba(0,151,167,0.35) !important;
}

.stTextInput > div > div > input {
    border: 1.5px solid #b2ebf2 !important; border-radius: 12px !important;
    padding: 0.65rem 1rem !important; background: #f5fcfe !important; color: #062030 !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00bcd4 !important;
    box-shadow: 0 0 0 3px rgba(0,188,212,0.15) !important;
}

details { border: 1px solid #b2ebf2 !important; border-radius: 12px !important; background: #f5fcfe !important; margin-bottom: 8px !important; }
details summary { padding: 10px 14px !important; font-weight: 500 !important; color: #0a3d52 !important; cursor: pointer !important; }

hr { border-color: #b2ebf2 !important; margin: 1.5rem 0 !important; }

.main .block-container { animation: fadeUp 0.35s ease both; }
@keyframes fadeUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
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
# VECTOR STORE  (TF-IDF + NumPy — no external API)
# ──────────────────────────────────────────────────────────────────────
class SimpleVectorStore:
    def __init__(self, chunks, matrix, vectorizer):
        self.chunks = chunks
        self.matrix = matrix
        self.vectorizer = vectorizer

    def similarity_search_with_score(self, query, k=3):
        query_vec = self.vectorizer.transform([query]).toarray()
        scores = np.dot(query_vec, self.matrix.T)[0]
        norms = np.linalg.norm(query_vec) * np.linalg.norm(self.matrix, axis=1)
        norms = np.where(norms == 0, 1e-10, norms)
        scores = scores / norms
        top_k = np.argsort(scores)[::-1][:k]

        class Doc:
            def __init__(self, content):
                self.page_content = content

        return [(Doc(self.chunks[i]), 1 - float(scores[i])) for i in top_k]


@st.cache_resource(show_spinner="Building search index…")
def build_vector_store(_documents: tuple):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sklearn.feature_extraction.text import TfidfVectorizer

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in _documents:
        chunks.extend(splitter.split_text(doc))

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunks).toarray()
    return SimpleVectorStore(chunks, matrix, vectorizer), chunks


# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
from streamlit_option_menu import option_menu

with st.sidebar:
    st.markdown(
        '<p style="font-family:Georgia,serif; font-size:1.1rem; font-weight:700; '
        'color:#062030; margin:0 0 4px;">⛰️ Hiking in Slovenia</p>',
        unsafe_allow_html=True,
    )
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
                "color": "#062030", "font-size": "0.88rem", "font-weight": "500",
                "padding": "8px 12px", "border-radius": "8px", "margin-bottom": "2px",
            },
            "nav-link-selected": {
                "background-color": "#e0f7fa", "color": "#0a3d52", "font-weight": "700",
            },
        },
    )
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size:0.75rem; color:#0097a7; margin:0;">'
        f'📚 {len(DOCUMENTS)} documents · TF-IDF search</p>',
        unsafe_allow_html=True,
    )

page = selected


# ──────────────────────────────────────────────────────────────────────
# HOME
# ──────────────────────────────────────────────────────────────────────
if page == "Home":

    st.markdown("""
    <style>
    .hero-chip {
        display: inline-block;
        background: #e0f7fa; border: 1px solid #b2ebf2; color: #0097a7;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
        text-transform: uppercase; padding: 4px 12px; border-radius: 30px;
        margin-bottom: 0.6rem;
    }
    .hero-h1 {
        font-family: Georgia, Cambria, serif;
        font-size: clamp(2.2rem, 5vw, 3.4rem);
        font-weight: 700; color: #0a3d52; line-height: 1.1;
        letter-spacing: -0.02em; margin: 0 0 0.3rem;
    }
    .hero-tagline {
        font-size: 0.95rem; color: #0097a7; font-weight: 400;
        letter-spacing: 0.06em; text-transform: uppercase; margin: 0 0 1rem;
    }
    .stats-ribbon {
        display: grid; grid-template-columns: repeat(4, 1fr);
        background: #0a3d52; border-radius: 0 0 16px 16px;
        overflow: hidden; margin-bottom: 1.8rem;
    }
    .stat-cell { text-align: center; padding: 14px 8px; border-right: 1px solid #0e5570; }
    .stat-cell:last-child { border-right: none; }
    .stat-val { font-family: Georgia, serif; font-size: 1.5rem; font-weight: 700; color: #00bcd4; line-height: 1; margin-bottom: 2px; }
    .stat-lbl { font-size: 0.63rem; color: #80d0dc; letter-spacing: 0.1em; text-transform: uppercase; }
    .feat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 1.4rem; }
    .feat-card {
        background: #fff; border: 1px solid #b2ebf2; border-radius: 12px;
        padding: 18px 16px; position: relative; overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .feat-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #00bcd4, #0097a7);
    }
    .feat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,188,212,0.14); }
    .feat-icon { font-size: 1.5rem; margin-bottom: 8px; display: block; }
    .feat-title { font-family: Georgia, serif; font-size: 0.95rem; font-weight: 700; color: #0a3d52; margin-bottom: 4px; }
    .feat-desc { font-size: 0.8rem; color: #4a7a8a; line-height: 1.5; margin: 0; }
    .cta-banner {
        background: linear-gradient(135deg, #0a3d52 0%, #0097a7 100%);
        border-radius: 14px; padding: 20px 26px;
        display: flex; align-items: center; justify-content: space-between; gap: 16px;
    }
    .cta-text { font-family: Georgia, serif; font-size: 1.15rem; font-weight: 700; color: #e0f7fa; margin: 0; }
    .cta-sub { font-size: 0.8rem; color: #80d0dc; margin: 3px 0 0; }
    .cta-arrow { font-size: 1.6rem; color: #00bcd4; flex-shrink: 0; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hero-chip">⛰️ Semantic Search App</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-h1">Hiking in Slovenia</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-tagline">10,000 km of trails &nbsp;·&nbsp; Alps &nbsp;·&nbsp; Forests &nbsp;·&nbsp; Wine hills</p>', unsafe_allow_html=True)

    st.image("image.jpg", use_container_width=True)

    st.markdown("""
    <div class="stats-ribbon">
        <div class="stat-cell"><div class="stat-val">10K+</div><div class="stat-lbl">km of trails</div></div>
        <div class="stat-cell"><div class="stat-val">3</div><div class="stat-lbl">alpine ranges</div></div>
        <div class="stat-cell"><div class="stat-val">2864m</div><div class="stat-lbl">highest peak</div></div>
        <div class="stat-cell"><div class="stat-val">60%</div><div class="stat-lbl">forest cover</div></div>
    </div>
    <div class="feat-grid">
        <div class="feat-card"><span class="feat-icon">🏔️</span><div class="feat-title">Mountain Ranges</div><p class="feat-desc">Julian Alps, Karawanks, and the wild Kamnik-Savinja Alps — three distinct alpine worlds.</p></div>
        <div class="feat-card"><span class="feat-icon">🗓️</span><div class="feat-title">Seasons & Timing</div><p class="feat-desc">Best windows for alpine huts, daffodil blooms, harvest walks, and golden autumn larches.</p></div>
        <div class="feat-card"><span class="feat-icon">🧭</span><div class="feat-title">Navigation & Safety</div><p class="feat-desc">Knafelc waymarks, Sidarta maps, digital tools, and mountain rescue essentials.</p></div>
        <div class="feat-card"><span class="feat-icon">🍷</span><div class="feat-title">Beyond the Alps</div><p class="feat-desc">Wine-region walks through Goriška Brda, Vipava, and Karst vineyards.</p></div>
        <div class="feat-card"><span class="feat-icon">🥾</span><div class="feat-title">Long-Distance Trails</div><p class="feat-desc">Slovene Mountain Trail, Alpe Adria, Via Dinarica, and the Juliana Trail.</p></div>
        <div class="feat-card"><span class="feat-icon">🍽️</span><div class="feat-title">Food & Water</div><p class="feat-desc">Mountain hut cuisine, tap water safety, springs, and hut-to-hut logistics.</p></div>
    </div>
    <div class="cta-banner">
        <div><p class="cta-text">Ready to explore Slovenia's trails?</p><p class="cta-sub">Open the Search page and ask anything about hiking in Slovenia.</p></div>
        <div class="cta-arrow">→</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Built with Streamlit · LangChain · scikit-learn TF-IDF")


# ──────────────────────────────────────────────────────────────────────
# SEARCH
# ──────────────────────────────────────────────────────────────────────
elif page == "Search":

    st.markdown("""
    <style>
    .section-head { font-family: Georgia, serif; font-size: 1.5rem; font-weight: 700; color: #0a3d52; margin-bottom: 0.2rem; }
    .section-rule { height: 3px; background: linear-gradient(90deg, #00bcd4, #b2ebf2 80%, transparent); border: none; border-radius: 2px; margin-bottom: 1.2rem; }
    .pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1rem; }
    .pill { background: #e0f7fa; color: #0097a7; border: 1px solid #b2ebf2; border-radius: 20px; padding: 4px 12px; font-size: 0.78rem; font-weight: 500; white-space: nowrap; }
    .result-card { border: 1px solid #b2ebf2; border-left: 5px solid #00bcd4; border-radius: 12px; padding: 16px 18px; margin-bottom: 12px; background: linear-gradient(135deg, #f5fcfe 0%, #e8f9fc 100%); }
    .result-num { font-family: Georgia, serif; font-size: 0.75rem; color: #0097a7; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 5px; }
    .result-score { display: inline-block; background: #e0f7fa; color: #0097a7; font-size: 0.7rem; font-weight: 600; padding: 2px 8px; border-radius: 20px; margin-bottom: 8px; border: 1px solid #b2ebf2; }
    .result-text { color: #062030; font-size: 0.9rem; line-height: 1.7; margin: 0; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-head">🔎 Search</h2><div class="section-rule"></div>', unsafe_allow_html=True)

    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))

    st.markdown("""
    <div class="pill-row">
        <span class="pill">Best time to hike</span>
        <span class="pill">Julian Alps vs Karawanks</span>
        <span class="pill">Safety risks</span>
        <span class="pill">Hut-to-hut routes</span>
        <span class="pill">Wine-region walks</span>
        <span class="pill">Knafelc waymark</span>
    </div>
    """, unsafe_allow_html=True)

    col_q, col_n = st.columns([4, 1])
    with col_q:
        query = st.text_input("", placeholder="Ask anything about hiking in Slovenia…", label_visibility="collapsed")
    with col_n:
        num_results = st.slider("Results", 1, 8, 3, label_visibility="collapsed")

    if query:
        with st.spinner("Searching…"):
            results = vector_store.similarity_search_with_score(query, k=num_results)
        st.markdown(f"<p style='color:#0097a7; font-size:0.8rem; margin-bottom:1rem;'>Showing {len(results)} results for <strong>&ldquo;{query}&rdquo;</strong></p>", unsafe_allow_html=True)
        for i, (doc, score) in enumerate(results, 1):
            similarity = max(0, 1 - score)
            bar_w = int(similarity * 100)
            st.markdown(
                f"""<div class="result-card">
                    <div class="result-num">Result {i}</div>
                    <div class="result-score">⬆ {similarity:.0%} match</div>
                    <div style="height:3px;background:#e0f7fa;border-radius:2px;margin-bottom:10px;">
                        <div style="height:3px;width:{bar_w}%;background:linear-gradient(90deg,#0097a7,#00e5ff);border-radius:2px;"></div>
                    </div>
                    <p class="result-text">{doc.page_content}</p>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="padding:28px;text-align:center;color:#0097a7;font-size:0.9rem;'
            'background:#e0f7fa;border-radius:12px;border:1px dashed #b2ebf2;">'
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

    st.markdown("""
    <style>
    .section-head { font-family: Georgia, serif; font-size: 1.5rem; font-weight: 700; color: #0a3d52; margin-bottom: 0.2rem; }
    .section-rule { height: 3px; background: linear-gradient(90deg, #00bcd4, #b2ebf2 80%, transparent); border: none; border-radius: 2px; margin-bottom: 1.2rem; }
    .result-card { border: 1px solid #b2ebf2; border-left: 5px solid #00bcd4; border-radius: 12px; padding: 16px 18px; margin-bottom: 12px; background: linear-gradient(135deg, #f5fcfe 0%, #e8f9fc 100%); }
    .result-num { font-family: Georgia, serif; font-size: 0.75rem; color: #0097a7; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 5px; }
    .result-score { display: inline-block; background: #e0f7fa; color: #0097a7; font-size: 0.7rem; font-weight: 600; padding: 2px 8px; border-radius: 20px; margin-bottom: 8px; border: 1px solid #b2ebf2; }
    .result-text { color: #062030; font-size: 0.9rem; line-height: 1.7; margin: 0; }
    </style>
    """, unsafe_allow_html=True)

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
            st.markdown(f"<p style='color:#0097a7; font-size:0.8rem; margin:0.5rem 0 1rem;'>Results for <strong>&ldquo;{q}&rdquo;</strong></p>", unsafe_allow_html=True)
            for i, (doc, score) in enumerate(results, 1):
                similarity = max(0, 1 - score)
                bar_w = int(similarity * 100)
                st.markdown(
                    f"""<div class="result-card">
                        <div class="result-num">Result {i}</div>
                        <div class="result-score">⬆ {similarity:.0%} match</div>
                        <div style="height:3px;background:#e0f7fa;border-radius:2px;margin-bottom:10px;">
                            <div style="height:3px;width:{bar_w}%;background:linear-gradient(90deg,#0097a7,#00e5ff);border-radius:2px;"></div>
                        </div>
                        <p class="result-text">{doc.page_content}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────────────────────────────
# GALLERY
# ──────────────────────────────────────────────────────────────────────
elif page == "Gallery":

    st.markdown("""
    <style>
    .section-head { font-family: Georgia, serif; font-size: 1.5rem; font-weight: 700; color: #0a3d52; margin-bottom: 0.2rem; }
    .section-rule { height: 3px; background: linear-gradient(90deg, #00bcd4, #b2ebf2 80%, transparent); border: none; border-radius: 2px; margin-bottom: 1.2rem; }
    .gallery-label { font-family: Georgia, serif; font-size: 0.92rem; font-weight: 700; color: #0a3d52; padding: 8px 0 5px; border-bottom: 2px solid #00bcd4; margin-bottom: 6px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-head">🖼️ Destination Gallery</h2><div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("A visual tour through Slovenia's most remarkable hiking landscapes.")

    gallery_items = [
        ("triglav.jpg",        "Mount Triglav",       "The highest peak in Slovenia and a symbol of national identity."),
        ("bohinj.jpg",         "Lake Bohinj",         "An alpine lake surrounded by the Julian Alps and forest trails."),
        ("soca.jpg",           "Soča River",          "Famous for its emerald color and scenic hiking paths."),
        ("velika_planina.jpg", "Velika Planina",       "A vast alpine meadow with traditional shepherd huts."),
        ("logar_valley.jpg",   "Logar Valley",        "One of the most beautiful glacial valleys in Slovenia."),
        ("karawanks.jpg",      "Karawanks",           "A long range with accessible and scenic ridge trails."),
        ("kamnik_alps.jpg",    "Kamnik–Savinja Alps", "A wilder and more demanding alpine hiking region."),
        ("pohorje.jpg",        "Pohorje",             "Forest-covered hills ideal for relaxed local hiking."),
        ("brda.jpg",           "Goriška Brda",        "Rolling wine hills combining hiking with food and wine culture."),
    ]

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for i, (filename, title, caption_text) in enumerate(gallery_items):
        with cols[i % 3]:
            st.markdown(f'<div class="gallery-label">📍 {title}</div>', unsafe_allow_html=True)
            st.image(filename, use_container_width=True)
            st.markdown(f'<p style="font-size:0.78rem;color:#4a7a8a;margin-top:3px;margin-bottom:16px;">{caption_text}</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("All destinations are located in Slovenia.")


# ──────────────────────────────────────────────────────────────────────
# EXPLORE CHUNKS
# ──────────────────────────────────────────────────────────────────────
elif page == "Explore Chunks":

    st.markdown("""
    <style>
    .section-head { font-family: Georgia, serif; font-size: 1.5rem; font-weight: 700; color: #0a3d52; margin-bottom: 0.2rem; }
    .section-rule { height: 3px; background: linear-gradient(90deg, #00bcd4, #b2ebf2 80%, transparent); border: none; border-radius: 2px; margin-bottom: 1.2rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-head">🗂️ Explore Chunks</h2><div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("Inspect how documents are split before vectorisation.")

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
        f'<p style="font-size:0.8rem;color:#0097a7;margin-bottom:0.8rem;">'
        f'Showing <strong>{len(filtered)}</strong> of {len(chunks)} chunks'
        + (f' &nbsp;·&nbsp; filtered by &ldquo;{keyword}&rdquo;' if keyword else '')
        + '</p>',
        unsafe_allow_html=True,
    )
    for i, chunk in enumerate(filtered, 1):
        with st.expander(f"Chunk {i} · {len(chunk)} chars"):
            st.code(chunk, language=None)


# ──────────────────────────────────────────────────────────────────────
# ABOUT
# ──────────────────────────────────────────────────────────────────────
elif page == "About":

    st.markdown("""
    <style>
    .section-head { font-family: Georgia, serif; font-size: 1.5rem; font-weight: 700; color: #0a3d52; margin-bottom: 0.2rem; }
    .section-rule { height: 3px; background: linear-gradient(90deg, #00bcd4, #b2ebf2 80%, transparent); border: none; border-radius: 2px; margin-bottom: 1.2rem; }
    .about-step { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 10px; padding: 11px 14px; background: #f0fbfe; border-radius: 10px; border: 1px solid #b2ebf2; }
    .step-num { background: #0a3d52; color: #b2ebf2; font-family: Georgia, serif; font-size: 0.95rem; font-weight: 700; min-width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; }
    .step-txt { font-size: 0.88rem; color: #062030; line-height: 1.5; }
    </style>
    """, unsafe_allow_html=True)

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
            st.markdown(f'<div style="padding:6px 0;border-bottom:1px solid #b2ebf2;font-size:0.86rem;color:#062030;">▸ {t}</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown("**How it works**")
        steps = [
            ("1", "Documents split into 500-char chunks"),
            ("2", "Each chunk vectorised with TF-IDF"),
            ("3", "Vectors stored as a NumPy array in memory"),
            ("4", "Query vectorised at search time"),
            ("5", "Nearest chunks returned by cosine similarity"),
        ]
        for num, txt in steps:
            st.markdown(
                f'<div class="about-step"><div class="step-num">{num}</div><div class="step-txt">{txt}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.markdown("**Embedding**  \nTF-IDF (scikit-learn)")
    col_t2.markdown("**Vector store**  \nNumPy in-memory array")
    col_t3.markdown("**Chunking**  \nRecursiveCharacterTextSplitter")
    st.caption("Built with Streamlit · LangChain · scikit-learn")