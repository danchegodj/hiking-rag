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
[data-testid="stSidebar"] * { color: #062030 !important; }
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

    """Slovenia offers more than 10,000 kilometers of marked and mostly 
well-maintained hiking trails across a highly diverse landscape. Within 
a relatively small country, hikers can move between alpine mountains, 
turquoise rivers and lakes, dense forests, wine regions, and rolling 
countryside. Slovenia lies between the south-eastern Alps, the Adriatic 
area, and the Pannonian Plain, and the combination of these geographical 
zones creates an unusual variety of terrain within a compact space.

One of Slovenia's key advantages as a hiking destination is the balance 
between accessibility and wilderness. Many trails are close to towns, 
roads, and accommodation, yet still provide a strong sense of being in 
nature. Close proximity to civilization combined with genuine wilderness 
makes Slovenia attractive both for beginners and for experienced hikers 
looking for longer and more demanding routes.

Hiking holds an important cultural role in Slovenia. Mountains are deeply 
connected to national identity, and Slovenia is one of the few countries 
in the world with a mountain symbol on its national flag. Forests cover 
almost 60 percent of the country, and outdoor recreation is a regular 
part of everyday life for many Slovenians. Hiking in Slovenia is therefore 
not only a tourist activity but also an expression of local culture and 
lifestyle.""",


    """The Slovenian Alps are the most important high-mountain hiking area 
in Slovenia. They consist of three main mountain ranges: the Julian Alps, 
the Karawanks, and the Kamnik-Savinja Alps. Together, these ranges form 
the core of alpine hiking in the country and offer routes ranging from 
accessible ridge walks to demanding high-altitude ascents.

Compared with some other Alpine regions in Europe, the Slovenian Alps 
have less large-scale tourist infrastructure in the mountains. Fewer 
high mountain roads and fewer cable cars mean that many routes require 
hikers to gain elevation entirely on foot, giving the hiking experience 
a more natural and less commercialized character.

Above the tree line, which lies at around 1,600 meters, the terrain 
becomes rougher and more exposed. Hikers should expect limestone rock, 
gravel, steep ascents and descents, and narrow trails. Because limestone 
is porous, surface water is often scarce at higher elevations, so 
carrying enough water is important on alpine routes. The lower valleys, 
by contrast, are rich in rivers, lakes, waterfalls, farms, and villages, 
and offer easier and more scenic day hikes. The contrast between rugged 
alpine terrain above and accessible green valleys below makes the 
Slovenian Alps suitable for many different hiking styles and fitness 
levels.""",


    """The Julian Alps are the largest and most important alpine region in 
Slovenia. The range contains fifteen of the country's highest peaks, 
including Mount Triglav, which at 2,864 meters is the highest mountain 
in Slovenia. Geologically, the Julian Alps are built mainly of limestone 
and dolomite, and parts of the rock are around 200 million years old. 
Because the area was once covered by sea, fossils of marine organisms 
can still be found in some places.

The Julian Alps are especially significant from a conservation 
perspective. UNESCO has recognized the range as a biosphere reserve, 
and most of the area lies within Triglav National Park, one of the 
oldest protected natural areas in Europe, established in 1924. Being 
inside Triglav National Park means the Julian Alps are protected from 
large-scale development and preserve a high degree of natural character.

The range stretches across northwestern Slovenia, with the Sava River 
forming part of its border on one side and the Soča River cutting 
through the mountains from north to south. Common bases for day hikes 
include Lake Bohinj, Lake Bled, Kranjska Gora, and the Soča Valley, 
all of which offer accommodation, trail access, and other outdoor 
activities.

The Julian Alps are also the main area for hut-to-hut hiking in 
Slovenia. Routes on the south side around Lake Bohinj are generally 
easier, while north-to-south traversals of the range are more demanding 
and may include secured sections with steel cables. Hut-to-hut hiking 
in the Julian Alps is typically only possible between late June and 
September, when mountain huts are open.""",


    """The Karawanks are the longest Slovenian mountain range. The ridges 
and saddles of the Karawanks extend from the triple border of Italy, 
Austria, and Slovenia and continue eastward for about 120 kilometers, 
mostly following the Slovenian-Austrian border.

For many hikers, the Karawanks are more accessible than the Julian Alps 
or the Kamnik-Savinja Alps. The terrain is generally less dramatic, 
especially on the Slovenian side, where the southern slopes are greener 
and gentler. Shorter and less technically demanding routes to summits 
and along ridges make the Karawanks an attractive area for day hikes 
without requiring previous alpine experience.

Well-known hiking destinations in the Karawanks include Golica, which 
is famous for daffodils in late May and June, Veliki Vrh on the Košuta 
ridge, Mount Stol as the highest peak in the range, and Begunjščica 
with panoramic views near Lake Bled. The range is also part of 
important long-distance hiking routes, including the Slovene Mountain 
Trail and the international Via Alpina.

Mountain huts in the Karawanks are fewer than in the higher and more 
remote alpine regions, partly because easier access means most hikes 
can be completed in a single day without overnight stops. Overall, the 
Karawanks offer beautiful mountain scenery with a lower level of 
technical difficulty compared to other alpine ranges in Slovenia.""",


    """The Kamnik-Savinja Alps occupy the central northern part of Slovenia 
and are clearly visible from Ljubljana on clear days. The range extends 
roughly 66 kilometers from west to east and offers some of the wildest 
and most demanding mountain terrain in the country.

Compared with the Karawanks, the Kamnik-Savinja Alps are more 
physically challenging and less developed for casual tourism. Very few 
high mountain roads bring hikers close to trailheads, so reaching the 
higher peaks often requires climbing up to 1,500 meters of elevation 
gain in a single day. Rocky karst terrain and occasional via ferrata 
sections make some routes suitable only for fit and experienced hikers.

At the same time, the region contains several accessible and highly 
attractive hiking destinations. Logar Valley and Rinka Waterfall are 
among the best-known natural landmarks in the range, while Velika 
Planina is one of the largest active alpine meadows in Europe and a 
popular destination for easier hikes in a traditional pastoral setting.

For hikers who prefer a quieter base away from the most tourist-heavy 
areas, Jezersko and the farms around Logar Valley are good options. 
The Kamnik-Savinja Alps are best suited to hikers who are looking for 
a more remote and demanding mountain experience than the Julian Alps 
or Karawanks typically offer.""",


    """South and east of the main alpine ranges, Slovenia transitions into 
a landscape of forests, valleys, hills, villages, and mountain farms. 
This subalpine and hilly countryside is less internationally famous 
than the Julian Alps, but it offers a culturally rich hiking experience 
that reflects everyday Slovenian life more directly than the high 
mountains.

Trails in the subalpine hills are often used by locals for after-work 
walks, family outings, and weekend hikes to mountain huts for a meal. 
Hiking in these regions provides insight into rural traditions, local 
food culture, and the close relationship between Slovenian settlements 
and the surrounding natural landscape.

The terrain is generally lower and less technical than in the high 
mountains, but some peaks still rise close to the tree line and offer 
wide panoramic views. Notable destinations in the subalpine region 
include Nanos above the Vipava Valley, Snežnik with its karst 
landscape and bear population, Blegoš and Porezen in the pre-alpine 
hills, and the Lovrenc Lakes area on Pohorje plateau.

The subalpine and hilly regions of Slovenia are a good option for 
hikers who want quieter trails, a sense of local atmosphere, and a 
more authentic experience beyond the country's best-known alpine 
destinations.""",


    """Slovenia offers hiking experiences outside the classic mountain 
environment, particularly in its wine-growing regions. Wine-region 
hiking routes combine walking through vineyards and rural villages 
with visits to family-run wine cellars and tastings of local food 
and wine. Instead of steep alpine ascents, the focus is on rolling 
hills, cultural landscapes, and regional gastronomy.

Wine-region hiking routes are available in several parts of Slovenia, 
including the Karst, Vipava Valley, Goriška Brda, Istria, Dolenjska, 
and Štajerska. A particularly well-developed area is Svečina on the 
Slovenian-Austrian border, where the local tourist office has designed 
and marked five circular hiking routes through the vineyards.

Wine-region hikes differ from mountain hikes in one important practical 
way: many routes include sections along low-traffic local asphalt or 
gravel roads rather than continuous mountain trails. Wine-region hikes 
are therefore usually less physically demanding than alpine routes, 
but still attractive for visitors who want to combine outdoor activity 
with local food culture and a different side of Slovenia.

Wine-region hiking shows that Slovenia is not only about alpine peaks 
and mountain scenery, but also about cultural landscapes, rural 
heritage, and regional identity expressed through food and wine.""",


    """Slovenia has a strong tradition of long-distance hiking and was 
among the European pioneers of organized long-distance trails. The 
Slovene Mountain Trail, established in 1953, was one of the first 
routes of its kind in Europe. Since then, additional long-distance 
routes have been created, ranging from demanding alpine traversals 
to more comfortable inn-to-inn journeys with luggage transfer.

The Slovene Mountain Trail is approximately 617 kilometers long and 
connects eastern Slovenia with the Adriatic coast. Starting in 
Maribor, the trail crosses the forests of Pohorje, climbs through 
the Kamnik-Savinja Alps and the Julian Alps, and then continues 
south through the Karst region toward the sea.

The Alpe Adria Trail is around 750 kilometers long and links Austria, 
Slovenia, and Italy. Although the route crosses the Karawanks and 
Julian Alps, it is not a high-alpine trail. The Alpe Adria Trail 
primarily follows valleys, mountain passes, and cultural landscapes, 
including the Soča Valley and wine-growing regions of western Slovenia.

Other notable long-distance options include the Via Dinarica, which 
extends through six countries across the Western Balkans with a 
160-kilometer section in Slovenia, and the Juliana Trail, a circular 
route around Triglav National Park and the eastern Julian Alps. 
Together, these trails make Slovenia an attractive destination not 
only for day hikers but also for those planning multi-day or 
multi-week walking journeys.""",


    """The best time for hiking in Slovenia depends on the type of route 
and the region being visited. May and June are especially popular 
months because temperatures are pleasant, days are long, nature is 
in bloom, and the summer peak season has not yet begun. Trails are 
less crowded in May and June than in July and August.

High-alpine hut-to-hut hiking is most suitable between late June and 
September. During this period, most mountain huts in the Slovenian 
Alps are open and snow has mostly cleared from higher trails. Outside 
this window, many alpine routes become less practical or potentially 
unsafe because of snow, closed huts, and unpredictable mountain 
conditions.

Wine-region hiking is better suited to spring and autumn than to 
summer, when heat can make walking through vineyard landscapes 
uncomfortable. September is a particularly attractive month for 
wine-region hikes because it coincides with grape harvest. October 
brings autumn colors across the country, including golden larches in 
the Slovenian Alps, making it a favorite month for hikers interested 
in autumn scenery.

Because hiking conditions vary significantly by altitude and region, 
seasonality should always be considered as a separate factor when 
planning any hiking trip to Slovenia.""",


    """Slovenia is ranked among the safer countries in the world, and 
general conditions for travelers are good. Hikers should still take 
normal precautions, especially at trailheads and parking areas where 
occasional car burglary has been reported. Leaving valuables in a 
visible place inside a parked car at a trailhead is not recommended.

The main safety risks during hiking in Slovenia are not crime-related 
but environmental. Mountain weather can change quickly, trails can be 
physically demanding, and mobile phone coverage is not always reliable 
in remote areas. Choosing routes that match personal fitness and 
experience, preparing properly, and monitoring weather forecasts 
before setting out are all important safety habits.

Slovenia's mountain rescue service is highly regarded, but mountain 
rescue should not be treated as a substitute for careful planning and 
preparation. Travel insurance covering mountain activities is strongly 
advisable, especially for more demanding hikes. Hikers should also be 
aware that ticks are common in many natural areas and can transmit 
tick-borne encephalitis and Lyme disease. Applying tick repellent 
before hiking and checking for ticks afterwards are simple but 
effective precautions.

For official trail conditions, safety information, and mountain 
guidance, the Alpine Association of Slovenia maintains regularly 
updated resources on its website.""",


    """Reliable navigation is an important part of hiking in Slovenia, 
especially in alpine or remote areas. Printed maps remain useful, 
particularly for hikers who want a broad overview of terrain and 
route alternatives. Sidarta's 1:25,000 hiking maps are widely 
regarded as among the best for high alpine terrain in Slovenia. Maps 
published by the Alpine Association of Slovenia cover a wider range 
of regions including subalpine and hilly areas.

Digital navigation tools such as AllTrails, Gaia GPS, and 
OutdoorActive can be helpful for route planning and real-time 
location tracking. The usefulness of digital tools depends on the 
quality of uploaded tracks, the user's ability to interpret terrain, 
and the battery life of the device. Digital tools work best as a 
complement to a printed map rather than a replacement for it.

Local tourist offices are a valuable source of practical hiking 
information, especially for regional trail recommendations. In 
mountain areas, hut keepers are often well-informed about current 
route conditions, weather, and local trail status. English is widely 
spoken in Slovenia, and locals are generally willing to help visitors 
with directions or suggestions.

Useful online resources for hiking in Slovenia include the route 
database on Hribi.net, the list of mountain huts and shelters 
maintained by the Alpine Association of Slovenia, official 
information about closed or damaged trails, and detailed weather 
forecasts and radar images from the Slovenian Environmental Agency.""",


    """Official hiking trails in Slovenia are marked with the Knafelc 
waymark, a red and white circle used consistently across the national 
trail network managed by the Alpine Association of Slovenia. On 
hiking maps, these trails appear as a dense web of marked routes 
covering most of the country. In addition to the national trail 
system, many municipalities and regions have developed local hiking 
paths with their own signposts, often using yellow markers and 
numbered routes that correspond to locally published maps.

Hiking etiquette is an important part of the experience on Slovenian 
trails. On hill and mountain paths, greeting other hikers is 
customary, even when passing strangers. Simple Slovenian expressions 
such as "Dober dan" for good day, "živjo" for hello, "hvala" for 
thank you, and "prosim" for please are appreciated and often lead to 
warmer and more friendly interactions with locals. Making an effort 
to use a few words in Slovenian is widely seen as a sign of respect 
for local culture, and can make the overall hiking experience 
noticeably more enjoyable.""",


    """Terrain in Slovenian mountain areas is often rocky, uneven, and 
physically demanding. Hikers should expect narrow paths, loose 
gravel, slippery roots, and steep ascents or descents depending on 
the route and altitude. Rocky and uneven terrain makes sturdy hiking 
footwear with good grip essential for safe and comfortable hiking 
in Slovenia. Hiking poles are also very useful, particularly on 
longer descents or when crossing unstable surfaces.

Equipment choices depend on the difficulty and length of the hike, 
but basic preparation should always include suitable shoes, 
weather-appropriate clothing, and sufficient food and water for the 
planned route. On multi-day hiking trips, many hikers carry light 
footwear such as sandals for use around mountain huts after 
completing each day's walking stage.

Good equipment does not replace experience or careful route planning, 
but appropriate gear can significantly reduce fatigue, improve 
stability, and lower the risk of slipping or injury. Gear should 
always be chosen based on the specific terrain and route rather than 
only on expected weather or general travel style.""",


    """Water in Slovenia is generally abundant and of high quality. Tap 
water is safe to drink throughout the country, and hikers in lower 
areas may also encounter springs or water troughs that are commonly 
used as reliable water sources. Some hikers prefer to use a portable 
water filter when drinking from open natural sources, especially on 
longer routes where avoiding bottled water is a priority.

Food is an important part of the hiking experience in Slovenia. In 
many regions, hikers can find local farm-to-table dishes and 
traditional Slovenian cuisine at guesthouses and smaller restaurants 
near hiking areas. Mountain huts typically serve simple, hearty food 
such as stews and other traditional meals that are practical and 
filling for hikers, although the culinary standard at remote mountain 
huts is usually more basic than at restaurants accessible by road.

High-alpine hut-to-hut hikes require more careful planning for both 
water and food logistics. Surface water is limited at higher 
elevations because of porous limestone terrain, which means hikers 
may need to buy bottled water at mountain huts or carry larger water 
reserves themselves. Thinking ahead about hydration and food supply 
is more important on alpine hut-to-hut routes than on lower valley 
or day hikes."""

]


# ──────────────────────────────────────────────────────────────────────
# CACHED RESOURCES
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


@st.cache_resource(show_spinner="Building vector database…")
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
                "color": "#062030",
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

        st.markdown(f"<p style='color:#0097a7; font-size:0.82rem; margin-bottom:1rem;'>Showing {len(results)} results for <strong>&ldquo;{query}&rdquo;</strong></p>", unsafe_allow_html=True)

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
            st.markdown(f"<p style='color:#0097a7; font-size:0.82rem; margin:0.5rem 0 1rem;'>Results for <strong>&ldquo;{q}&rdquo;</strong></p>", unsafe_allow_html=True)
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
            ("2", "Each chunk vectorised with TF-IDF"),
            ("3", "Vectors stored as a NumPy array in memory"),
            ("4", "Query vectorised at search time"),
            ("5", "Nearest chunks returned by cosine similarity"),
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
    col_t1.markdown("**Embedding**  \nTF-IDF (scikit-learn)")
    col_t2.markdown("**Vector store**  \nNumPy in-memory array")
    col_t3.markdown("**Chunking**  \nRecursiveCharacterTextSplitter")
    st.caption("Built with Streamlit · LangChain · scikit-learn")