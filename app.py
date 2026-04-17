"""
RAG Knowledge Base - Starter Template
FAMNIT AI Course - Day 3

A simple Retrieval-Augmented Generation (RAG) app built with
Streamlit, LangChain, and ChromaDB. No API keys needed!

Instructions:
  1. Replace the DOCUMENTS list below with your own texts
  2. Update the app title and description
  3. Run locally:  streamlit run app.py
  4. Deploy to Render (see assignment instructions)
"""

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Hiking in Slovenia",
    page_icon="🥾",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
div[data-testid="stMetric"] {
    background-color: #f8f9fa;
    border: 1px solid #e6e6e6;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# YOUR DOCUMENTS — Replace these with your own topic!
# Each string is one "document" that will be chunked, embedded, and
# stored in the vector database for semantic search.
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
# Cached heavy resources (loaded once, reused across reruns)
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Building vector database...")
def build_vector_store(_documents: tuple):
    """Chunk documents, embed them, and store in ChromaDB."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    # --- Chunking ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in _documents:
        chunks.extend(splitter.split_text(doc))

    embeddings = load_embedding_model()

    # --- Store in ChromaDB ---
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="knowledge_base",
    )
    return vector_store, chunks


# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Hiking in Slovenia")
st.sidebar.caption("Semantic search app")
st.sidebar.markdown("---")

from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Search", "Examples", "Explore Chunks", "About"],
        icons=["house", "search", "lightbulb", "boxes", "info-circle"],
        default_index=0,
    )

page = selected



# ──────────────────────────────────────────────────────────────────────
# HOME PAGE
# ──────────────────────────────────────────────────────────────────────
if page == "Home":
    st.title("🥾 Hiking in Slovenia Knowledge Base")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Explore hiking information about Slovenia using semantic search.  
        This app helps you find relevant information about mountain ranges, seasons, safety, navigation, trail culture, and practical hiking preparation.
        """)

        st.subheader("What you can explore")
        st.markdown("""
        - Major hiking regions such as the Julian Alps, Karawanks, and Kamnik-Savinja Alps  
        - Best seasons for different hiking types  
        - Safety, navigation, and trail markings  
        - Hut-to-hut hiking, wine-region walks, and long-distance trails  
        """)

        st.success("Start exploring → Open the **Search** page and ask your question.")

    with col2:
        st.image("image.jpg", caption="Explore Slovenia", use_container_width=True)

    st.markdown("---")
    st.caption("Built with Streamlit, LangChain, and ChromaDB")

    st.info(f"Knowledge base contains **{len(DOCUMENTS)} documents**.")



# ──────────────────────────────────────────────────────────────────────
# SEARCH PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "Search":
    st.title("🔎 Search Hiking Information")
    st.markdown("""
    Ask a question about hiking in Slovenia and the app will return the most relevant passages from the knowledge base.
    """)

    st.info("""
    Example questions:
    - What is the best time for hiking in Slovenia?
    - How do the Julian Alps differ from the Karawanks?
    - What safety risks should hikers consider in Slovenia?
    - What does the Knafelc waymark mean?
    """)

    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Your question",
            placeholder="e.g. What is the best time for hiking in Slovenia?",
        )

    with col2:
        num_results = st.slider("Results", 1, 10, 3)

    if query:
        with st.spinner("Searching..."):
            results = vector_store.similarity_search_with_score(query, k=num_results)

        st.subheader(f"Top {len(results)} results")

        for i, (doc, score) in enumerate(results, 1):
            similarity = max(0, 1 - score)

            st.markdown(
                f"""
                <div style="
                    border: 1px solid #d9d9d9;
                    border-radius: 12px;
                    padding: 16px;
                    margin-bottom: 16px;
                    background-color: #fafafa;
                ">
                    <h4 style="margin-top: 0;">Result {i}</h4>
                    <p style="margin-bottom: 8px;"><strong>Relevance score:</strong> {similarity:.2f}</p>
                    <p style="margin-bottom: 0;">{doc.page_content}</p>
                </div>
                """,
                unsafe_allow_html=True
            )        
    


    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Documents", len(DOCUMENTS))
    stat2.metric("Chunks", len(chunks))
    stat3.metric("Query length", len(query) if query else 0)

    st.markdown("---")
    st.caption("Powered by all-MiniLM-L6-v2 embeddings + ChromaDB")


# ──────────────────────────────────────────────────────────────────────
# EXPLORE CHUNKS PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "Explore Chunks":
    st.title("Explore Chunks")
    st.markdown("See how your documents are split into chunks by the recursive text splitter.")

    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))

    st.metric("Total chunks", len(chunks))

    lengths = [len(c) for c in chunks]
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg chunk size", f"{np.mean(lengths):.0f} chars")
    col2.metric("Min chunk size", f"{min(lengths)} chars")
    col3.metric("Max chunk size", f"{max(lengths)} chars")

    st.subheader("Chunk length distribution")
    st.bar_chart(lengths)

    st.subheader("Filter chunks")
    keyword = st.text_input(
        "Enter a keyword to search inside chunks",
        placeholder="e.g. Alps"
    )

    filtered_chunks = chunks
    if keyword:
        filtered_chunks = [chunk for chunk in chunks if keyword.lower() in chunk.lower()]

    st.write(f"Showing {len(filtered_chunks)} chunk(s)")

    for i, chunk in enumerate(filtered_chunks, 1):
        with st.expander(f"Chunk {i} ({len(chunk)} chars)"):
            st.text(chunk)

# ──────────────────────────────────────────────────────────────────────
# ABOUT PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "About":
    st.title("About This App")

    st.markdown("""
    This application is a semantic search tool focused on hiking in Slovenia.

    It allows users to search information about:
    - mountain regions
    - hiking seasons
    - safety and navigation
    - trail systems and culture

    ### How it works
    This app lets you **search documents by meaning**, not just keywords.
    1. **Documents** are split into small chunks
    2. Each chunk is converted to an **embedding** (a vector of numbers)
    3. Chunks are stored in a **vector database** (ChromaDB)
    4. When you search, your query is embedded and compared to all chunks
    5. The most **semantically similar** chunks are returned

    ### Technical setup
    - Embedding model: all-MiniLM-L6-v2  
    - Vector database: ChromaDB  
    - Chunking method: RecursiveCharacterTextSplitter  
    - Chunk size: 500  
    - Chunk overlap: 50  
    """)

    
# ──────────────────────────────────────────────────────────────────────
# EXAMPLES PAGE
# ──────────────────────────────────────────────────────────────────────

elif page == "Examples":
    st.title("Try Example Queries")

    st.markdown("Click a question to test the search system:")

    example_queries = [
        "What is the best time for hiking in Slovenia?",
        "Why is Slovenia a good hiking destination?",
        "How do the Julian Alps differ from the Karawanks?",
        "What safety risks should hikers consider?",
        "What does the Knafelc waymark mean?"
    ]

    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))

    for q in example_queries:
        if st.button(q):
            results = vector_store.similarity_search_with_score(q, k=3)

            st.subheader("Results")
            for i, (doc, score) in enumerate(results, 1):
                similarity = max(0, 1 - score)
                st.markdown(f"**Result {i} — {similarity:.2f}**")
                st.write(doc.page_content)
                st.divider()