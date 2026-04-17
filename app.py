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

    """Slovenia offers more than 10,000 kilometers of marked and mostly well-maintained hiking trails across a highly diverse landscape. Within a relatively small country, hikers can move between alpine mountains, turquoise rivers and lakes, forests, and rolling countryside. Because Slovenia lies between the south-eastern Alps, the Adriatic area, and the Pannonian Plain, it combines several geographical environments in a compact space.

One of Slovenia’s main advantages as a hiking destination is the balance between accessibility and wilderness. Many trails are close to towns, roads, and accommodation, yet still provide a strong sense of being in nature. Hiking also has an important cultural role in Slovenia. Mountains are closely connected to national identity, forests cover almost 60 percent of the country, and outdoor recreation is part of everyday life for many Slovenians.""",


    """The Slovenian Alps are the main high-mountain hiking area in Slovenia. They consist of three major ranges: the Julian Alps, the Karawanks, and the Kamnik-Savinja Alps. Together, these ranges form the core of alpine hiking in the country, with routes ranging from accessible ridge walks to demanding high-altitude ascents.

Compared with some other Alpine regions, the Slovenian Alps have less large-scale tourist infrastructure. Fewer high mountain roads and cable cars mean that many hikes require hikers to gain elevation on foot. Above the tree line, around 1,600 meters, the terrain becomes rougher and more exposed. Hikers should expect limestone rock, gravel, narrow trails, and limited surface water. Lower valleys, by contrast, offer easier routes with rivers, lakes, waterfalls, farms, and villages.""",


    """The Julian Alps are the largest and most important alpine region in Slovenia. They contain fifteen of the country’s highest peaks, including Mount Triglav, which is the highest mountain in Slovenia. Geologically, the region is built mainly of limestone and dolomite, and fossils of marine organisms can still be found because the area was once covered by sea.

The Julian Alps are significant from a conservation perspective. UNESCO has recognized the range as a biosphere reserve, and most of it lies within Triglav National Park, established in 1924. The range stretches across northwestern Slovenia, with common bases for hiking around Lake Bohinj, Lake Bled, Kranjska Gora, and the Soča Valley.

The Julian Alps are also the main area for hut-to-hut hiking in Slovenia. Routes around Bohinj are generally easier, while north-to-south traversals are more demanding and may include secured sections with steel cables.""",


    """The Karawanks are the longest Slovenian mountain range. Their ridges extend from the triple border of Italy, Austria, and Slovenia and continue eastward for about 120 kilometers, mostly along the Slovenian-Austrian border.

For many hikers, the Karawanks are more accessible than the Julian Alps or the Kamnik-Savinja Alps. The terrain is generally less dramatic, especially on the Slovenian side, where the southern slopes are greener and gentler. This makes the Karawanks attractive for day hikes without requiring strong alpine experience.

Well-known hiking destinations include Golica, famous for daffodils in late spring, Veliki Vrh on the Košuta ridge, Mount Stol, and Begunjščica. The Karawanks are also part of long-distance routes such as the Slovene Mountain Trail and the Via Alpina.""",


    """The Kamnik-Savinja Alps occupy the central northern part of Slovenia and offer some of the wildest and most demanding mountain terrain in the country. Compared with the Karawanks, they are more physically challenging and less developed for casual tourism.

Very few high mountain roads bring hikers close to trailheads, so reaching the higher peaks often requires large elevation gain in a single day. Rocky karst terrain and occasional via ferrata sections make some routes suitable mainly for fit and experienced hikers.

At the same time, the region includes several accessible and popular destinations. Logar Valley and Rinka Waterfall are among the best-known natural landmarks, while Velika Planina is one of the largest active alpine meadows in Europe and a well-known easier hiking destination. Jezersko and the farms around Logar Valley are good bases for quieter hiking holidays.""",


    """Slovenia has a strong tradition of long-distance hiking and was among the European pioneers of organized long-distance trails. The Slovene Mountain Trail, established in 1953, was one of the first routes of its kind in Europe. Since then, more routes have been created, from demanding alpine traverses to easier inn-to-inn journeys.

The Slovene Mountain Trail is about 617 kilometers long and connects eastern Slovenia with the Adriatic coast. Another major route is the Alpe Adria Trail, around 750 kilometers long, linking Austria, Slovenia, and Italy through valleys, mountain passes, and cultural landscapes. Other notable options include the Via Dinarica and the Juliana Trail around Triglav National Park.

These routes make Slovenia attractive not only for day hikers, but also for people planning multi-day or multi-week walking journeys.""",


    """The best time for hiking in Slovenia depends on the type of route and the region. May and June are especially popular because temperatures are pleasant, days are long, and nature is in bloom, while trails are still less crowded than in peak summer.

High-alpine hut-to-hut hiking is most suitable between late June and September, when most mountain huts are open and snow has mostly cleared from higher trails. Outside this period, alpine routes may become less practical or unsafe because of snow, closed huts, and unstable mountain conditions.

Wine-region and lower-altitude hiking are often better in spring and autumn than in the summer heat. September is attractive because of grape harvest, while October brings strong autumn colors across the country.""",


    """Slovenia is ranked among the safer countries in the world, and general conditions for travelers are good. Hikers should still take normal precautions, especially at trailheads and parking areas where occasional car burglary has been reported.

The main safety risks during hiking are environmental. Mountain weather can change quickly, trails can be physically demanding, and mobile phone coverage is not always reliable in remote areas. Choosing routes that match personal fitness and experience, preparing properly, and monitoring forecasts are essential safety habits.

Slovenia’s mountain rescue service is highly regarded, but it should not replace careful planning. Travel insurance for mountain activities is advisable. Hikers should also be aware of ticks, which can transmit tick-borne encephalitis and Lyme disease.""",


    """Reliable navigation is an important part of hiking in Slovenia, especially in alpine or remote areas. Printed maps remain useful, particularly for hikers who want a broad overview of terrain and route alternatives. Sidarta’s 1:25,000 hiking maps are widely regarded as excellent for high alpine terrain, while maps from the Alpine Association of Slovenia cover a wider range of regions.

Digital tools such as AllTrails, Gaia GPS, and OutdoorActive can be helpful for route planning and real-time location tracking, but they work best as a complement to printed maps rather than a full replacement. Local tourist offices, hut keepers, and official Alpine Association resources are also useful sources of current route and weather information.""",


    """Official hiking trails in Slovenia are marked with the Knafelc waymark, a red and white circle used across the national trail network managed by the Alpine Association of Slovenia. On hiking maps, these trails appear as a dense web of marked routes. In addition to the national trail system, many municipalities and regions have developed local hiking paths with their own signposts and route numbers.

Hiking etiquette is also an important part of the experience. Greeting other hikers is customary on Slovenian trails, even when passing strangers. Simple expressions such as “Dober dan,” “živjo,” “hvala,” or “prosim” are appreciated and are seen as respectful toward local culture.""",


    """Terrain in Slovenian mountain areas is often rocky, uneven, and physically demanding. Hikers should expect narrow paths, loose gravel, slippery roots, and steep ascents or descents depending on the route. Sturdy hiking footwear with good grip is therefore essential, and hiking poles are often useful on longer descents or unstable surfaces.

Equipment should match the difficulty and length of the route. Basic preparation includes suitable shoes, weather-appropriate clothing, and enough food and water. On multi-day hiking trips, some hikers also carry light footwear for use around mountain huts.

Good equipment does not replace experience or careful route planning, but it can reduce fatigue and lower the risk of slipping or injury.""",


    """Water in Slovenia is generally abundant and of high quality. Tap water is safe to drink throughout the country, and hikers in lower areas may also encounter springs or water troughs that are commonly used as reliable water sources. Some hikers prefer to use a portable filter when drinking from open natural sources.

Food is an important part of the hiking experience. In many regions, hikers can find traditional Slovenian dishes near hiking areas, while mountain huts typically serve simple, hearty meals such as stews.

High-alpine hut-to-hut hikes require more careful planning for both water and food logistics. Surface water is limited at higher elevations because of porous limestone terrain, so hikers may need to buy bottled water at huts or carry larger reserves themselves."""
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