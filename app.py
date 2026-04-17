"""
RAG Knowledge Base - Starter Template
FAMNIT AI Course - Day 3
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
# DOCUMENTS
# ──────────────────────────────────────────────────────────────────────

DOCUMENTS = [

    """Slovenia has more than 10,000 kilometers of marked hiking trails across a diverse landscape. Within a small country, hikers can move between alpine mountains, rivers, forests, and rolling countryside. Its location between the Alps, the Adriatic, and the Pannonian Plain creates a wide variety of hiking environments.

One of Slovenia’s main strengths as a hiking destination is the balance between accessibility and wilderness. Many trails are close to towns and accommodation, yet still feel natural and remote. Hiking also has a strong cultural role in Slovenia, where mountains and forests are closely tied to everyday life and national identity.""",


    """The Slovenian Alps are the country’s main high-mountain hiking area. They consist of three ranges: the Julian Alps, the Karawanks, and the Kamnik-Savinja Alps. Together, they form the core of alpine hiking in Slovenia, with routes from easier ridge walks to demanding mountain ascents.

Compared with some other Alpine regions, tourist infrastructure is less developed. Many routes require hikers to gain elevation on foot. Above the tree line, around 1,600 meters, terrain becomes rougher, with rock, gravel, narrow trails, and limited surface water, while lower valleys offer easier routes with rivers, lakes, and villages.""",


    """The Julian Alps are the largest alpine region in Slovenia and include Mount Triglav, the country’s highest peak. The area is built mainly of limestone and dolomite, and fossils can still be found because the region was once covered by sea.

Most of the range lies within Triglav National Park and a UNESCO biosphere reserve. Popular hiking bases include Lake Bohinj, Lake Bled, Kranjska Gora, and the Soča Valley. The Julian Alps are also the main area for hut-to-hut hiking, with routes ranging from moderate to demanding.""",


    """The Karawanks are the longest Slovenian mountain range and stretch along the border with Austria. Their terrain is generally gentler than in the Julian Alps, especially on the southern side, which makes them attractive for day hikes without requiring advanced alpine skills.

Popular destinations include Golica, Veliki Vrh, Mount Stol, and Begunjščica. The Karawanks are known for accessible mountain scenery and lower technical difficulty compared with the more rugged alpine ranges.""",


    """The Kamnik-Savinja Alps offer some of the most demanding mountain terrain in Slovenia. Compared with the Karawanks, they are more physically challenging and less developed for casual tourism.

Very few high mountain roads reach the trailheads, so hikers often need to climb large elevation gains in a single day. Rocky terrain and occasional via ferrata sections make some routes more suitable for experienced hikers. At the same time, accessible highlights include Logar Valley, Rinka Waterfall, and Velika Planina.""",


    """The best time for hiking in Slovenia depends on the type of route and altitude. May and June are popular because temperatures are mild, days are long, and nature is in bloom, while trails are usually less crowded than in peak summer.

High-alpine hut-to-hut hiking is best between late June and September, when most mountain huts are open and higher trails are mostly free of snow. Lower-altitude hiking is often more comfortable in spring and autumn, when temperatures are cooler and scenery can be especially attractive.""",


    """Slovenia is generally considered a safe country for hiking, but the main risks are environmental rather than crime-related. Mountain weather can change quickly, terrain can be demanding, and mobile phone coverage may be limited in remote areas.

Hikers should choose routes that match their fitness and experience, prepare properly, and check forecasts before starting. Travel insurance for mountain activities is advisable, and hikers should also be aware of ticks, which can transmit diseases such as Lyme disease and tick-borne encephalitis.""",


    """Reliable navigation is important, especially in alpine and remote areas. Printed maps remain useful, particularly for understanding terrain and route alternatives. Sidarta’s 1:25,000 maps are widely regarded as strong hiking maps for high alpine terrain.

Digital tools such as AllTrails, Gaia GPS, and OutdoorActive can help with route planning and real-time tracking, but they work best as a complement to maps rather than a full replacement. Hut keepers, tourist offices, and Alpine Association resources are also useful sources of current trail information.""",


    """Official hiking trails in Slovenia are marked with the Knafelc waymark, a red and white circle used across the national trail network managed by the Alpine Association of Slovenia. Local hiking paths often use additional signs and route numbers.

Hiking etiquette is also an important part of the experience. Greeting other hikers is customary, and simple Slovenian phrases such as “Dober dan” or “hvala” are appreciated and seen as respectful toward local culture.""",


    """Mountain terrain in Slovenia is often rocky, uneven, and physically demanding. Hikers should expect narrow paths, loose gravel, slippery roots, and steep ascents or descents depending on the route. Good hiking footwear with strong grip is therefore essential, and hiking poles are often useful on longer descents.

Basic preparation should include suitable clothing, enough food and water, and awareness of route difficulty. Equipment cannot replace experience or careful planning, but it can reduce fatigue and lower the risk of injury."""
]


# ──────────────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")


@st.cache_resource(show_spinner="Splitting documents into chunks...")
def build_chunks(_documents: tuple):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in _documents:
        chunks.extend(splitter.split_text(doc))

    return chunks


@st.cache_resource(show_spinner="Building vector database...")
def build_vector_store(_documents: tuple):
    from langchain_community.vectorstores import Chroma

    chunks = build_chunks(_documents)
    embeddings = load_embedding_model()

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
        options=["Home", "Search", "Explore Chunks", "About"],
        icons=["house", "search", "boxes", "info-circle"],
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
        This app helps you find relevant information about mountain ranges, seasons, safety, navigation, trail culture, and hiking preparation.
        """)

        st.subheader("What you can explore")
        st.markdown("""
        - Major hiking regions such as the Julian Alps, Karawanks, and Kamnik-Savinja Alps  
        - Best seasons for different hiking types  
        - Safety, navigation, and trail markings  
        - Terrain, difficulty, and practical preparation  
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
    st.caption("Powered by paraphrase-MiniLM-L3-v2 embeddings + ChromaDB")


# ──────────────────────────────────────────────────────────────────────
# EXPLORE CHUNKS PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "Explore Chunks":
    st.title("Explore Chunks")
    st.markdown("See how your documents are split into chunks by the recursive text splitter.")

    chunks = build_chunks(tuple(DOCUMENTS))

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
    1. **Documents** are split into chunks
    2. Each chunk is converted to an **embedding** (a vector of numbers)
    3. Chunks are stored in a **vector database** (ChromaDB)
    4. When you search, your query is embedded and compared to all chunks
    5. The most **semantically similar** chunks are returned

    ### Technical setup
    - Embedding model: sentence-transformers/paraphrase-MiniLM-L3-v2  
    - Vector database: ChromaDB  
    - Chunking method: RecursiveCharacterTextSplitter  
    - Chunk size: 600  
    - Chunk overlap: 50  
    """)