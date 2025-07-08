import streamlit as st
import pandas as pd
import os
import openai
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from typing import List
import openai
import faiss
import numpy as np
import tiktoken

load_dotenv()
openai.api_key = st.secrets("OPENAI_API_KEY")

st.set_page_config(page_title="Tovvra Demo", layout='wide')

st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
    }
    .stMainBlockContainer {
        display: flex;
        justify-content: center;
        width: fit-content;
    }


    .stButton{ 
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: #483248;
        color: black;
        width: 250px;
        height: 60px;
        border-radius: 10px;
        font-size: 17px;
        font-weight: bold;
        box-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;

        .st-emotion-cache-1rwb540 { 
        width: 90%;
        
        }
    }
    .st-emotion-cache-1rwb540:hover{
        color: black;
        font-weight: 50px;
        border-color: #483248;
        background-color: #E6E6FA;
        box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.15);

    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "label" not in st.session_state:
    st.session_state["label"] = None

# Your button labels and keys
pages = {
    "Nutrition Calculator": "nutrition_calculator",
    "Label Review Assistant": "label_review",
    "Search Rules": "search_specs",
    "Manage Docs": "manage_docs"
}

st.image("images/logo.svg", width=35)

# Create 4 horizontal columns
cols = st.columns(4)

# Loop through your pages and put each button in its own column
for i, (label, key) in enumerate(pages.items()):
    with cols[i]:
        if st.button(label, key=key):
            st.session_state["label"] = key
            st.rerun()

# Display content based on selected page
if st.session_state["label"] == "nutrition_calculator":
    st.title("🧪 Nutrition Calculator")
    st.write("Upload a CSV with lab results (nutrients per 100g), and we'll help convert them to serving sizes.")
    
    uploaded_file = st.file_uploader("Upload your lab results CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Here's your uploaded data:")
        st.dataframe(df)

        # Let user enter a serving size (e.g. 150g)
        serving_size = st.number_input("Enter serving size in grams (e.g. 150g)", min_value=1.0, step=1.0)

        if serving_size:
            st.write(f"Nutrient values per {serving_size}g serving:")

            # Scale the values from 100g to the serving size
            df_converted = df.copy()
            df_converted["Amount per Serving"] = (df["Amount per 100g"] * serving_size) / 100
            df_converted = df_converted[["Nutrient", "Amount per Serving", "Unit"]]

            # Convert units like IU → µg for specific vitamins
            def convert_iu_to_mcg(row):
                if row["Unit"] == "IU":
                    if "Vitamin A" in row["Nutrient"]:
                        return row["Amount per Serving"] * 0.3, "µg"
                    elif "Vitamin D" in row["Nutrient"]:
                        return row["Amount per Serving"] * 0.025, "µg"
                    elif "Vitamin E" in row["Nutrient"]:
                        return row["Amount per Serving"] * 0.67, "µg"
                return row["Amount per Serving"], row["Unit"]

            df_converted[["Amount per Serving", "Unit"]] = df_converted.apply(convert_iu_to_mcg, axis=1, result_type="expand")

            # Show the converted table
            st.dataframe(df_converted)

            # Let user download the new table as a CSV
            csv = df_converted.to_csv(index=False)
            b64 = csv.encode()
            st.download_button(
                label="📥 Download Converted Table as CSV",
                data=b64,
                file_name="converted_nutrition_data.csv",
                mime="text/csv"
            )

# logic for label review assistant 
# display file uploader for old artwork 
# display file uploader for new artwork 
if st.session_state["label"] == "label_review":
    st.title("Label Review Assistant")
    st.write("Upload the old and new label artwork, and we'll help you review the changes.")

    old_artwork = st.file_uploader("Upload old label artwork", type=["png", "jpg", "jpeg"])
    new_artwork = st.file_uploader("Upload new label artwork", type=["png", "jpg", "jpeg"])

    def extract_text_from_image(image_file):

        image = Image.open(image_file).convert("RGB")
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        return text

    if old_artwork and new_artwork:
        col1, col2 = st.columns(2)
        with col1:
            st.image(old_artwork, caption="Old Label Artwork")
        with col2:
            st.image(new_artwork, caption="New Label Artwork")

        # Display button to compare
        if st.button("Compare Labels"):
            old_text = extract_text_from_image(old_artwork)
            new_text = extract_text_from_image(new_artwork)

            old_lines = set(old_text.splitlines())
            new_lines = set(new_text.splitlines())

            added_text = new_lines - old_lines
            removed_text = old_lines - new_lines
            unchanged_text = old_lines & new_lines

            st.subheader("Comparison Results")
            st.markdown("### ➕ Added Text")
            st.write("\n".join(added_text) if added_text else "None")

            st.markdown("### ➖ Removed Text")
            st.write("\n".join(removed_text) if removed_text else "None")

            st.markdown("### ✅ Unchanged Text")
            st.write("\n".join(unchanged_text) if unchanged_text else "Mostly unchanged or similar")

    else:
        st.warning("Please upload both the old and new label artwork to begin.")


if st.session_state["label"] == "search_specs":
    st.title("🔍 Search Canadian Regulations (Smart)")
    st.write("Ask a regulatory question or type a keyword — we'll search across CHFA + Health Canada and explain what we find.")

    # Scraper
    # def scrape_page(url):
    #     try:
    #         response = requests.get(url, timeout=10)
    #         soup = BeautifulSoup(response.text, 'html.parser')
    #         for tag in soup(["nav", "footer", "header", "script", "style"]):
    #             tag.decompose()
    #         text = soup.get_text(separator="\n")
    #         lines = [line.strip() for line in text.splitlines() if line.strip()]
    #         return "\n".join(lines)
    #     except Exception as e:
    #         return f"Error fetching {url}: {e}"
    def scrape_page(url):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)

            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(["nav", "footer", "header", "script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)
        except Exception as e:
            return f"Error fetching {url}: {e}"

    pages_to_scrape = [
        {
            "source": "CHFA - Regulatory Affairs",
            "url": "https://chfa.ca/regulatory-affairs"
        },
        {
            "source": "Health Canada - Probiotics Guidance",
            "url": "https://www.canada.ca/en/health-canada/services/probiotics.html"
        },
        {
            "source": "Health Canada - Ingredient Database Overview",
            "url": "https://www.canada.ca/en/health-canada/services/drugs-health-products/natural-non-prescription/ingredient-database.html"
        },
        {
            "source": "Health Canada - Food Label Requirements",
            "url": "https://www.canada.ca/en/health-canada/services/food-labelling.html"
        }
    ]

    # Embedding + Indexing
    def get_embedding(text, model="text-embedding-3-small"):
        result = openai.embeddings.create(input=[text], model=model)
        return np.array(result.data[0].embedding, dtype="float32")

    def chunk_text(text: str, chunk_size: int = 500):
        paragraphs = text.split("\n")
        chunks, current = [], ""
        for para in paragraphs:
            if len(current) + len(para) < chunk_size:
                current += " " + para
            else:
                chunks.append(current.strip())
                current = para
        if current:
            chunks.append(current.strip())
        return chunks

    def build_search_index(documents):
        index = faiss.IndexFlatL2(1536)  # OpenAI embedding dimension
        metadata = []

        for doc in documents:
            chunks = chunk_text(doc["content"])
            for chunk in chunks:
                embedding = get_embedding(chunk)
                index.add(np.array([embedding]))
                metadata.append({
                    "source": doc["source"],
                    "url": doc["url"],
                    "content": chunk
                })
        return index, metadata

    # Build index once per session
    if "semantic_docs" not in st.session_state:
        scraped_docs = [
    {
        "source": "CHFA - Probiotic Claims Guidance",
        "url": "https://chfa.ca/guidance/probiotics",  # fake link or real one
        "content": """
            In Canada, probiotic claims are regulated by Health Canada and must meet specific evidence-based standards. 
            For a product to carry a claim such as 'supports digestive health', it must include strains with documented effects.
            These claims are classified under structure-function claims, not disease risk reduction.
        """
    },
    {
        "source": "Health Canada - Food Labeling Requirements",
        "url": "https://www.canada.ca/en/health-canada/services/food-labelling.html",
        "content": """
            Food labeling in Canada requires the Nutrition Facts table, ingredient list, and bilingual information. 
            Claims like 'low fat' or 'good source of fiber' must meet regulatory thresholds.
            Labels must not be misleading, and format rules (font size, contrast) must be followed.
        """
    }
]
        
        st.session_state["semantic_docs"] = scraped_docs
        st.session_state["semantic_index"], st.session_state["semantic_meta"] = build_search_index(scraped_docs)

    # Input UI
#     user_query = st.text_input("Ask your question", placeholder="e.g. What claims can I make about probiotics?")

#     if user_query:
#         query_embedding = get_embedding(user_query)
#         D, I = st.session_state["semantic_index"].search(np.array([query_embedding]), k=3)
#         results = [st.session_state["semantic_meta"][i] for i in I[0]]

#         if D[0][0] > 0.85:
#             st.info("We're working on a feature to help answer broader questions — for now, this tool focuses on very specific content.")
#         else:
#             st.subheader("🧠 AI Answer")
#             context = "\n\n".join([r["content"] for r in results])
#             prompt = f"""You're a regulatory assistant helping with Canadian food and health rules. Based on this content, answer the following question clearly, with reference to CHFA and Health Canada if relevant.

# Context:
# {context}

# Question: {user_query}

# Answer:"""

#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a helpful regulatory expert."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.2
#         )
#         answer = response.choices[0].message.content
#         st.markdown(answer)

#         st.subheader("📎 Source Documents")
#         for res in results:
#             st.markdown(f"**{res['source']}**  \n[🔗 Link]({res['url']})")
#             st.markdown(f"> {res['content'][:300]}...")
#             st.write("DEBUG URL:", res["url"])

#     except Exception as e:
#         st.error(f"OpenAI error: {e}")

def get_embedding(text, model="text-embedding-3-small"):
    result = openai.embeddings.create(input=[text], model=model)
    return np.array(result.data[0].embedding, dtype="float32")

def chunk_text(text: str, chunk_size: int = 500):
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += " " + para
        else:
            chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())
    return chunks

def build_search_index(documents):
    index = faiss.IndexFlatL2(1536)  # OpenAI embedding dimension
    metadata = []
    for doc in documents:
        chunks = chunk_text(doc["content"])
        for chunk in chunks:
            embedding = get_embedding(chunk)
            index.add(np.array([embedding]))
            metadata.append({
                "source": doc["source"],
                "url": doc["url"],
                "content": chunk
            })
    return index, metadata


user_query = st.text_input("Ask your question", placeholder="e.g. What claims can I make about probiotics?")

if user_query:
    query_embedding = get_embedding(user_query)
    D, I = st.session_state["semantic_index"].search(np.array([query_embedding]), k=3)
    results = [st.session_state["semantic_meta"][i] for i in I[0]]

    if D[0][0] > 0.85:
        st.info("We're working on a feature to help answer broader questions — for now, this tool focuses on very specific content.")
    else:
        st.subheader("🧠 AI Answer")
        context = "\n\n".join([r["content"] for r in results])
        prompt = f"""You're a regulatory assistant helping with Canadian food and health rules. Based on this content, answer the following question clearly, with reference to CHFA and Health Canada if relevant.

Context:
{context}

Question: {user_query}

Answer:"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful regulatory expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            answer = response.choices[0].message.content
            st.markdown(answer)

            st.subheader("📎 Source Documents")
            for res in results:
                st.markdown(f"**{res['source']}**  \n[🔗 Link]({res['url']})")
                st.markdown(f"> {res['content'][:300]}...")
                st.write("DEBUG URL:", res["url"])

        except Exception as e:
            st.error(f"OpenAI error: {e}")