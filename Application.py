import os
import re
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI
from mp_api.client import MPRester
import py3Dmol
import base64
from typing import Union
import ast
import json
import html
import uuid
import openai
import logging

# ======================================================
# API SETUP (TOP LEVEL)
# ======================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MATERIALS_PROJECT_API_KEY = os.getenv("MATERIALS_PROJECT_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV") or "us-west1-gcp"
PINECONE_INDEX_NAME = "materials-repos"

if not PINECONE_API_KEY:
    st.error("❌ Pinecone API key is missing from `.env`.")
    st.stop()

try:
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=3072,  # Top-level argument for dense index
            metric="cosine",
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": "us-east-1"
                }
            }
        )
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    st.error(f"Pinecone initialization failed: {e}")
    st.stop()

# ======================================================
# UTILITY FUNCTIONS
# ======================================================
def extract_formulas(text, source="LLM"):
    """Parse and validate formulas from LLM output."""
    try:
        if not text:
            return []
        text_clean = text

    # --- Extract formulas from LLM output (expects a Python list in brackets) ---
        match = re.search(r"\[.*?\]", text_clean, re.DOTALL)
        if match:
            candidate = match.group()
            formulas = None
            try:
                formulas = ast.literal_eval(candidate)
            except Exception:
                try:
                    formulas = json.loads(candidate)
                except Exception:
                    formulas = None
            # --- Validate and deduplicate formulas ---
            if isinstance(formulas, list):
                valid = []
                for item in formulas:
                    if not isinstance(item, str):
                        continue
                    s = item.strip()
                    if re.match(r"^[A-Za-z0-9()]+$", s):
                        valid.append(s)
                seen = set()
                out = []
                for v in valid:
                    if v not in seen:
                        seen.add(v)
                        out.append(v)
                return out
        return []
    except Exception:
    # --- Warn if parsing fails ---
        st.warning(f"⚠️ {source} output parsing failed:\n{text}")
        return []


# ======================================================
# 3D STRUCTURE VIEWER UTILITY
# ======================================================
def show_structure(structure, supercell=(2, 2, 2)):
    """Render 3D structure viewer for a material."""
    if structure is None:
        return None
    try:
        conventional = structure.get_conventional_standard_structure()
    except Exception:
        conventional = structure
    big = conventional * supercell
    cif = big.to(fmt="cif")
    viewer = py3Dmol.view(width=500, height=500)
    viewer.addModel(cif, "cif")
    viewer.setStyle({}, {
        "stick": {"radius": 0.15, "colorscheme": "Jmol"},
        "sphere": {"scale": 0.3, "colorscheme": "Jmol"}
    })
    viewer.zoomTo()
    return viewer


# ======================================================
# DOWNLOAD LINK UTILITY
# ======================================================
def make_data_download_link(content: Union[bytes, str], filename: str, mime: str = "text/plain", label: str = "Download") -> str:
    """Create a styled download link for data (CSV, CIF, etc)."""
    if isinstance(content, str):
        b = content.encode('utf-8')
    else:
        b = content
    b64 = base64.b64encode(b).decode()
    href = f"data:{mime};base64,{b64}"
    return f'<a href="{href}" download="{filename}" style="text-decoration:none; background:#00c3ff; color:#001218; padding:8px 12px; border-radius:6px; font-weight:600;">{label}</a>'

@st.cache_resource()

# ======================================================
# MATERIALS PROJECT QUERY UTILITY
# ======================================================
def query_mp_for_formula(formula):
    """Query Materials Project for a given formula, return any entry (stable or unstable)."""
    from pymatgen.core import Composition
    try:
        # Normalize formula (e.g., Fe3O4 -> Fe3O4)
        norm_formula = str(Composition(formula).reduced_formula)
        # 1. Try with is_stable=True
        entries = mpr.materials.summary.search(
            formula=norm_formula,
            fields=["material_id", "formula_pretty", "formation_energy_per_atom", "band_gap", "density", "structure", "is_stable"]
        )
        # 3. If still no results, try chemsys search (e.g., Fe-O)
        if not entries:
            elements = sorted(set(Composition(formula).elements))
            chemsys = "-".join([el.symbol for el in elements])
            entries = mpr.materials.summary.search(
                chemsys=chemsys,
                fields=["material_id", "formula_pretty", "formation_energy_per_atom", "band_gap", "density", "structure"]
            )
        if not entries:
            return None
        # Return entry with lowest formation energy
        return sorted(entries, key=lambda x: getattr(x, "formation_energy_per_atom", 9999))[0]
    except Exception as e:
        st.error(f"🚫 MP query failed for {formula}: {e}")
        return None


# ======================================================
# PINECONE UPSERT UTILITY
# ======================================================
def upsert_query_and_materials(query: str, materials: list):
    """Upsert a query and its materials to Pinecone vector DB."""
    embedding = get_query_embedding(query)
    item_id = str(uuid.uuid4())
    metadata = {"query": query, "materials": materials}
    index.upsert(vectors=[{"id": item_id, "values": embedding, "metadata": metadata}])
    return item_id


# ======================================================
# PINECONE SIMILARITY SEARCH UTILITY
# ======================================================
def search_similar_queries(query: str, top_k: int = 3, threshold: float = 0.8):
    """Search Pinecone for similar queries using vector embedding."""
    embedding = get_query_embedding(query)
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    matches = [m for m in getattr(results, "matches", []) if getattr(m, "score", 0) >= threshold]
    return [(m.score, m.metadata.get("query"), m.metadata.get("materials")) for m in matches]


# ======================================================
# QUERY EMBEDDING UTILITY
# ======================================================
def get_query_embedding(query: str):
    """Generate embedding for a query, removing generic words for focus."""
    # Preprocess: remove generic words to focus on requirements
    generic_words = [
        "material", "materials", "with", "a", "an", "the", "of", "for", "is", "are", "as", "and"
    ]
    query_words = query.split()
    filtered_words = [w for w in query_words if w.lower() not in generic_words]
    filtered_query = " ".join(filtered_words)
    try:
        response = openai.embeddings.create(
            input=filtered_query,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return [0.0] * 1024


# ======================================================
# API KEY VALIDATION
# ======================================================
if not OPENAI_API_KEY or not MATERIALS_PROJECT_API_KEY:
    st.error("❌ One or more API keys are missing from `.env`.")
    st.stop()


# ======================================================
# API CLIENT INITIALIZATION
# ======================================================
openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)
mpr = MPRester(MATERIALS_PROJECT_API_KEY)


# ======================================================
# LOGGING UTILITY
# ======================================================
logging.basicConfig(level=logging.INFO)
def log(msg):
    # Only print and log, do not show in Streamlit UI
    print(f"[DEBUG] {msg}")
    logging.info(msg)


# ======================================================
# LLM GENERATION FUNCTIONS
# ======================================================
def generate_openai_formulas(user_prompt):
    """Generate a list of chemical formulas using OpenAI LLM, present in Materials Project (stable or unstable)."""
    try:
        system_prompt = (
            "You are a materials scientist. "
            "Reply ONLY with a Python list of valid chemical formulas. "
            "Suggest the best materials from the Materials Project database that are highly compliant with the user query and are present in the Materials Project database, regardless of their stability. "
            "Each formula must be present in The Materials Project DataBase. "
            "Suggested materials must be highly compliant for the user query. "
            "You may include both stable and unstable materials . Unstable materials are those which is having positive formation energy. While Stable materials are those which is having negative or zero formation energy. "
        )
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Suggest chemical formulas for: {user_prompt}"}
            ]
        )
        raw_content = response.choices[0].message.content
        print(f"[DEBUG] Raw LLM response: {raw_content}")
        logging.info(f"[DEBUG] Raw LLM response: {raw_content}")
        return extract_formulas(raw_content, source="OpenAI")
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return []

def evaluate_openai_materials(material_list, user_prompt):
    # (Optional) Evaluate or re-rank LLM materials for compliance. Currently returns all.
    return material_list

# ======================================================
# STRUCTURE CONVERSION UTILITY
# ======================================================
def structure_to_cif_and_poscar(structure, supercell=(2, 2, 2)):
    """Convert a structure to CIF and POSCAR formats for download."""
    try:
        try:
            conventional = structure.get_conventional_standard_structure()
        except Exception:
            conventional = structure
        big = conventional * supercell
        cif = big.to(fmt="cif")
        try:
            poscar = big.to(fmt="poscar")
        except Exception:
            poscar = None
        return cif, poscar
    except Exception:
        return None, None

# ======================================================
# MATERIAL SELECTION LOGIC
# ======================================================
def find_best_material(formulas):
    """Filter and select the top 3 materials from Materials Project (stable or unstable)."""
    candidates = []
    filtered_out = []
    for formula in formulas:
        result = query_mp_for_formula(formula)
        if result:
            candidates.append(result)
        else:
            filtered_out.append(formula)
            st.warning(f"⚠️ LLM also suggested for {formula}, but material is not present in the Materials Project database.")
    if filtered_out and candidates:
        st.info(f"The following suggestions were filtered out as they are not present in the Materials Project: {', '.join(filtered_out)}")
    if not candidates:
        st.error("No suitable material found in the Materials Project database.")
        return []
    # Sort by formation energy (ascending) and return top 3
    return sorted(candidates, key=lambda x: getattr(x, "formation_energy_per_atom", 9999))[:3]


# ======================================================
# STREAMLIT UI & USER INTERACTION
# ======================================================
st.set_page_config(page_title="Materials PredictAI", layout="wide")
st.title("🔬 Materials PredictAI")
st.markdown("Enter **Requirements to be present in material** and let AI suggest candidates from the Materials Project database.")




# --- User input for material requirements ---
user_query = st.text_input(
    "Enter your materials design goal:",
    placeholder="e.g., Materials for Artificial Bone Development or Generate Crystal Structure for NaCoPo4"
)


# --- Stability filter toggle (centered and prominent, now below info message) ---
st.markdown("<div style='display:flex; justify-content:center; margin-bottom:0;'>", unsafe_allow_html=True)
show_unstable = st.toggle(
    label="Allow passage of unstable materials, if suggested by LLM (turn off for only stable)",
    value=False,
    help="(EXPERIMENTAL) Turn ON to include unstable materials in results, if suggested by LLM (higher formation energy materials). Turn OFF to show only stable materials."
)
st.markdown("</div>", unsafe_allow_html=True)


# --- Heuristic: detect if user query has new constraints (numbers, properties, etc) ---
def query_has_new_constraints(query):
    constraint_keywords = [
        "less than", "greater than", "above", "below", "between", "filter", "range", "under", "over", "equal to", "not equal", "within", "outside", "at least", "at most", "minimum", "maximum", "limit", "threshold"
    ]
    property_keywords = [
        "band gap", "formation energy", "density", "conductivity", "hardness", "stability", "gap", "energy", "property", "thermal", "electrical", "magnetic", "optical", "permittivity", "permeability", "modulus", "strength", "toughness", "melting point", "boiling point", "temperature", "pressure", "voltage", "current", "resistance", "capacitance", "inductance", "charge", "mass", "weight", "volume", "area", "length", "thickness", "width", "height", "diameter", "radius", "size", "dimension", "lattice", "crystal", "structure", "symmetry", "space group", "mp id", "mpid", "id"
    ]
    unit_keywords = [
        "eV", "cm^3", "g/cm3", "g/cm^3", "K", "C", "F", "GPa", "MPa", "V", "A", "S/m", "ohm", "Ω", "nm", "μm", "mm", "m", "kg", "g", "mol", "atm", "bar", "Pa", "W", "J", "s", "min", "h", "day", "year"
    ]
    logic_keywords = ["and", "or", "not", "&&", "||", "!"]
    query_lc = query.lower()
    has_number = bool(re.search(r"\d", query))
    has_constraint = any(w in query_lc for w in constraint_keywords)
    has_property = any(w in query_lc for w in property_keywords)
    has_unit = any(w in query_lc for w in unit_keywords)
    has_logic = any(w in query_lc for w in logic_keywords)
    # If query contains numbers + (constraint/property/unit/logic), treat as new constraint
    return has_number and (has_constraint or has_property or has_unit or has_logic)


# --- Main button: Find Best Material ---
if st.button("Find Best Material"):
    if not user_query.strip():
        st.warning("⚠️ Please enter a valid materials design goal.")
    else:
    # --- Try to match query in Pinecone vector DB ---
        with st.spinner("Generating query embedding and searching vector DB..."):
            try:
                embedding = get_query_embedding(user_query)
                log(f"Query embedding: {embedding[:5]}... (len={len(embedding)})")
            except Exception as e:
                st.error(f"Error during embedding: {e}")
                log(f"Embedding error: {e}")
                embedding = None
            try:
                matches = search_similar_queries(user_query, top_k=3, threshold=0.8)
                log(f"Pinecone matches: {matches}")
            except Exception as e:
                st.error(f"Error during Pinecone search: {e}")
                log(f"Pinecone search error: {e}")
                matches = []
            relevant_match = matches and not query_has_new_constraints(user_query)

        if relevant_match:
            # --- Use pre-predicted knowledge base if match found ---
            st.success("✅ Query intentions matched with vector intelligence. Delivering optimized material candidates from pre-predicted knowledge base.")
            _, matched_query, matched_materials = matches[0]
            log(f"Matched materials: {matched_materials}")
            # Deduplicate and take top 3
            seen = set()
            final_materials = []
            for f in matched_materials:
                if f not in seen:
                    seen.add(f)
                    final_materials.append(f)
                if len(final_materials) == 3:
                    break
            log(f"Final deduped materials: {final_materials}")
        else:
            # --- Use LLM to generate new suggestions if query is new or has constraints ---
            st.info("❇️ User query is new or contains constraints not seen before. Generating candidate materials using LLM and updating the knowledge base.")
            with st.spinner("Generating suggestions with OpenAI LLM..."):
                try:
                    openai_formulas = generate_openai_formulas(user_query)
                    log(f"LLM returned formulas: {openai_formulas}")
                except Exception as e:
                    st.error(f"Error during LLM/materials generation: {e}")
                    log(f"LLM/materials generation error: {e}")
                    openai_formulas = []
            # Deduplicate and take top 3
            combined = []
            seen = set()
            for f in openai_formulas:
                if f not in seen:
                    seen.add(f)
                    combined.append(f)
                if len(combined) == 3:
                    break
            final_materials = combined
            log(f"Final deduped LLM materials: {final_materials}")
            try:
                upsert_query_and_materials(user_query, final_materials)
                log(f"Upserted to Pinecone: {user_query} -> {final_materials}")
            except Exception as e:
                st.error(f"Error during Pinecone upsert: {e}")
                log(f"Pinecone upsert error: {e}")
        if not final_materials:
            st.warning("No materials found for this query.")

    # --- Query Materials Project for properties of final materials ---
        with st.spinner("Querying Materials Project for properties..."):
            best_list = find_best_material(final_materials)

    # --- Display results table and structure viewers ---
        if best_list:
            st.subheader("🧬 Suggested Compounds")
            def get_stability_str(obj):
                is_stable = getattr(obj, 'is_stable', None)
                if is_stable is True:
                    return 'Stable'
                elif is_stable is False:
                    return 'Unstable'
                else:
                    return 'Unknown'
            # Use the toggle value to filter
            if show_unstable:
                filtered_list = best_list
            else:
                filtered_list = [m for m in best_list if getattr(m, 'is_stable', None) is True]
            table_data = [
                {
                    "#": idx + 1,
                    "Formula": getattr(best, "formula_pretty", "N/A"),
                    "MP ID": getattr(best, "material_id", "N/A"),
                    "Formation Energy (eV/atom)": f"{getattr(best, 'formation_energy_per_atom', 0.0):.3f}",
                    "Band Gap (eV)": f"{getattr(best, 'band_gap', 0.0):.3f}",
                    "Density (g/cm³)": f"{getattr(best, 'density', 0.0):.3f}",
                    "Stability": get_stability_str(best),
                }
                for idx, best in enumerate(filtered_list)
            ]
            st.table(table_data)
            import csv
            from io import StringIO
            csv_buf = StringIO()
            if table_data:
                w = csv.DictWriter(csv_buf, fieldnames=list(table_data[0].keys()))
                w.writeheader()
                w.writerows(table_data)
            csv_bytes = csv_buf.getvalue().encode("utf-8")
            st.markdown(make_data_download_link(csv_bytes, "top3_materials.csv", mime="text/csv", label="Download table CSV"), unsafe_allow_html=True)
            VIEWER_IFRAME_HEIGHT = 500
            st.subheader("🔍 Structure Viewers")
            cols = st.columns(min(3, len(best_list)))
            for idx, best in enumerate(best_list):
                col = cols[idx] if idx < len(cols) else st.container()
                with col:
                    safe_formula = html.escape(str(getattr(best, "formula_pretty", "N/A")))
                    safe_mid = html.escape(str(getattr(best, "material_id", "N/A")))
                    ef_val = getattr(best, "formation_energy_per_atom", 0.0)
                    bg_val = getattr(best, "band_gap", 0.0)
                    st.markdown(
                        f"""
                        <div style='background:#181c24; border-radius:12px; padding:10px; margin-bottom:8px; border:1px solid #23272f; text-align:center;'>
                          <div style='font-weight:700; color:#00c3ff; font-size:16px;'>🧬 {safe_formula}</div>
                          <div style='font-size:12px; color:#b0b8c1;'>MP ID: {safe_mid} | E_f: {ef_val:.3f} eV/atom | Band Gap: {bg_val:.3f} eV</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    try:
                        viewer_key = f"viewer_html_{getattr(best, 'material_id', 'N/A')}"
                        existing = st.session_state.get(viewer_key)
                        if existing and ("height_slider_" in existing or "viewer_container_" in existing or "height_value_" in existing):
                            viewer = show_structure(getattr(best, "structure", None))
                            if viewer is not None:
                                raw_html = viewer._make_html()
                                st.session_state[viewer_key] = raw_html
                        elif not existing:
                            viewer = show_structure(getattr(best, "structure", None))
                            if viewer is not None:
                                raw_html = viewer._make_html()
                                st.session_state[viewer_key] = raw_html
                        viewer_html = st.session_state.get(viewer_key)
                        if viewer_html:
                            components.html(viewer_html, height=VIEWER_IFRAME_HEIGHT)
                        else:
                            st.error(f"No viewer HTML for {safe_formula}")
                    except Exception as e:
                        st.error(f"Error rendering structure for {safe_formula}: {e}")
                    cif_str, poscar_str = structure_to_cif_and_poscar(getattr(best, "structure", None))
                    dlc1, dlc2 = st.columns([1, 1])
                    def sanitize_filename(name: str) -> str:
                        return re.sub(r"[^A-Za-z0-9._-]", "_", name)
                    if cif_str:
                        with dlc1:
                            fname = sanitize_filename(f"{getattr(best, 'formula_pretty', 'N/A')}.cif")
                            label = f"CIF ({html.escape(str(getattr(best, 'formula_pretty', 'N/A')))} )"
                            st.markdown(make_data_download_link(cif_str, fname, mime="chemical/x-cif", label=label), unsafe_allow_html=True)
                    if poscar_str:
                        with dlc2:
                            fname = sanitize_filename(f"{getattr(best, 'formula_pretty', 'N/A')}_POSCAR.vasp")
                            label = f"POSCAR ({html.escape(str(getattr(best, 'formula_pretty', 'N/A')))} )"
                            st.markdown(make_data_download_link(poscar_str, fname, mime="text/plain", label=label), unsafe_allow_html=True)
        else:
            st.error("No suitable material found.")

# --- Initial info message ---
if not user_query:
    st.info("👆 Enter a goal above and click **Find Best Material/Compound** to get started.")

            # --- Legal & Disclaimer Section ---
st.markdown("""
---
**Third-Party Terms & Conditions**

- By using Material PredictAI, you agree to the [OpenAI Terms of Service](https://openai.com/policies/terms-of-use), [Materials Project Terms of Use](https://materialsproject.org/about/terms), and [Pinecone Terms of Service](https://www.pinecone.io/terms/).


**Disclaimer**

- This software is distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). 
- Use of this software does not imply any endorsement by IIT BHU. Please cite appropriately in academic or research work.
- This application is for research and educational purposes only. All data and predictions are provided “as is” without warranty of any kind.           
            
Developed at:  
**School of Materials Science & Technology**  
**Indian Institute of Technology (IIT BHU), Varanasi**.  
            
""")
