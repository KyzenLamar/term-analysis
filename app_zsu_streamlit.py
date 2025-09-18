import streamlit as st
import tempfile
import os
import pandas as pd
import re
import csv

from term_analysis_core import (
    load_terms_dictionary_csv,
    analyze_document_v2,
    get_unmatched_sentences,
    export_results_to_csv,
)
# from ml_semantic_search import load_wrong_usages, semantic_search

# ----------- BRANDING / DESIGN -------------
st.markdown("""
<style>
@font-face {
    font-family: 'Volja';
    src: url('volja-regular_w.ttf') format('truetype');
    font-weight: normal;
    font-style: normal;
}
.stApp, body {
    background-color: #F5F5F2;
    font-family: 'Volja', Montserrat, Arial, sans-serif !important;
}
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #6A653A;
    font-family: 'Volja', Montserrat, Arial, sans-serif !important;
    font-weight: 700;
    letter-spacing: 0.03em;
}
.stButton > button {
    background-color: #6A653A;
    color: #fff;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1em;
    padding: 0.5em 2em;
    border: none;
    transition: background 0.3s;
    font-family: 'Volja', Montserrat, Arial, sans-serif !important;
}
.stButton > button:hover {
    background-color: #F39200;
    color: #232323;
}
.block-container, .css-18e3th9 {
    max-width: 70vw !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    border-radius: 16px;
    box-shadow: 0 2px 8px rgba(106,101,58, 0.08);
}
.stDataFrame, .stTable {
    background: #fff;
    border-radius: 10px;
    font-family: 'Volja', Montserrat, Arial, sans-serif !important;
}
.stDataFrame thead tr th {
    background-color: #6A653A !important;
    color: #fff !important;
}
.stDataFrame tbody tr td {
    background-color: #fff !important;
    color: #232323 !important;
}
.highlight-term {
    background-color: #F39200;
    color: #232323 !important;
    border-radius: 6px;
    padding: 2px 4px;
    font-weight: bold;
    font-family: 'Volja', Montserrat, Arial, sans-serif !important;
}
.footer {
    color: #6A653A;
    font-size: 0.9em;
    margin-top: 2em;
    text-align: center;
    font-family: 'Volja', Montserrat, Arial, sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# Логотип Сухопутних військ ЗСУ (SVG або PNG)
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:1em;">
        <h1 style="margin-bottom:0;">⚡️ Term Analysis Tool ⚡️</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- END BRANDING ------------

# --- AI (семантика) — ледаче завантаження (коли користувач увімкне)
from ml_semantic_search import load_wrong_usages, semantic_search

@st.cache_resource(show_spinner=False)
def load_ai_resources():
    """
    Завантажує модель + ембеддинги wrong_usage один раз, коли AI увімкнено.
    """
    wrong_csv_path = "all_generated_wrong_usages.csv"
    wrong_sentences, wrong_terms, wrong_comments, wrong_embeds = load_wrong_usages(wrong_csv_path)
    return wrong_sentences, wrong_terms, wrong_comments, wrong_embeds


def highlight_term_in_sentence(sentence, term):
    if not sentence or not term:
        return sentence
    pattern = r'\b{}\b'.format(re.escape(term))
    highlighted = re.sub(
        pattern,
        lambda m: f'<span class="highlight-term">{m.group(0)}</span>',
        sentence,
        flags=re.IGNORECASE
    )
    return highlighted

def save_terms_dictionary_csv(terms, file_path):
    fieldnames = ["approved_term", "synonyms", "category", "wrong_usages", "context_examples"]
    with open(file_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in terms:
            writer.writerow({
                "approved_term": t["approved_term"],
                "synonyms": ";".join(t["synonyms"]),
                "category": t["category"],
                "wrong_usages": ";".join(t["wrong_usages"]),
                "context_examples": ";".join(t["context_examples"])
            })

# --- ML Semantic Search: Load model & data (один раз при старті!)
# @st.cache_resource
# def get_semantic_search_model():
#     wrong_csv_path = "all_generated_wrong_usages.csv"  # шлях до твого CSV з wrong_usage
#     wrong_sentences, wrong_terms, wrong_comments, wrong_embeds = load_wrong_usages(wrong_csv_path)
#     return wrong_sentences, wrong_terms, wrong_comments, wrong_embeds
#
# wrong_sentences, wrong_terms, wrong_comments, wrong_embeds = get_semantic_search_model()

# ---- 1. Навігація між сторінками ----
page = st.sidebar.radio("Сторінка", ["Аналіз документів", "Редактор словника"])

# ---- 2. Параметри session_state ----
if "dict_path" not in st.session_state:
    st.session_state["dict_path"] = None
if "dict_df" not in st.session_state:
    st.session_state["dict_df"] = None
if "doc_path" not in st.session_state:
    st.session_state["doc_path"] = None
if "doc_name" not in st.session_state:
    st.session_state["doc_name"] = None
if "results" not in st.session_state:
    st.session_state["results"] = None
if "unmatched_sentences" not in st.session_state:
    st.session_state["unmatched_sentences"] = None

# ---- 3. АНАЛІЗ ДОКУМЕНТІВ ----
if page == "Аналіз документів":
    st.write(
        "Автоматичний аналіз документів на відповідність затвердженим термінам та стандартизованим назвам. "
        "Завантаж .docx або .pdf, вибери словник — отримай миттєвий звіт про помилки термінології!"
    )

    with st.sidebar:
        st.header("Параметри")
        synonyms_file = st.file_uploader("Завантаж словник синонімів", type=["csv"])
        fuzzy_threshold = st.slider("Поріг fuzzy matching (Левенштейн)", 60, 100, 88)
        semantic_threshold = st.slider("Поріг Semantic Search (0.7-0.95)", 0.7, 0.95, 0.8, step=0.01)
        st.markdown("**Рекомендація:** 88 — оптимально для військових термінів")
        ai_enabled = st.toggle("Увімкнути AI-пошук (повільніше)", value=False,
                               help="Семантичний пошук на основі моделі з Hugging Face. Може завантажуватися довше на холодному старті.")

    # --- Завантаження/підвантаження словника
    if synonyms_file:
        temp_dict_path = f"uploaded_{synonyms_file.name}"
        with open(temp_dict_path, "wb") as f:
            f.write(synonyms_file.read())
        st.session_state["dict_path"] = temp_dict_path
        terms = load_terms_dictionary_csv(temp_dict_path)
        df_dict = pd.DataFrame([{
            "approved_term": t["approved_term"],
            "synonyms": "; ".join(t["synonyms"]),
            "category": t["category"],
            "wrong_usages": "; ".join(t["wrong_usages"]),
            "context_examples": "; ".join(t["context_examples"])
        } for t in terms])
        st.session_state["dict_df"] = df_dict
        st.success(f"Словник підвантажено: {synonyms_file.name}")
    elif st.session_state["dict_path"] is not None:
        st.info(f"Використовується словник: {os.path.basename(st.session_state['dict_path'])}")

    # --- Завантаження/підвантаження документа
    uploaded_file = st.file_uploader("Завантаж документ (.docx або .pdf)", type=["docx", "pdf"])
    if uploaded_file:
        temp_doc_path = f"uploaded_{uploaded_file.name}"
        with open(temp_doc_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state["doc_path"] = temp_doc_path
        st.session_state["doc_name"] = uploaded_file.name
        st.success(f"Документ підвантажено: {uploaded_file.name}")
    elif st.session_state["doc_path"] is not None:
        st.info(f"Використовується документ: {st.session_state['doc_name']}")

    analyze_btn = st.button("Запустити аналіз", use_container_width=True)
    if analyze_btn:
        if st.session_state["doc_path"] is None or st.session_state["dict_path"] is None:
            st.error("Завантаж і документ, і словник!")
        else:
            with st.spinner("Аналізуємо документ..."):
                terms = load_terms_dictionary_csv(st.session_state["dict_path"])
                results = analyze_document_v2(st.session_state["doc_path"], terms, fuzzy_threshold=fuzzy_threshold)
                # get_unmatched_sentences повинна повертати [{"page": ..., "sentence": ...}]
                unmatched_sentences = get_unmatched_sentences(st.session_state["doc_path"], terms, fuzzy_threshold)
                st.session_state["results"] = results
                st.session_state["unmatched_sentences"] = unmatched_sentences

    results = st.session_state.get("results", [])
    unmatched_sentences = st.session_state.get("unmatched_sentences", [])

    # --- Rule-based results
    if results:
        st.warning(f"Знайдено {len(results)} випадків некоректного застосування термінів!")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
            export_results_to_csv(results, tmp_csv.name)
            tmp_csv.flush()
            with open(tmp_csv.name, "rb") as f:
                st.download_button(
                    label="⬇️ Завантажити звіт (.csv)",
                    data=f,
                    file_name="term_analysis_report.csv",
                    mime="text/csv",
                )
        st.subheader("Виявлені контексти (з підсвіченням):")
        for i, row in df.iterrows():
            st.markdown(
                f'<b>Сторінка:</b> <span style="color:#6A653A;font-weight:bold;">{row.get("page", "-")}</span> | '
                f'<b>Термін:</b> <span style="color:#1976d2">{row["found_term"]}</span> | '
                f'<b>Контекст:</b> {highlight_term_in_sentence(row["context"], row["found_term"])}',
                unsafe_allow_html=True
            )

    # --- SEMANTIC SEARCH (AI пошук помилок) ---
    st.subheader("🔎 Семантичний аналіз (AI пошук помилок)")
    if not ai_enabled:
        st.info(
            "AI-пошук вимкнено. Увімкни перемикач у сайдбарі, якщо хочеш доповнити rule-based перевірку семантичним аналізом.")
    else:
        st.caption("⚠️ Перший запуск може бути довшим — модель кешується.")
        if unmatched_sentences:
            with st.status(
                    "Завантажуємо AI-ресурси (модель + ембеддинги). Це може зайняти 20–60 сек. на холодному старті…",
                    expanded=True):
                wrong_sentences, wrong_terms, wrong_comments, wrong_embeds = load_ai_resources()
            semantic_results = []
            with st.spinner("Проводимо семантичний аналіз…"):
                for unmatched in unmatched_sentences:
                    sentence = unmatched["sentence"] if isinstance(unmatched, dict) else unmatched
                    page = unmatched["page"] if isinstance(unmatched, dict) else "-"
                    hits = semantic_search(
                        sentence,
                        wrong_sentences,
                        wrong_terms,
                        wrong_comments,
                        wrong_embeds,
                        topn=5,
                        threshold=semantic_threshold
                    )
                    if hits:
                        for match in hits:
                            semantic_results.append({
                                "page": page,
                                "sentence": sentence,
                                "approved_term": match["approved_term"],
                                "wrong_usage": match["wrong_usage"],
                                "comment": match["comment"],
                                "score": match["score"]
                            })
            if semantic_results:
                st.success(f"Знайдено {len(semantic_results)} потенційних семантичних помилок!")
                df_sem = pd.DataFrame(semantic_results)
                st.dataframe(df_sem, use_container_width=True)
                st.subheader("Виявлені AI-помилки (контексти):")
                for _, row in df_sem.iterrows():
                    st.markdown(
                        f'<b>Сторінка:</b> <span style="color:#6A653A;font-weight:bold;">{row.get("page", "-")}</span> | '
                        f'<b>Вхідне речення:</b> <span style="color:#1976d2">{row["sentence"]}</span><br>'
                        f'<b>Схожий приклад помилки:</b> <span class="highlight-term">{row["wrong_usage"]}</span> '
                        f'(<b>Термін:</b> {row["approved_term"]}, <b>Схожість:</b> <b>{row["score"]}</b>, <b>Коментар:</b> {row["comment"]})',
                        unsafe_allow_html=True
                    )
                # Download CSV
                csv_sem = df_sem.to_csv(index=False, encoding="utf-8")
                st.download_button(
                    label="⬇️ Завантажити AI semantic звіт (.csv)",
                    data=csv_sem,
                    file_name="ai_semantic_report.csv",
                    mime="text/csv"
                )
            else:
                st.info("Семантичних помилок не виявлено на вибраному порозі.")

    # Очищення тільки результатів
    if st.button("Очистити результати"):
        st.session_state["results"] = None
        st.session_state["unmatched_sentences"] = None
        st.rerun()
    # Очищення всіх файлів і результатів
    if st.button("Очистити все (словник, документ, результати)"):
        st.session_state["dict_path"] = None
        st.session_state["dict_df"] = None
        st.session_state["doc_path"] = None
        st.session_state["doc_name"] = None
        st.session_state["results"] = None
        st.session_state["unmatched_sentences"] = None
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
        Застосунок створено для Сухопутних військ Збройних Сил України.<br>
        Made with 💙💛 by [Твій помічник Igor🧔🏻‍]
        </div>
        """,
        unsafe_allow_html=True
    )

# ---- 4. РЕДАКТОР СЛОВНИКА ----
elif page == "Редактор словника":
    st.header("📖 Редактор словника термінів")

    if st.session_state.get("dict_df") is not None:
        df_dict = st.session_state["dict_df"]
        dict_path = st.session_state["dict_path"]
        st.success(f"Словник підвантажено з Аналізу: {os.path.basename(dict_path)}")
    else:
        dict_file = st.file_uploader("Завантаж словник термінів (CSV)", type=["csv"])
        dict_path = None
        df_dict = None
        if dict_file:
            dict_path = f"uploaded_{dict_file.name}"
            with open(dict_path, "wb") as f:
                f.write(dict_file.read())
            terms = load_terms_dictionary_csv(dict_path)
            df_dict = pd.DataFrame([{
                "approved_term": t["approved_term"],
                "synonyms": "; ".join(t["synonyms"]),
                "category": t["category"],
                "wrong_usages": "; ".join(t["wrong_usages"]),
                "context_examples": "; ".join(t["context_examples"])
            } for t in terms])
            st.session_state["dict_df"] = df_dict
            st.session_state["dict_path"] = dict_path

    if st.session_state.get("dict_df") is not None:
        st.subheader("Редагуй словник прямо у таблиці:")
        edited_df = st.data_editor(
            st.session_state["dict_df"],
            num_rows="dynamic",
            key="edit_dict"
        )
        if st.button("💾 Зберегти зміни у CSV"):
            terms = []
            for i, row in edited_df.iterrows():
                terms.append({
                    "approved_term": str(row["approved_term"]).strip(),
                    "synonyms": [s.strip() for s in str(row["synonyms"]).split(';') if s.strip()],
                    "category": str(row.get("category", "")).strip(),
                    "wrong_usages": [s.strip() for s in str(row.get("wrong_usages", "")).split(';') if s.strip()],
                    "context_examples": [s.strip() for s in str(row.get("context_examples", "")).split(';') if s.strip()],
                })
            save_terms_dictionary_csv(terms, st.session_state["dict_path"])
            st.session_state["dict_df"] = edited_df
            st.success("Зміни збережено у CSV!")

        st.markdown("---")
        st.subheader("Завантажити оновлений словник:")
        with open(st.session_state["dict_path"], "rb") as f:
            st.download_button(
                label="⬇️ Завантажити CSV",
                data=f,
                file_name=f"updated_{os.path.basename(st.session_state['dict_path'])}",
                mime="text/csv",
            )