import streamlit as st

from ml_learning_lab.content import LESSONS
from ml_learning_lab.plots import learning_map
from ml_learning_lab.ui import (
    apply_global_styles,
    concept_card,
    page_header,
    render_sidebar,
    stat_card,
)


st.set_page_config(
    page_title="ML Learning Lab",
    page_icon="ML",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()
render_sidebar(active_page="Home")

page_header(
    eyebrow="Interactive classical machine learning",
    title="ML Learning Lab",
    body=(
        "A lightweight learning lab for building intuition about classical ML "
        "through visual explanations, controlled experiments, and plain-English takeaways."
    ),
)

left, right = st.columns([1.08, 0.92], gap="large")

with left:
    st.subheader("Start with the mental model")
    st.write(
        "Each lesson will pair a compact explanation with sliders, model behavior, "
        "and visual feedback. The first version establishes the product shell and "
        "a consistent lesson structure."
    )

    cards = st.columns(3)
    with cards[0]:
        stat_card("Lessons", "2", "Tree-based learning modules")
    with cards[1]:
        stat_card("Focus", "Intuition", "Concepts before equations")
    with cards[2]:
        stat_card("Stack", "Streamlit", "Python, sklearn, Plotly")

    st.markdown("### Current Lesson Tracks")
    for lesson in LESSONS:
        concept_card(
            title=lesson.title,
            body=lesson.summary,
            tag=lesson.status,
            page=lesson.page,
        )

with right:
    st.plotly_chart(learning_map(), use_container_width=True)
    st.info(
        "Next build step: add the first live Decision Trees experiment with depth, "
        "split criteria, and overfitting visuals."
    )
