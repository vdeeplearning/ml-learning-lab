import streamlit as st

from ml_learning_lab.content import LESSONS


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --lab-border: #d8dee9;
            --lab-muted: #64748b;
            --lab-ink: #172033;
            --lab-panel: #f8fafc;
        }
        .block-container {
            padding-top: 4rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--lab-ink);
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--lab-border);
        }
        .lab-eyebrow {
            color: #0f766e;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            line-height: 1.35;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .lab-hero {
            border-bottom: 1px solid var(--lab-border);
            margin-top: 0.75rem;
            padding-top: 0.75rem;
            padding-bottom: 1.1rem;
            margin-bottom: 1.25rem;
            overflow: visible;
        }
        .lab-hero p {
            max-width: 760px;
            color: var(--lab-muted);
            font-size: 1.08rem;
            line-height: 1.6;
        }
        .lab-card {
            border: 1px solid var(--lab-border);
            border-radius: 8px;
            padding: 1rem;
            background: white;
            margin-bottom: 0.75rem;
        }
        .lab-card h3 {
            font-size: 1.05rem;
            margin-bottom: 0.35rem;
        }
        .lab-card p {
            color: var(--lab-muted);
            margin-bottom: 0.6rem;
        }
        .lab-tag {
            display: inline-flex;
            border: 1px solid #99f6e4;
            border-radius: 999px;
            padding: 0.16rem 0.55rem;
            background: #f0fdfa;
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 650;
        }
        .lab-stat {
            border: 1px solid var(--lab-border);
            border-radius: 8px;
            padding: 0.9rem;
            min-height: 116px;
            background: var(--lab-panel);
        }
        .lab-stat strong {
            display: block;
            color: var(--lab-muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .lab-stat span {
            display: block;
            color: var(--lab-ink);
            font-size: 1.7rem;
            font-weight: 750;
            margin: 0.15rem 0;
        }
        .lab-stat p {
            color: var(--lab-muted);
            margin: 0;
            font-size: 0.9rem;
            line-height: 1.35;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(active_page: str) -> None:
    with st.sidebar:
        st.markdown("## ML Learning Lab")
        st.caption("Interactive classical ML lessons")
        st.markdown("---")
        st.caption(f"Current page: {active_page}")


def page_header(eyebrow: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div style="height: 1.25rem;"></div>
        <section class="lab-hero">
            <div class="lab-eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{body}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def concept_card(title: str, body: str, tag: str, page: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="lab-card">
            <span class="lab-tag">{tag}</span>
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if page:
        st.page_link(page, label=f"Open {title}")


def stat_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="lab-stat">
            <strong>{label}</strong>
            <span>{value}</span>
            <p>{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def lesson_shell(title: str, objective: str, intuition: str) -> None:
    page_header("Lesson scaffold", title, objective)
    st.markdown("### Core intuition")
    st.write(intuition)

    st.markdown("### What this page will teach")
    st.write(
        "The full lesson will add controls, visual feedback, model behavior, and a short "
        "takeaway so users can experiment without getting lost in implementation details."
    )


def synthetic_feature_note() -> None:
    st.info(
        "This demo is a binary classification task. Each dot is one example with two input measurements: "
        "`Signal A` and `Signal B`. The color shows its label, either `Positive` or `Negative`. "
        "The model's job is to learn from labeled examples and predict the label for new examples."
    )
