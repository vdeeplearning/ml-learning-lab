import streamlit as st

from ml_learning_lab.content import lesson_by_title
from ml_learning_lab.models import make_tree_overfitting_dataset, train_decision_tree
from ml_learning_lab.plots import (
    decision_tree_regions,
    tree_accuracy_curve,
    tree_depth_concept_diagram,
    tree_route_diagram,
    tree_training_process_diagram,
)
from ml_learning_lab.ui import apply_global_styles, lesson_shell, render_sidebar, synthetic_feature_note


st.set_page_config(page_title="Decision Trees | ML Learning Lab", page_icon="ML", layout="wide")
apply_global_styles()
render_sidebar(active_page="Decision Trees")

lesson = lesson_by_title("Decision Trees")
lesson_shell(lesson.title, lesson.objective, lesson.intuition)
synthetic_feature_note()
st.write(
    "In the plots, the horizontal axis is `Signal A` and the vertical axis is `Signal B`. "
    "A decision tree learns a sequence of yes/no questions about these two signals, then uses "
    "the answers to decide whether a new point should be classified as Positive or Negative."
)

dataset = make_tree_overfitting_dataset()
depth_scores = [
    {
        "Depth": depth,
        "Train accuracy": train_decision_tree(dataset, max_depth=depth).train_score,
        "Test accuracy": train_decision_tree(dataset, max_depth=depth).test_score,
    }
    for depth in range(1, 11)
]

left, right = st.columns([0.95, 1.05], gap="large")
with left:
    st.subheader("Experiment")
    max_depth = st.slider(
        "Tree depth / levels",
        min_value=1,
        max_value=10,
        value=3,
        help="Higher depth lets the tree ask more follow-up questions.",
    )
    result = train_decision_tree(dataset, max_depth=max_depth)

    metric_cols = st.columns(2)
    metric_cols[0].metric("Train accuracy", f"{result.train_score:.1%}")
    metric_cols[1].metric("Test accuracy", f"{result.test_score:.1%}")

    st.write(
        f"Depth `{max_depth}` controls how many layers of questions the tree can ask. "
        "Training accuracy should usually rise as depth increases, but test accuracy can flatten "
        "or fall once the tree starts memorizing noisy details."
    )

    gap = result.train_score - result.test_score
    if gap > 0.08:
        st.warning(
            "The training score is pulling away from the test score. That is the overfitting pattern: "
            "the tree is getting better at the examples it saw, but less reliably generalizing."
        )
    else:
        st.success(
            "The train and test scores are still close. This is the useful zone where the tree has "
            "learned structure without memorizing too much noise."
        )

    st.caption(
        "Try moving from depth 1 to 10. The colored regions get more jagged, training accuracy climbs, "
        "and test accuracy eventually stops improving."
    )

with right:
    st.plotly_chart(
        decision_tree_regions(dataset, result.model, f"Decision regions at depth {max_depth}"),
        use_container_width=True,
    )

lower_left, lower_right = st.columns([1, 1], gap="large")
with lower_left:
    st.plotly_chart(tree_accuracy_curve(depth_scores, max_depth), use_container_width=True)

with lower_right:
    tracked_example = dataset.test_features.iloc[0]
    st.plotly_chart(tree_route_diagram(result.model, tracked_example), use_container_width=True)
    st.caption(
        "This is part of the actual trained tree. It can look uneven because the model only grows "
        "branches where the training data suggests another split is useful."
    )

st.markdown("### What depth means")
depth_left, depth_right = st.columns([0.9, 1.1], gap="large")
with depth_left:
    st.write(
        "Depth is the number of question layers from the root of the tree to a leaf. "
        "Depth `1` means the tree asks one question and makes two broad groups. "
        "Depth `5` means it can ask follow-up questions inside follow-up questions."
    )
    st.write(
        "More depth gives the model more chances to carve the data into small regions. "
        "That usually helps training accuracy, but it can also make the tree chase noise."
    )
    st.write(
        "The diagram on the right is intentionally symmetrical because it is a concept diagram: "
        "it shows the maximum possible shape if every branch kept splitting. A real trained tree "
        "does not have to fill every possible branch."
    )
with depth_right:
    st.plotly_chart(tree_depth_concept_diagram(max_depth), use_container_width=True)

st.markdown("### How training works")
training_left, training_right = st.columns([1.05, 0.95], gap="large")
with training_left:
    st.plotly_chart(tree_training_process_diagram(), use_container_width=True)
with training_right:
    st.write(
        "The tree structure is not preset. During training, the tree starts with all labeled examples "
        "at the root and searches for a question that best separates them, such as `Signal A <= 0.42`."
    )
    st.write(
        "After the first split, it repeats the same search inside each branch. Each new split tries "
        "to make the resulting groups more pure: mostly Positive on one side, mostly Negative on the other."
    )
    st.write(
        "The depth slider sets a stopping rule. A shallow tree stops early and keeps broad patterns. "
        "A deep tree keeps splitting until it can explain tiny pockets of the training data."
    )
    st.write(
        "That is why learned trees are often asymmetrical: one side of the data may need several "
        "follow-up questions, while another side may already be pure enough to stop."
    )
