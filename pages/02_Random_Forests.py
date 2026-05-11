import streamlit as st

from ml_learning_lab.content import lesson_by_title
from ml_learning_lab.models import make_forest_voting_dataset, random_forest_score_path, train_random_forest
from ml_learning_lab.plots import (
    accuracy_curve,
    classification_regions,
    feature_importance_bar,
    forest_size_concept_diagram,
    forest_training_process_diagram,
    forest_vote_diagram,
)
from ml_learning_lab.ui import apply_global_styles, lesson_shell, render_sidebar, synthetic_feature_note


st.set_page_config(page_title="Random Forests | ML Learning Lab", page_icon="ML", layout="wide")
apply_global_styles()
render_sidebar(active_page="Random Forests")

lesson = lesson_by_title("Random Forests")
lesson_shell(lesson.title, lesson.objective, lesson.intuition)
synthetic_feature_note()
st.write(
    "In the plots, the horizontal axis is `Signal A` and the vertical axis is `Signal B`. "
    "A random forest trains many decision trees on variations of the labeled data, then lets "
    "those trees vote on whether a new point is Positive or Negative."
)

dataset = make_forest_voting_dataset()


@st.cache_data(show_spinner=False)
def forest_accuracy_scores() -> list[dict[str, float]]:
    return random_forest_score_path(make_forest_voting_dataset(), max_estimators=100)


forest_scores = forest_accuracy_scores()


def select_split_vote_example(model):
    if len(model.estimators_) < 9:
        return dataset.test_features.iloc[0]

    best_sample = dataset.test_features.iloc[0]
    best_distance = 9
    for _, sample in dataset.test_features.iterrows():
        sample_array = sample.to_numpy().reshape(1, -1)
        votes = [int(tree.predict(sample_array)[0]) for tree in model.estimators_[:9]]
        majority_votes = max(sum(votes), 9 - sum(votes))
        distance = abs(majority_votes - 6)
        if distance < best_distance:
            best_sample = sample
            best_distance = distance
        if majority_votes == 6:
            break
    return best_sample

left, right = st.columns([0.95, 1.05], gap="large")
with left:
    st.subheader("Experiment")
    trees = st.slider(
        "Number of trees",
        min_value=1,
        max_value=100,
        value=40,
        step=1,
        help="More trees usually make the forest steadier, though each update takes slightly more work.",
    )
    result = train_random_forest(dataset, n_estimators=trees)

    metric_cols = st.columns(2)
    metric_cols[0].metric("Train accuracy", f"{result.train_score:.1%}")
    metric_cols[1].metric("Test accuracy", f"{result.test_score:.1%}")

    st.write(
        f"A forest with `{trees}` trees will let learners compare one flexible model against "
        "many diverse models voting together. Moving the slider retrains the forest, redraws "
        "the regions, and changes how many votes are averaged."
    )
    if trees < 10:
        st.warning("With very few trees, a forest can behave like a small committee with unstable opinions.")
    else:
        st.success("As more trees vote, the boundary usually stabilizes and test accuracy tends to plateau.")

    st.plotly_chart(feature_importance_bar(result.model), use_container_width=True)

with right:
    st.plotly_chart(
        classification_regions(dataset, result.model, f"Forest decision regions with {trees} trees"),
        use_container_width=True,
    )

lower_left, lower_right = st.columns([1, 1], gap="large")
with lower_left:
    st.plotly_chart(
        accuracy_curve(
            forest_scores,
            x_key="Trees",
            selected_x=trees,
            title="More trees: noisy at first, then stable",
            x_title="Number of trees",
        ),
        use_container_width=True,
    )
    st.caption(
        "Train accuracy is usually higher because it is measured on examples the forest learned from. "
        "Test accuracy is measured on held-out examples the model did not train on, so it is the better "
        "signal for how well the forest generalizes."
    )

with lower_right:
    st.plotly_chart(forest_vote_diagram(result.model, select_split_vote_example(result.model)), use_container_width=True)
    st.caption(
        "This voting graphic shows a small sample of trees from the trained forest. The individual "
        "trees are learned from data and may have different, uneven structures."
    )

st.markdown("### What number of trees means")
concept_left, concept_right = st.columns([0.9, 1.1], gap="large")
with concept_left:
    st.write(
        "A random forest is a group of decision trees trained with slightly different views of the data. "
        "The `Number of trees` slider controls how many individual trees join the vote."
    )
    st.write(
        "One tree can be sensitive to quirks in its training sample. Many trees can cancel out some of "
        "that randomness because the final prediction is based on a vote instead of one path."
    )
    st.write(
        "The diagram on the right is a simplified count of trees, not the exact shapes of those trees. "
        "Real trees inside the forest are learned from data and are often asymmetrical."
    )
with concept_right:
    st.plotly_chart(forest_size_concept_diagram(trees), use_container_width=True)

st.markdown("### How training works")
training_left, training_right = st.columns([1.05, 0.95], gap="large")
with training_left:
    st.plotly_chart(forest_training_process_diagram(), use_container_width=True)
with training_right:
    st.write(
        "During training, each tree gets a bootstrapped sample: a random sample of rows where some "
        "examples may appear more than once and some may be left out."
    )
    st.write(
        "Each tree learns splits like a normal decision tree, but the forest introduces randomness so "
        "the trees do not all make identical mistakes."
    )
    st.write(
        "At prediction time, the trees vote. Adding trees often improves stability early, then the test "
        "accuracy usually plateaus because the forest has already captured the main signal."
    )
    st.write(
        "So the forest is not a preset collection of identical trees. It is an ensemble of learned trees, "
        "each shaped by the sample and split choices it saw during training."
    )
