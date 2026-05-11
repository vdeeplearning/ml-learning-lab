import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from ml_learning_lab.config import PRIMARY_ACCENT, SECONDARY_ACCENT
from ml_learning_lab.models import DatasetBundle, make_intro_classification_dataset


def base_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=12, r=12, t=56 if title else 20, b=12),
        font=dict(family="Inter, Segoe UI, sans-serif", size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def learning_map() -> go.Figure:
    data = pd.DataFrame(
        {
            "Topic": ["Data", "Model", "Experiment", "Takeaway"],
            "Progress": [1, 2, 3, 4],
            "Emphasis": [22, 34, 30, 24],
            "Color": ["#475569", "#0f766e", "#7c3aed", "#334155"],
        }
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["Progress"],
            y=[1, 1.35, 1.1, 1.5],
            mode="lines+markers+text",
            line=dict(color="#334155", width=3),
            marker=dict(size=data["Emphasis"], color=data["Color"], line=dict(color="white", width=3)),
            text=data["Topic"],
            textposition="bottom center",
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )
    fig.update_xaxes(visible=False, range=[0.65, 4.35])
    fig.update_yaxes(visible=False, range=[0.55, 1.85])
    return base_layout(fig, "How each lesson is structured")


def sample_dataset_scatter(title: str = "Sample classification playground") -> go.Figure:
    dataset = make_intro_classification_dataset()
    frame = dataset.features.copy()
    frame["Class"] = dataset.target
    fig = px.scatter(
        frame,
        x="Signal A",
        y="Signal B",
        color="Class",
        color_discrete_map={"Positive": PRIMARY_ACCENT, "Negative": SECONDARY_ACCENT},
    )
    fig.update_traces(marker=dict(size=10, opacity=0.86, line=dict(color="white", width=0.8)))
    return base_layout(fig, title)


def classification_regions(dataset: DatasetBundle, model, title: str) -> go.Figure:
    x_min, x_max = dataset.features["Signal A"].min() - 0.7, dataset.features["Signal A"].max() + 0.7
    y_min, y_max = dataset.features["Signal B"].min() - 0.7, dataset.features["Signal B"].max() + 0.7
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 180), np.linspace(y_min, y_max, 180))
    grid = pd.DataFrame({"Signal A": xx.ravel(), "Signal B": yy.ravel()})
    predictions = model.predict(grid).reshape(xx.shape)
    encoded = np.where(predictions == "Positive", 1, 0)

    frame = dataset.features.copy()
    frame["Class"] = dataset.target
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, 180),
            y=np.linspace(y_min, y_max, 180),
            z=encoded,
            colorscale=[[0, "rgba(245, 158, 11, 0.16)"], [1, "rgba(37, 99, 235, 0.16)"]],
            contours=dict(start=0, end=1, size=0.5, coloring="heatmap"),
            showscale=False,
            hoverinfo="skip",
        )
    )
    for label, color in {"Negative": SECONDARY_ACCENT, "Positive": PRIMARY_ACCENT}.items():
        subset = frame[frame["Class"] == label]
        fig.add_trace(
            go.Scatter(
                x=subset["Signal A"],
                y=subset["Signal B"],
                mode="markers",
                name=label,
                marker=dict(size=9, color=color, opacity=0.9, line=dict(color="white", width=0.8)),
                hovertemplate="Signal A: %{x:.2f}<br>Signal B: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_xaxes(title="Signal A", zeroline=False)
    fig.update_yaxes(title="Signal B", zeroline=False)
    return base_layout(fig, title)


def decision_tree_regions(dataset: DatasetBundle, model, title: str) -> go.Figure:
    return classification_regions(dataset, model, title)


def tree_accuracy_curve(depth_scores: list[dict[str, float]], selected_depth: int) -> go.Figure:
    frame = pd.DataFrame(depth_scores)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["Depth"],
            y=frame["Train accuracy"],
            mode="lines+markers",
            name="Train accuracy",
            line=dict(color="#334155", width=3),
            marker=dict(color="#334155"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["Depth"],
            y=frame["Test accuracy"],
            mode="lines+markers",
            name="Test accuracy",
            line=dict(color="#64748b", width=3, dash="dash"),
            marker=dict(color="#64748b"),
        )
    )
    fig.add_vline(x=selected_depth, line_width=2, line_dash="dash", line_color="#334155")
    fig.update_xaxes(title="Tree depth", dtick=1)
    fig.update_yaxes(title="Accuracy", tickformat=".0%", range=[0.72, 1.03])
    return base_layout(fig, "Complexity: training fit vs generalization")


def tree_depth_concept_diagram(selected_depth: int) -> go.Figure:
    visible_depth = min(selected_depth, 4)
    node_positions = []
    edge_segments = []
    for depth in range(visible_depth + 1):
        nodes_at_depth = 2**depth
        for index in range(nodes_at_depth):
            x_position = (index + 0.5) / nodes_at_depth
            y_position = -depth
            node_positions.append((depth, x_position, y_position))
            if depth > 0:
                parent_x = ((index // 2) + 0.5) / (nodes_at_depth // 2)
                parent_y = -(depth - 1)
                edge_segments.append((parent_x, parent_y, x_position, y_position))

    fig = go.Figure()
    for x0, y0, x1, y1 in edge_segments:
        fig.add_shape(
            type="line",
            x0=x0,
            y0=y0 - 0.08,
            x1=x1,
            y1=y1 + 0.08,
            line=dict(color="#cbd5e1", width=2),
        )

    x_values = []
    y_values = []
    labels = []
    colors = []
    sizes = []
    for depth, x_position, y_position in node_positions:
        x_values.append(x_position)
        y_values.append(y_position)
        colors.append("#334155" if depth == selected_depth else "#f8fafc")
        sizes.append(48 if depth == selected_depth else 38)
        if depth == 0:
            labels.append("Root<br>depth 0")
        elif depth == selected_depth:
            labels.append(f"Depth<br>{depth}")
        else:
            labels.append(str(depth))

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers+text",
            marker=dict(size=sizes, color=colors, line=dict(color="#334155", width=1.5)),
            text=labels,
            textposition="middle center",
            textfont=dict(size=10, color="#172033"),
            hoverinfo="skip",
        )
    )
    for depth in range(visible_depth + 1):
        fig.add_annotation(
            x=-0.08,
            y=-depth,
            text=f"level {depth}",
            showarrow=False,
            xanchor="right",
            font=dict(size=12, color="#64748b"),
        )
    if selected_depth > visible_depth:
        fig.add_annotation(
            x=0.5,
            y=-visible_depth - 0.48,
            text=f"Depth {selected_depth} continues beyond this simplified diagram",
            showarrow=False,
            font=dict(size=12, color="#475569"),
        )
    fig.update_xaxes(visible=False, range=[-0.22, 1.08])
    fig.update_yaxes(visible=False, range=[-visible_depth - 0.7, 0.45])
    return base_layout(fig, "What tree depth means")


def tree_training_process_diagram() -> go.Figure:
    labels = [
        "Labeled<br>points",
        "Test possible<br>questions",
        "Pick best<br>split",
        "Grow branches<br>where useful",
        "Stop and<br>predict",
    ]
    notes = [
        "examples already marked<br>Positive or Negative",
        "Signal A <= value<br>Signal B <= value",
        "separates labels<br>most cleanly",
        "shape is learned<br>from the data",
        "max depth or<br>no useful split",
    ]
    colors = ["#475569", "#0f766e", "#7c3aed", "#64748b", "#334155"]
    fig = go.Figure()
    for index in range(len(labels) - 1):
        fig.add_annotation(
            x=index + 0.78,
            y=0,
            ax=index + 0.28,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.15,
            arrowwidth=2.5,
            arrowcolor="#94a3b8",
        )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(labels))),
            y=[0] * len(labels),
            mode="markers+text",
            marker=dict(size=[78, 82, 82, 86, 82], color=colors, line=dict(color="white", width=2)),
            text=labels,
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hoverinfo="skip",
        )
    )
    for index, note in enumerate(notes):
        fig.add_annotation(
            x=index,
            y=-0.68,
            text=note,
            showarrow=False,
            font=dict(size=11, color="#475569"),
        )
    fig.update_xaxes(visible=False, range=[-0.6, 4.6])
    fig.update_yaxes(visible=False, range=[-1.08, 0.72])
    return base_layout(fig, "How a decision tree learns its structure")


def accuracy_curve(
    scores: list[dict[str, float]],
    x_key: str,
    selected_x: float,
    title: str,
    x_title: str,
    log_x: bool = False,
) -> go.Figure:
    frame = pd.DataFrame(scores)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame[x_key],
            y=frame["Train accuracy"],
            mode="lines+markers",
            name="Train accuracy",
            line=dict(color="#334155", width=3),
            marker=dict(color="#334155"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame[x_key],
            y=frame["Test accuracy"],
            mode="lines+markers",
            name="Test accuracy",
            line=dict(color="#64748b", width=3, dash="dash"),
            marker=dict(color="#64748b"),
        )
    )
    fig.add_vline(x=selected_x, line_width=2, line_dash="dash", line_color="#334155")
    fig.update_xaxes(title=x_title, type="log" if log_x else None)
    fig.update_yaxes(title="Accuracy", tickformat=".0%", range=[0.68, 1.03])
    return base_layout(fig, title)


def tree_route_diagram(model, sample: pd.Series, title: str = "One example moving through the tree") -> go.Figure:
    tree = model.tree_
    feature_names = list(sample.index)
    sample_frame = pd.DataFrame([sample], columns=feature_names)
    path_nodes = set(model.decision_path(sample_frame).indices)
    display_depth = min(model.get_depth(), 3)
    positions: dict[int, tuple[float, float]] = {}
    visible_nodes: list[tuple[int, int]] = []

    def visit(node: int, depth: int, x_position: float) -> None:
        positions[node] = (x_position, -depth)
        visible_nodes.append((node, depth))
        if depth >= display_depth:
            return
        left = tree.children_left[node]
        right = tree.children_right[node]
        if left == right:
            return
        gap = 1.0 / (2 ** depth)
        visit(left, depth + 1, x_position - gap)
        visit(right, depth + 1, x_position + gap)

    visit(0, 0, 0.0)

    fig = go.Figure()
    for node, depth in visible_nodes:
        left = tree.children_left[node]
        right = tree.children_right[node]
        if depth >= display_depth or left == right:
            continue
        x0, y0 = positions[node]
        for child in (left, right):
            x1, y1 = positions[child]
            on_path = node in path_nodes and child in path_nodes
            fig.add_shape(
                type="line",
                x0=x0,
                y0=y0 - 0.08,
                x1=x1,
                y1=y1 + 0.08,
                line=dict(color="#334155" if on_path else "#cbd5e1", width=4 if on_path else 2),
            )

    x_values = []
    y_values = []
    colors = []
    sizes = []
    labels = []
    font_colors = []
    for node, depth in visible_nodes:
        x_position, y_position = positions[node]
        x_values.append(x_position)
        y_values.append(y_position)
        on_path = node in path_nodes
        sizes.append(76 if on_path else 68)
        feature_index = tree.feature[node]
        if feature_index >= 0 and depth < display_depth:
            colors.append("#334155" if on_path else "#f8fafc")
            font_colors.append("white" if on_path else "#172033")
            feature = "A" if feature_names[feature_index] == "Signal A" else "B"
            threshold = tree.threshold[node]
            labels.append(f"{feature}<br><= {threshold:.2f}")
        else:
            class_index = int(np.argmax(tree.value[node][0]))
            predicted_class = model.classes_[class_index]
            colors.append(PRIMARY_ACCENT if predicted_class == "Positive" else SECONDARY_ACCENT)
            font_colors.append("white")
            label = "Pos" if predicted_class == "Positive" else "Neg"
            labels.append(f"Predict<br>{label}")

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers+text",
            marker=dict(size=sizes, color=colors, line=dict(color="#334155", width=1.5)),
            text=labels,
            textposition="middle center",
            textfont=dict(size=10, color=font_colors),
            hoverinfo="skip",
        )
    )
    fig.add_annotation(
        x=0,
        y=0.34,
        text=f"Tracked point: Signal A {sample['Signal A']:.2f}, Signal B {sample['Signal B']:.2f}",
        showarrow=False,
        font=dict(size=13, color="#475569"),
    )
    fig.update_xaxes(visible=False, range=[-2.15, 2.15])
    fig.update_yaxes(visible=False, range=[-display_depth - 0.45, 0.55])
    return base_layout(fig, title)


def feature_importance_bar(model, title: str = "Feature importance") -> go.Figure:
    values = getattr(model, "feature_importances_", [0, 0])
    frame = pd.DataFrame({"Feature": ["Signal A", "Signal B"], "Importance": values})
    fig = px.bar(
        frame,
        x="Feature",
        y="Importance",
        color="Feature",
        color_discrete_map={"Signal A": "#64748b", "Signal B": "#94a3b8"},
    )
    fig.update_yaxes(range=[0, max(1.0, float(frame["Importance"].max()) * 1.1)])
    fig.update_layout(showlegend=False)
    return base_layout(fig, title)


def forest_vote_diagram(model, sample: pd.Series, title: str = "A forest turns many tree votes into one prediction") -> go.Figure:
    sample_frame = pd.DataFrame([sample], columns=sample.index)
    sample_array = sample.to_numpy().reshape(1, -1)
    estimators = model.estimators_[: min(9, len(model.estimators_))]
    votes = [model.classes_[int(tree.predict(sample_array)[0])] for tree in estimators]
    positive_votes = sum(vote == "Positive" for vote in votes)
    negative_votes = len(votes) - positive_votes
    final_prediction = "Positive" if positive_votes > negative_votes else "Negative"
    final_color = PRIMARY_ACCENT if final_prediction == "Positive" else SECONDARY_ACCENT

    x_values = []
    y_values = []
    colors = []
    labels = []
    cols = 3
    for index, vote in enumerate(votes):
        x_values.append(index % cols)
        y_values.append(2 - index // cols)
        colors.append(PRIMARY_ACCENT if vote == "Positive" else SECONDARY_ACCENT)
        labels.append(f"Tree {index + 1}<br>{vote}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers+text",
            marker=dict(size=62, color=colors, opacity=0.9, line=dict(color="white", width=2)),
            text=labels,
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hoverinfo="skip",
            name="Tree votes",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[4.1],
            y=[1],
            mode="markers+text",
            marker=dict(size=92, color=final_color, line=dict(color="white", width=3)),
            text=[f"Final vote<br>{final_prediction}"],
            textposition="middle center",
            textfont=dict(size=12, color="white"),
            hoverinfo="skip",
            name="Final vote",
        )
    )
    for x_position, y_position in zip(x_values, y_values):
        fig.add_shape(
            type="line",
            x0=x_position + 0.18,
            y0=y_position,
            x1=3.72,
            y1=1,
            line=dict(color="#cbd5e1", width=1.4),
        )
    fig.add_annotation(
        x=2.05,
        y=2.62,
        text=f"Visible votes: {positive_votes} Positive, {negative_votes} Negative",
        showarrow=False,
        font=dict(size=13, color="#475569"),
        xanchor="center",
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False, range=[-0.55, 4.95])
    fig.update_yaxes(visible=False, range=[-0.55, 2.95])
    return base_layout(fig, title)


def forest_size_concept_diagram(tree_count: int) -> go.Figure:
    visible_trees = min(tree_count, 12)
    x_values = []
    y_values = []
    labels = []
    colors = []
    for index in range(visible_trees):
        x_values.append(index % 6)
        y_values.append(1 - index // 6)
        labels.append(f"Tree<br>{index + 1}")
        colors.append("#f8fafc")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers+text",
            marker=dict(size=58, color=colors, line=dict(color="#64748b", width=1.6)),
            text=labels,
            textposition="middle center",
            textfont=dict(size=10, color="#334155"),
            hoverinfo="skip",
        )
    )
    fig.add_annotation(
        x=2.5,
        y=-0.72,
        text=f"{tree_count} total trees vote; this diagram shows up to 12 of them",
        showarrow=False,
        font=dict(size=13, color="#475569"),
    )
    fig.update_xaxes(visible=False, range=[-0.7, 5.7])
    fig.update_yaxes(visible=False, range=[-1.05, 1.55])
    return base_layout(fig, "What number of trees means")


def forest_training_process_diagram() -> go.Figure:
    labels = [
        "Sample<br>rows",
        "Train one<br>tree",
        "Repeat<br>many times",
        "Collect<br>votes",
        "Final<br>prediction",
    ]
    notes = [
        "each tree sees a<br>different dataset",
        "structure is learned<br>from that sample",
        "many uneven trees<br>are created",
        "classification uses<br>majority vote",
        "less sensitive to<br>one tree's quirks",
    ]
    colors = ["#475569", "#0f766e", "#7c3aed", "#64748b", "#334155"]
    fig = go.Figure()
    for index in range(len(labels) - 1):
        fig.add_annotation(
            x=index + 0.78,
            y=0,
            ax=index + 0.28,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.15,
            arrowwidth=2.5,
            arrowcolor="#94a3b8",
        )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(labels))),
            y=[0] * len(labels),
            mode="markers+text",
            marker=dict(size=[84, 80, 82, 86, 82], color=colors, line=dict(color="white", width=2)),
            text=labels,
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hoverinfo="skip",
        )
    )
    for index, note in enumerate(notes):
        fig.add_annotation(
            x=index,
            y=-0.68,
            text=note,
            showarrow=False,
            font=dict(size=11, color="#475569"),
        )
    fig.update_xaxes(visible=False, range=[-0.6, 4.6])
    fig.update_yaxes(visible=False, range=[-1.08, 0.72])
    return base_layout(fig, "How a random forest learns")
