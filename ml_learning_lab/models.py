from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    train_target: pd.Series
    test_target: pd.Series


@dataclass(frozen=True)
class ModelResult:
    model: Any
    train_score: float
    test_score: float


def make_intro_classification_dataset(
    samples: int = 240,
    class_sep: float = 1.35,
    test_size: float = 0.25,
    random_state: int = 7,
) -> DatasetBundle:
    features, target = make_classification(
        n_samples=samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=class_sep,
        random_state=random_state,
    )
    frame = pd.DataFrame(features, columns=["Signal A", "Signal B"])
    labels = pd.Series(np.where(target == 1, "Positive", "Negative"), name="Class")
    train_x, test_x, train_y, test_y = train_test_split(
        frame,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    return DatasetBundle(frame, labels, train_x, test_x, train_y, test_y)


def make_tree_overfitting_dataset(
    samples: int = 340,
    noise: float = 0.26,
    test_size: float = 0.35,
    random_state: int = 11,
) -> DatasetBundle:
    features, target = make_moons(
        n_samples=samples,
        noise=noise,
        random_state=random_state,
    )
    frame = pd.DataFrame(features, columns=["Signal A", "Signal B"])
    labels = pd.Series(np.where(target == 1, "Positive", "Negative"), name="Class")
    train_x, test_x, train_y, test_y = train_test_split(
        frame,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    return DatasetBundle(frame, labels, train_x, test_x, train_y, test_y)


def make_forest_voting_dataset() -> DatasetBundle:
    return make_tree_overfitting_dataset(noise=0.32, random_state=13)


def train_decision_tree(dataset: DatasetBundle, max_depth: int) -> ModelResult:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=7)
    model.fit(dataset.train_features, dataset.train_target)
    return ModelResult(
        model=model,
        train_score=model.score(dataset.train_features, dataset.train_target),
        test_score=model.score(dataset.test_features, dataset.test_target),
    )


def train_random_forest(dataset: DatasetBundle, n_estimators: int) -> ModelResult:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        random_state=7,
        n_jobs=1,
    )
    model.fit(dataset.train_features, dataset.train_target)
    return ModelResult(
        model=model,
        train_score=model.score(dataset.train_features, dataset.train_target),
        test_score=model.score(dataset.test_features, dataset.test_target),
    )
