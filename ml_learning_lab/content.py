from dataclasses import dataclass

from ml_learning_lab.config import LESSON_STATUS_DRAFT


@dataclass(frozen=True)
class Lesson:
    title: str
    page: str
    summary: str
    objective: str
    intuition: str
    status: str = LESSON_STATUS_DRAFT


LESSONS = [
    Lesson(
        title="Decision Trees",
        page="pages/01_Decision_Trees.py",
        summary="Learn how repeated yes/no questions carve data into useful prediction regions.",
        objective="Understand splits, depth, impurity, and why trees overfit when they grow too freely.",
        intuition=(
            "A decision tree behaves like a flowchart. Each split should make the next groups "
            "more pure, but too many splits can memorize quirks instead of learning patterns."
        ),
    ),
    Lesson(
        title="Random Forests",
        page="pages/02_Random_Forests.py",
        summary="See how many imperfect trees can vote together to produce steadier predictions.",
        objective="Understand bagging, randomness, voting, and why forests reduce variance.",
        intuition=(
            "A random forest trains many trees on slightly different views of the data. "
            "Their average prediction is usually less jumpy than a single tree."
        ),
    ),
]


def lesson_by_title(title: str) -> Lesson:
    for lesson in LESSONS:
        if lesson.title == title:
            return lesson
    raise ValueError(f"Unknown lesson: {title}")
