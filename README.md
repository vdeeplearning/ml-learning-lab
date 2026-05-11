# ML Learning Lab

ML Learning Lab is a polished educational Streamlit app for learning classical machine learning through concise explanations, interactive controls, and live visualizations.

The current public demo focuses on two tree-based topics:

- **Decision Trees**: depth, learned splits, decision regions, train/test accuracy, and overfitting.
- **Random Forests**: many learned trees, voting, feature importance, stability, and train/test generalization.

Logistic Regression is intentionally set aside for now and is not included in this demo.

## Project Structure

```text
ml-learning-lab/
  app.py
  pages/
    01_Decision_Trees.py
    02_Random_Forests.py
  ml_learning_lab/
    config.py
    content.py
    models.py
    plots.py
    ui.py
  requirements.txt
  README.md
  .gitignore
```

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Design Goals

- Teach classical ML concepts with beginner-friendly explanations.
- Keep every control connected to visible model behavior.
- Use class colors consistently: blue for `Positive`, orange for `Negative`.
- Separate page layout from reusable model, plotting, content, and UI helpers.
- Keep the app portfolio-ready while avoiding unnecessary complexity.

## Current Lessons

### Decision Trees

The Decision Trees lesson explains binary classification with two synthetic input features, `Signal A` and `Signal B`. Users can change tree depth and see:

- decision regions update live
- train accuracy increase with depth
- test accuracy plateau or decline as overfitting appears
- a sample moving through the learned tree
- how tree depth and training splits work

### Random Forests

The Random Forests lesson shows how many decision trees vote together. Users can change the number of trees and see:

- decision regions update live
- train/test accuracy over every tree count from 1 to 200
- feature importance
- a split vote among displayed trees
- how bootstrapped samples and majority voting work

## Roadmap

1. Polish the Decision Trees and Random Forests learning flow.
2. Add tests and deployment instructions.
3. Revisit Logistic Regression with a clearer mathematical narrative.
4. Add future lessons for k-Nearest Neighbors, SVMs, cross validation, bias vs variance, and model comparison.
