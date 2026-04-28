"""
hello.py — first Python script for MLE Lab 1.

Runs a tiny end-to-end ML example to confirm the Docker environment
has all the libraries we need.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def main() -> None:
    print("Hello from inside the Docker container!")

    # Make a tiny dataset: y = 2x + 1 with a bit of noise.
    rng = np.random.default_rng(seed=42)
    x = np.arange(0, 10).reshape(-1, 1)
    y = 2 * x.ravel() + 1 + rng.normal(0, 0.5, size=10)

    df = pd.DataFrame({"x": x.ravel(), "y": y})
    print("\nSample data:")
    print(df)

    # Fit a simple linear regression.
    model = LinearRegression().fit(x, y)
    print(f"\nLearned slope:     {model.coef_[0]:.3f}  (true value: 2)")
    print(f"Learned intercept: {model.intercept_:.3f}  (true value: 1)")


if __name__ == "__main__":
    main()
