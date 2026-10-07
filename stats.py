

# 0. PROBABILITY RULES, CONDITIONAL PROBABILITY & BAYES’ THEOREM
# 1. Random Variables & Probability Distributions — One E-Commerce Scenario

import numpy as np
import pandas as pd

x = np.array([0, 1, 2, 3])
p = np.array([0.50, 0.30, 0.15, 0.05])

print("Probability sum:", p.sum())

mean = np.sum(x * p)
print("Expected value:", mean)

variance = np.sum((x - mean)**2 * p)
print("Variance:", variance)

std = np.sqrt(variance)
print("Standard deviation:", std)

cdf = np.cumsum(p)

df = pd.DataFrame({"Orders": x, "PMF": p, "CDF": cdf})
print(df)



# 2. Binomial Distribution, Normal Distribution & Central Limit Theorem


from math import comb
n = 10
p = 0.20
k = 2

prob = comb(n, k) * p**k * (1-p)**(n-k)
print("P(X=2):", prob)

mean = n*p
variance = n*p*(1-p)
std = np.sqrt(variance)

print("Mean:", mean)
print("Variance:", variance)
print("Standard deviation:", std)



# 3.Sampling Techniques & Sampling Distributions

# 3.1 Simple Random Sampling
import numpy as np

rng = np.random.default_rng(42)
customers = np.arange(1, 10001)

sample = rng.choice(customers, size=100, replace=False)

print("Sample size:", len(sample))
print("First 10 IDs:", sample[:10])


# 3.2 Sampling Distribution of the Mean
import numpy as np

rng = np.random.default_rng(42)

population = rng.normal(loc=2500, scale=600, size=100000)

sample_size = 100
number_of_samples = 5000

sample_means = np.array([
    rng.choice(population, size=sample_size, replace=False).mean()
    for _ in range(number_of_samples)
])

print("Population mean:", population.mean())
print("Mean of Ms:", sample_means.mean())
print("Std of Ms:", sample_means.std())


# 3.3 Sampling Distribution of a Proportion
import numpy as np

rng = np.random.default_rng(42)
p = 0.20
n = 100
number_of_samples = 5000

sample_proportions = np.array([
    rng.binomial(n=n, p=p) / n
    for _ in range(number_of_samples)
])

print("Mean:", sample_proportions.mean())
print("Std:", sample_proportions.std())

theoretical_se = np.sqrt(p*(1-p)/n)
print("Theoretical SE:", theoretical_se)



# 4. COVARIANCE & CORRELATION

# Correlation in Python — Pandas
import pandas as pd

data = {
    "study_hours": [1, 2, 3, 4, 5],
    "marks": [50, 55, 60, 65, 70]
}

df = pd.DataFrame(data)
print(df["study_hours"].corr(df["marks"]))


