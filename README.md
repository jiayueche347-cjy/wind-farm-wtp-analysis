# Willingness to Pay for Offshore Wind Farm Relocation

This repository contains a group project completed for the course **42014 Environmental and Resource Economics** at the Technical University of Denmark (DTU).

The project investigates households' **willingness to pay (WTP)** for relocating a proposed nearshore wind farm farther offshore, where it would no longer be visible from the coast. Using survey-based valuation data, the analysis explores the distribution of WTP and its relationship with socioeconomic and spatial characteristics, with particular attention to income, age, and distance from the proposed wind farm.

A distinctive component of the project was the critical use of **Generative AI**: AI-generated answers and analyses were systematically evaluated, verified against the underlying data and economic theory, and corrected where necessary.

---

## Project Overview

The dataset contains respondents' willingness to pay for moving a proposed nearshore wind farm farther offshore (`WTP_farshore`) together with several explanatory variables:

- **Age** (`age`)
- **Gender** (`male`)
- **Household income** (`income`)
- **Permanent or summer residence in the area** (`perm`)
- **Distance from residence to the wind farm** (`D_WF`)

Because WTP was elicited directly through a survey, the analysis is based on a **stated-preference valuation framework**, such as Contingent Valuation or Choice Experiments.

The project focuses on both statistical patterns and their environmental-economic interpretation.

---

## Research Questions

The analysis addresses several key questions:

- What does the distribution of WTP for farshore wind farm relocation look like?
- How representative is the mean WTP when the distribution contains extreme values?
- How does WTP vary across different parts of the WTP distribution?
- Is household income associated with WTP?
- Does WTP decline with increasing distance from the proposed wind farm?
- Is the distance–WTP relationship linear or nonlinear?
- How does WTP vary across age groups?
- How should negative WTP values be interpreted economically?
- How reliable are AI-generated analyses when compared with calculations based on the underlying data?

---

## Methodology

The project combines descriptive statistics, data visualisation and inferential statistical analysis.

### WTP Distribution

The WTP distribution was examined using:

- Histograms and kernel density estimation (KDE)
- Box plots and individual observations
- Mean and median WTP
- Share of negative WTP responses
- WTP decile analysis

Because extreme positive observations strongly affect the scale of the distribution, both full-range and zoomed visualisations were used.

### Socioeconomic and Spatial Analysis

Relationships between WTP and explanatory variables were investigated using:

- Decile-based comparisons
- Spearman rank correlation
- Ordinary Least Squares (OLS) regression
- Kruskal–Wallis tests
- 95% confidence intervals
- Standard-deviation comparisons

Non-parametric methods were particularly useful because the WTP distribution is strongly skewed and contains extreme outliers.

---

## Key Findings

### 1. WTP is strongly right-skewed

The WTP distribution is highly right-skewed and fat-tailed.

- **Mean WTP:** approximately **4,933 DKK per household per year**
- **Median WTP:** approximately **−455 DKK per household per year**
- **Negative WTP:** approximately **53.2% of respondents**
- Extreme positive observations reach approximately **500,000 DKK**

The large difference between the mean and median shows that a small number of very high WTP observations strongly influence the average.

As a result, the mean WTP alone does not adequately represent the valuation of a typical respondent.

### 2. The upper tail dominates aggregate WTP

When respondents are divided into WTP deciles, the first five deciles have negative mean WTP, while the upper deciles become increasingly positive.

The highest WTP decile has a mean of approximately **21,026 DKK/year**, substantially higher than the lower deciles.

This demonstrates how the fat right tail disproportionately influences the overall mean WTP.

### 3. Income has only a weak relationship with WTP

The analysis identifies a weak positive monotonic relationship between income and WTP.

Spearman rank correlation indicates a statistically significant positive association, while the linear OLS relationship explains very little of the overall variation in WTP.

Extreme WTP observations and the ordinal nature of the income variable make the income–WTP relationship difficult to represent using a simple linear model.

### 4. Distance is significantly associated with WTP

Respondents living closer to the proposed wind farm generally have higher WTP for moving it farther offshore.

The analysis found:

- **Spearman ρ ≈ −0.248**
- **OLS slope ≈ −587 DKK/km**
- A statistically significant Kruskal–Wallis result across distance groups

This negative relationship is consistent with a **distance-decay / NIMBY effect**: households located closer to the wind farm experience greater visual impacts and therefore tend to value relocation more strongly.

### 5. The distance effect is nonlinear

Although the OLS coefficient is statistically significant, the model has a very low explanatory power (**R² ≈ 0.012**).

WTP falls sharply between the closest distance groups and then becomes substantially flatter.

This suggests that the effect of an additional kilometre is much stronger for households located close to the wind farm than for households already located farther away.

### 6. Age shows no clear monotonic relationship with WTP

Mean WTP varies across age groups, but the pattern is not consistently increasing or decreasing.

Some age groups show unusually high mean WTP because of extreme observations, again demonstrating the importance of considering the underlying distribution rather than relying only on group means.

---

## Economic Interpretation

The analysis illustrates an important challenge in environmental valuation: **aggregate mean WTP can differ substantially from the preference of the typical household**.

Although the overall mean WTP is positive, the median is negative and more than half of respondents report negative WTP.

This distinction matters for environmental policy and cost-benefit analysis. Reporting only the mean could suggest strong public support for paying to relocate the wind farm farther offshore, while the full distribution reveals much greater heterogeneity in preferences.

The results also demonstrate the spatial nature of environmental externalities. Households located closer to the proposed wind farm tend to place a higher value on relocating it farther offshore, consistent with the visual-disamenity and distance-decay effects discussed in environmental economics.

---

## Critical Evaluation of Generative AI

Generative AI was used throughout the assignment to support statistical analysis, interpretation and visualisation.

Rather than accepting AI-generated answers directly, the project involved:

1. generating an initial answer or analysis using AI;
2. evaluating whether the result was correct;
3. checking claims against the dataset, statistical results and economic theory;
4. identifying errors or unsupported assumptions; and
5. producing a corrected answer where necessary.

For example, an initial AI-generated response described the expected pattern of WTP deciles without calculating the actual values from the dataset. The deciles were subsequently calculated directly from the data.

This process highlights both the usefulness and limitations of Generative AI in quantitative environmental-economic analysis: AI can support coding, visualisation and interpretation, but numerical results and economic conclusions still require independent verification.

---

## Tools & Methods

**Programming and analysis**

- Python
- pandas
- NumPy
- SciPy
- Matplotlib

**Statistical methods**

- Descriptive statistics
- Decile analysis
- Spearman rank correlation
- Ordinary Least Squares regression
- Kruskal–Wallis test
- Confidence intervals
- Distribution and outlier analysis

**Environmental economics**

- Willingness to Pay (WTP)
- Stated Preference Methods
- Contingent Valuation
- Choice Experiments
- Distance-decay effects
- NIMBY effects
- Environmental valuation

---

## Repository Structure

```text
wind-farm-wtp-analysis/
├── README.md
├── code/
│   └── wtp-statistical-analysis.py
└── report/
    └── wind-farm-wtp-analysis-report.pdf
