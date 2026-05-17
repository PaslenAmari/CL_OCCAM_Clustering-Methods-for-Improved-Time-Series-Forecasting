Clustering Methods for Improved Time-Series Forecasting 

The goals of the research were the following:
To develop and validate a cluster-ensemble forecasting framework for heterogeneous time series panels that improves predictive accuracy over global baseline models through unsupervised structural segmentation and automated per-cluster model selection.

The objectives:
1. Review and classify existing approaches to time series clustering (shape-based, feature-based, model-based) and cluster-ensemble forecasting methodologies.
2. Design and implement a modular two-stage pipeline integrating unsupervised clustering with per-cluster model selection.
3. Validate clustering approaches and baseline forecasting behaviour on a synthetic dataset with known ground-truth cluster structure.
4. Experimentally evaluate three clustering paradigms (DTW agglomerative, TSFresh feature-based, sktime TimeSeriesKMeans) on the PowerCons benchmark (360 series) and select the best-performing approach.
5. Extend the pipeline with automated per-cluster model selection based on Occam's Razor principle, and apply it to the Worldwide Electricity Load Dataset (42 regions) and the Rossmann Store Sales retail panel (50 stores).
6. Rigorously assess statistical significance of performance improvements.

Brief descriotion of the obtained results:
The cluster-ensemble pipeline demonstrated statistically significant improvements over global baselines across three real-world heterogeneous panels. On the PowerCons dataset (360 series), three clustering paradigms were evaluated (DTW agglomerative, TSFresh feature-based, sktime TimeSeriesKMeans), with the DTW-based ensemble achieving the best performance (DM-test statistic: 8.226, p < 0.0001). Building on this, the DTW agglomerative + CatBoost pipeline was applied to the Worldwide Electricity Load Dataset (42 regions), reducing MAE by 7.3% and sMAPE by 19.3% (Wilcoxon test: W = 11,633.0, p < 0.001). Finally, on the Rossmann retail panel (50 stores), consistent MAE improvements of 6–11% were observed across forecast horizons h ∈ {7, 14, 30} days, confirming domain-agnostic applicability. The pipeline is CPU-only and fully Docker-containerized for reproducibility.

The research was conducted over multiple consecutive stages, with each stage building upon the findings and validated components of the previous one. This progressive design was adopted deliberately to ensure that the complexity of the experimental environment was increased in a controlled manner, so that performance improvements could be attributed to the methodology rather than to confounding factors in the data.
In the first stage of the research, the clustering methodology was explored on benchmark medical time series from the UCR Time Series Archive, specifically the ECG200 and ECG5000 datasets. These datasets, comprising electrocardiogram recordings of normal heartbeats and myocardial infarction events, were used to evaluate the relative performance of K-Means clustering with Euclidean distance and Spectral Clustering with DTW distance in a classification-adjacent context. The ECG200 dataset contains 200 labelled instances of length 96, while ECG5000 comprises 5,000 instances derived from a 20-hour Physionet recording. Although these datasets are primarily classification benchmarks, they provided a controlled testbed for validating clustering algorithms before applying them to unlabelled forecasting data.
In the second stage, a synthetic dataset was designed and generated to serve as a controlled forecasting laboratory. The synthetic data consisted of 8,760 data points simulating one full year of hourly observations, structured into four pre-defined behavioral archetypes with known ground truth cluster assignments. This stage validated the end-to-end cluster-ensemble forecasting pipeline under ideal conditions.
In the third stage, the validated pipeline was applied to a real-world electricity consumption panel — the Worldwide Electricity Load Dataset — comprising 42 geographical regions observed over a full calendar year at daily resolution. This dataset was selected for its extreme structural heterogeneity and direct relevance to industrial energy management applications. In the fourth and final stage, the pipeline was extended without architectural modification to the Rossmann Store Sales panel, a retail dataset of 50 German stores drawn from the Kaggle forecasting competition. This stage served a dual purpose: to validate the generalizability of the methodology beyond the energy domain, and to test whether the cluster-ensemble advantage persists when the global baseline is enriched with domain-relevant exogenous regressors — Promo, StateHoliday, SchoolHoliday, and Open — supplied as known-future inputs.
The four experimental stages above reflect a deliberate progression in both dataset complexity and methodological scope. The ECG benchmark served as a controlled validation testbed for the clustering algorithm itself, confirming that DTW-based segmentation reliably recovers pre-defined structural patterns before any forecasting objective was introduced. PowerCons then provided the first real-world stress test for the complete cluster-ensemble pipeline, enabling a head-to-head comparison of all three clustering paradigms under identical conditions. The Worldwide Electricity Load dataset constituted the primary empirical contribution of the thesis, applying the full pipeline — including the Occam's Razor model selection module — to a globally heterogeneous energy panel. Finally, Rossmann extended the evaluation beyond the energy domain to a retail setting, testing whether the cluster-ensemble advantage generalizes to a structurally different form of heterogeneity driven by business archetypes rather than geographic factors.

Data:
Data files required on premises:
- PowerCons dataset: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
- ELEC dataset: https://data.mendeley.com/datasets/ybggkc58fz/1
- Rossmann dataset: https://www.kaggle.com/c/rossmann-store-sales

To run the project:
# Build the image
docker compose build

# Run the main experiment (ELEC + ROSSMANN, all horizons)
docker compose run --rm etna-app python run_cl_occam.py

# Or run ablations
docker compose run --rm etna-app python ablation.py

# Or run the Rossmann-with-exogenous experiment
docker compose run --rm etna-app python rossmann_exog.py
