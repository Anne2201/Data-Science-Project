# Proposal - Predicting Movie Box-Office Performance: revenues predictions and trajectory (forecasting). 

# 1. Research Question:

Part 1: Can we predict a film’s box office revenues based on data, what are the success factors?
Part 2: Are there any patterns from which we can accurately forecast the revenue trajectory of a film?

# Problem Statement & Motivation

-	Movie studios face uncertainty in predicting financial success
-	Understand which factors drive revenue: help with risk assessment, budget allocation, and marketing/ communication strategy
-	Predicting the box-office curves offers an insight into film life cycles and audience behaviour

# Dataset & Variables

-	Metadata: budget, genres, production company, release date and runtime.
-	Financials: revenues, sales

# Inputs & Features

For Regression: final revenue prediction
-	Budget, genre, production company, date of release (seasonality), runtime, sales (international, World Wilde)

Forecasting curve:
-	Revenues, targets

# Planned Approach & Technologies

Part 1: Predict Final Revenue
-	Explore and clean data using pandas
-	Train:
        - Multiple linear regression
        - Model complexity & regularization
        - Decision trees
        - Combinations of models
        - Clustering
-	Cross-validation

Part 2: Forecast Box-Office Revenue Trajectory
-	Build a time-series sequences (which movie, when, how much revenue)
-	Normalize the sequences
-	Train LSTM
-	Predict the revenue trajectory and compare it to the bassline model: simple linear regression
-	See how consistent it can be

# Expected Challenges & Solutions:
-	High variance in revenues and budget depending on the size of the film production (ex: blockbusters vs. small productions): try to use log-transformation target variable

-	Incomplete data: filter, conservative data imputation
-	Difference in timing (of the movies): train on older and new years

# Success Criteria:

-	Final model provides a final revenue prediction and a revenue curve prediction based on the different values (with a time frame: older vs new movies).
-	Clear, interpretable revenue drives (actors, budget, season of release, etc.)
-	Regression & LSTM models that go further than the baseline 
-	Clear interpretation of the forecast curve shape: performance

# Stretch Goals

-	Cluster movies by revenue trajectory using k-means or GMM
-	Model ROI instead of revenue
-	Use GRU or stacked LSTM for better sequence modelling