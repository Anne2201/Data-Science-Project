## Title

From Data to Screen: A Strategic Framework for Movie Investment & Risk Optimization

## Project description
"This project focuses on transforming raw cinematic data into actionable investment strategies. By bridging the gap between decades of historical market trends and state-of-the-art Deep Learning models, the framework identifies high-potential movie projects while quantifying the financial risks associated with the upcoming 2026–2035 decade. The goal is to move beyond simple predictions to provide a robust decision-making tool for studio portfolio management."

## Objective

The aim of this project is to provide studio executives and financial analysts with a **Prescriptive Strategy Tool**. 

* **Predict**: Forecast worldwide box-office revenue using temporal trends.

* **Optimize**: Identify the ideal "Theatrical Window" (release month) and budget allocation per genre.

* **Mitigate**: Use stochastic modeling to provide a "Safety Floor" for capital investment, moving away from high-risk "guesswork."

## Methodology

This project follows a four-stage analytical path:

1. Descriptive Analytics: A deep dive into nearly a century of cinema history (1937–2023) to understand market concentration, distributor power, and the evolution of genre popularity.

2. Market Segmentation (Clustering): Utilizing K-Means Clustering to categorize films into distinct strategic tiers (e.g., Global Blockbusters, Mid-Range, and Niche Indie).

3. Temporal Forecasting (LSTM): A Long Short-Term Memory (PyTorch) neural network architecture trained to recognize non-linear patterns and cyclical trends in theatrical sales.

4. Risk Assessment (Monte Carlo): A stochastic simulation engine that runs 100 iterations per project, injecting market noise to generate probabilistic outcomes and ROI confidence intervals.

## Data

The analysis is based on a comprehensive dataset (`Data1.xlsx`), obteined on kaggle.com,  containing detailed financial and technical attributes of theatrical releases:

* **Financials**: Production Budgets, Domestic Opening, and Worldwide Sales.

* **Metadata**: Main Genre, Runtime (standardized into minutes), and Distributor.

* **Temporal**: Precise Release Dates transformed into seasonal categorical variables.


## Project Structure
```text
PROJET DATA SCIENCE                      
├── data/
│   └── raw/
│       └── Data1.xlsx   
│── notebooks 
├── results/
│   ├── plots/              
│   └── numerics/
├── src/
│   ├── _pycache_
│   ├── _init_.py 
│   ├── data_loader.py      
│   ├── evaluation.py            
│   └── models.py       
│── environment.yml
├── main.py 
│── proposal/
│── README.md
│── requirements.txt      