# Real-Time Smart Energy Demand Forecasting & RAG Insights

This project is a Retrieval-Augmented Generation (RAG) powered assistant for real-time electricity demand forecasting and operational insights. Unlike traditional dashboards that only show forecasted numbers, this assistant explains why demand is expected to change, retrieves relevant historical analogs, and generates human-readable recommendations for grid operators.

# Overview

The system combines three layers:

Forecasting Layer: Predicts next-hour electricity demand using engineered time-series, weather, and calendar features.

Dynamic Thresholding Layer: Flags risk when forecasted demand exceeds the 90th percentile of historical demand for the same month and hour.

RAG Layer: Retrieves historical high-demand events and generates a narrative explanation with recommendations using a large language model (FLAN-T5).

Users interact with the assistant through a Streamlit interface where they can select a time, type in queries, and receive contextual explanations alongside charts and similar past events.

# Key Features

Next-hour demand forecasting with linear regression and engineered features

Dynamic thresholds (p90 by month and hour) to detect unusual demand

Weather-aware event flags such as heatwave, heavy rain, and high humidity

Retrieval of similar historical events from a vector database (Chroma)

Narrative generation using FLAN-T5, transforming raw forecasts into clear operator briefings

Interactive Streamlit UI for queries, forecasts, and visualizations

# Data Sources

The assistant uses a mix of cleaned and standardized datasets:

Spain hourly demand and weather data

World energy consumption (country-level)

India power demand

IEA energy demand browser

OEDI U.S. energy scenarios

Hourly weather features from open APIs (temperature, humidity, rainfall, wind, pressure, clouds)

All datasets were standardized to a consistent schema with timestamp, region, demand, and weather variables.

# Data Processing Pipeline

Step 1: Data ingestion
CSV files from Kaggle, IEA, India portal, OEDI, and weather sources are read into pandas dataframes.

Step 2: Data parsing
Datasets are merged on timestamp and region. Features are aligned to a common schema with standardized units.

Step 3: Feature engineering
Lag features: demand one hour ago and demand one day ago.
Rolling features: 24-hour rolling mean of demand.
Calendar features: hour, day of week, month, weekend flag.
Weather features: temperature, humidity, rain, snow, clouds, pressure, wind.
Binary flags: heatwave, heavy rain, high humidity.

Step 4: Threshold calculation
The 90th percentile of historical demand is calculated by month and hour, creating a dynamic threshold table. A global p90 is used as fallback.

Step 5: Event summarization
Daily demand peaks and weather conditions are aggregated into short natural language event summaries, stored along with metadata (date, region, demand, weather stats).

Step 6: Embeddings
Sentence-transformers (MiniLM-L6) encode each summary into dense vectors.

Step 7: Vector database
Embeddings, summaries, and metadata are persisted into a Chroma collection for retrieval.

Step 8: Retrieval
At runtime, the forecast context and optional user query are converted into an embedding. Similar past events are retrieved from Chroma using cosine similarity.

Step 9: Generation
The forecast, dynamic threshold, and retrieved analogs are composed into a structured prompt. FLAN-T5 generates a professional operator briefing in natural language.

Step 10: UI and orchestration
A Streamlit interface ties all steps together, showing forecasts, threshold comparisons, retrieved events, and narrative explanations with visualizations of recent demand.

# Forecasting Model

The baseline model is a linear regression trained on:

Demand lags and rolling averages

Weather variables and derived flags

Calendar and seasonal features

Performance on Spain hourly data:
Mean Absolute Error ≈ 225 MW
Root Mean Square Error ≈ 520 MW

Other models tested included Decision Trees, Random Forest, Gradient Boosting, and XGBoost. Linear Regression and Gradient Boosting provided the best balance of interpretability and performance.

# RAG Layer

The RAG implementation consists of:

Document creation: daily event summaries with demand and weather context

Embeddings: generated using sentence-transformers

Vector storage: Chroma persistent client storing vectors, summaries, and metadata

Retrieval: similarity search on event summaries, guided by current context and user query

Augmentation: retrieved evidence is appended to the model prompt

Generation: FLAN-T5 produces a cohesive briefing with forecast summary, historical context, and actionable recommendations

# Features Explained

Demand MW: electricity drawn from the grid at a given time, in megawatts
Demand lag 1h: demand exactly one hour earlier
Demand lag 24h: demand exactly 24 hours earlier
Demand roll 24h: rolling average of the last 24 hours
Temperature: air temperature in Celsius
Humidity: percent water vapor in the air
Pressure: atmospheric pressure
Rainfall and Snow: precipitation volumes in mm
Wind speed and direction: wind conditions affecting demand and supply
Cloud cover: percentage of cloudiness affecting solar generation
Calendar features: hour of day, day of week, month, weekend
Event flags: binary indicators of heatwaves, heavy rain, high humidity
p90 threshold: 90th percentile of historical demand for the same month and hour, defining unusual demand

# Streamlit UI

The interface provides:

Time selection slider to analyze past conditions

Input box for natural language queries (e.g. “High demand during heatwave?”)

Forecast vs threshold visualization

Retrieved historical analogs with metadata

Narrative explanation (LLM or deterministic markdown)

Demand visualization for the last 7 days

#Future Improvements

Stronger forecasting models: gradient boosting, LSTMs, or transformer-based time-series models

Multi-step forecasting: predict up to 24 hours ahead

Expanded retrieval corpus: add news articles, regulatory announcements, festival calendars

Explainability: use SHAP values for feature attribution in forecasts

Scenario analysis: let users simulate “what if” questions such as “what if temperature rises 3°C?”

Deployment: containerize with Docker and serve via API for integration with grid systems

Global extension: ingest more regional datasets and scale the assistant for multi-country support
