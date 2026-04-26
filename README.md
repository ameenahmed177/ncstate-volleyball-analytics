# NC State Women's Volleyball Analytics Project

## Overview
This project analyzes NC State women's volleyball match data and builds machine learning models to predict match outcomes (win/loss) based on team performance statistics.

## Tools Used
- Python (data extraction, cleaning, data pipeline)
- R (statistical modeling and model evaluation)

## Data Pipeline
1. Extract match-level data from NCAA statistics website using Python
2. Clean and preprocess data (remove non-match rows, standardize features)
3. Combine 2024 and 2025 datasets
4. Perform a time-based train/test split to avoid data leakage

## Modeling
Implemented and compared:
- Logistic Regression
- k-Nearest Neighbors (kNN)
- Decision Tree
- Random Forest

Used repeated 5-fold cross-validation (10 repeats) to evaluate performance.

## Key Results
- Random Forest achieved the highest ROC AUC (~0.90)
- Decision Tree achieved the highest accuracy (~0.81)
- Important predictors included hitting percentage, aces, and kills

## Goal
Provide insights into which performance metrics most influence match outcomes and build predictive models for future matches.
