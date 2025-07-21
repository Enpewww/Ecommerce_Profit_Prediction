# Building Predictive Regression Algorithms to Predict Profitability of E-Commerce Segments

## Repository Outline
1. *Shop Direct Sale Data For Research (1).csv* - Original dataset from [Shop Direct Sale Data For Research](https://www.kaggle.com/datasets/kapoorprakhar/shop-direct-sale-e-commerce-dataset).
2. *README.md* - Guideline for this project repository.
3. *Shop Direct Sale Data For Research (1).csvs* - Dataset used in this project.
4. *P1M2_Luthfi_Nadyan.ipynb* - Notebook file containing all analysis, visualization, modelling for the project.
5. *P1M2_Luthfi_Nadyan_Inference.ipynb* - Notebook file containing for inference or testing the prediction of the model.
6. *custom_date.py* - Custom transformer uses to transform one of the feature and inser it in pipeline.
7. *best_model_pred.pkl* - File pickle consisting the best model for prediction.
8. *app.py* - Main application for the deployment.
9. *home.py* - Home background for the application in the deployment.
10. *plot.py* - Script for figure in Exploratory Data Analysis.
11. *prediction.py* - Application for running the prediction in the deployment.
12. *Data_Deploy.csv* - Excel file consisting the dataset for EDA purposes only.
13. *requirements.txt* - All the necessary requirements library for deployments


## Problem Background
In today’s competitive e-commerce landscape, understanding profitability drivers is essential for growth and strategic decision-making. Profit is influenced by various factors including customer segmentation, shipping modes, product performance, and order characteristics. Different customer segments—like corporate, home office, and individual consumers—exhibit unique buying behaviors, enabling targeted pricing and promotions. Shipping choices affect both customer satisfaction and operational costs, requiring a balance between speed and efficiency. Product categories perform differently across regions, highlighting the importance of local preferences. Additionally, order size and cost of goods directly impact margins.By applying regression models to historical sales data, businesses can forecast profit more accurately. This enables smarter inventory planning, tailored marketing, and improved logistics—ultimately driving operational efficiency and competitive advantage.

## Project Output
This project resulting machine learning model for predicting profitability for e-commerce dataset environment.

## Data
Data columns (total 16 columns):
| # | Column | Non-Null | Count | Dtype | Type |
| --- | --- | --- | --- | --- | --- |
| 0 | Order ID | 4115 | non-null | int64 | Feature |
| 1 | Order Date | 4115 | non-null | object | Feature |
| 2 | Customer Name | 4115 | non-null | object | Feature |
| 3 | City | 4115 | non-null | object | Feature |
| 4 | Country | 4115 | non-null | object | Feature |
| 5 | State | 4115 | non-null | object | Feature |
| 6 | Region | 4115 | non-null | object | Feature |
| 7 | Segment | 4115 | non-null | object | Feature |
| 8 | Category | 4115 | non-null | object | Feature |
| 9 | Ship Mode | 4115 | non-null | object | Feature |
| 10 | Sub-Category | 4115 | non-null | object | Feature |
| 11 | Product Name | 4115 | non-null | object | Feature |
| 12 | Quantity | 4115 | non-null | int64 | Feature |
| 13 | Cost | 4115 | non-null | int64 | Feature |
| 14 | Profit | 4115 | non-null | int64 | Target |
| 15 | Sales | 4115 | non-null | int64 | Feature |
dtypes: int64(5), object(11)
Dataset 4115 rows and 16 columns.

## Method
Using 5 regression algorithms consisting KNN Regressor, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, and SVR Regressor. The results was finds that Random Forest Regressor is the best model for prediction.

## Stacks
- pandas: using packages for dataframe manipulation and analysis. 
- numpy: used for mathematical and array function.
- matplotlib: base of visualization library for seaborn.
- seaborn: used for creating charts, stats plots, and other statistics visualization.
- scipy: for all statistics build in Pandas.
- pickle: Saving and Loading the model.
- sklearn: Main libraries for building and evaluated the machine learning models.
- statsmodel: Main libraries for statistics testing.

## Reference
Link Deployment Hugging Face: [Deployment: E-Commerce Profitability Prediction](https://huggingface.co/spaces/LuthfiNadyan/P1M2_Luthfi_Nadyan_Deployment)

Journal 1: [Profit Prediction Using Machine Learning and Regression Models: A Comparative Study](https://drive.google.com/file/d/1yNv4_SN-os4e9_xNrC9irW5naPi4Iuqd/view?usp=sharing)

Journal 2: [Forecasting Retail Sales using Machine Learning Models](https://ajpojournals.org/journals/index.php/AJSAS/article/view/2679/3563)

Journal 3: [Enhancing Retail Sales Forecasting with Optimized Machine Learning Models](https://arxiv.org/pdf/2410.13773)


**Referensi tambahan:**
- [Efficient Representations for High-Cardinality Categorical Variables in Machine Learning](https://drive.google.com/drive/folders/1tBiZZlOJYqTruWz88ml0Ja0htdEx2y2b)