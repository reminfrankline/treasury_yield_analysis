!pip install quandl
!pip install fredapi
!pip install shap

from datetime import datetime
from fredapi import Fred
import quandl
import pandas as pd
import seaborn as sns
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from scipy.stats import ttest_ind
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

fred_api_key = 'your_api_key'
quandl_api_key = 'your_api_key'
fred = Fred(api_key=fred_api_key)
quandl.ApiConfig.api_key = quandl_api_key

def etl_process():
  try:
    treasury_yields = fred.get_series('DGS10', start='2010-01-01', end='2023-12-31')
    federal_funds_rate = fred.get_series('FEDFUNDS', start='2010-01-01', end='2023-12-31')

    treasury_yields = treasury_yields.to_frame(name='10Y_Treasury_Yields')
    federal_funds_rate = federal_funds_rate.to_frame(name='Federal_Fund_Rate')

    yield_curve = quandl.get('USTREASURY/YIELD', start_date='2010-01-01', end_date = '2023-12-31')

    data = pd.concat([treasury_yields, federal_funds_rate, yield_curve], axis=1)

    data.reset_index(inplace=True)
    data.rename(columns={'index':'Date'}, inplace=True)

    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)


    return data

    print(f'{datetime.now()}: ETL Process completed Successfully' )


  except Exception as e:

    print(f'Error occurred during ETL process: {str(e)}')

def aggregation_and_simplification(data):
  try:
    summary_stats = data.describe()

    aggregated_views = data.groupby('Date').agg({'10Y_Treasury_Yields': 'mean', 'Federal_Fund_Rate': 'mean'}).reset_index()

    print("Summary statistics:")
    print(summary_stats)
    print("\nAggregated Views:")
    print(aggregated_views)

    print(f'{datetime.now()}: Data Aggregation and Simplification completed Successfully')

    return data

  except Exception as e:
    print(f'Error occurred during Data Aggregation and Simplification: {str(e)}')

def analytics_automation(X_train, y_train, X_test, y_test):
    try:
        pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42))
        param_grid = {'randomforestregressor__n_estimators':[100, 200, 300],
                      'randomforestregressor__max_depth' : [None, 10, 20]}

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_mse_mean = -cv_scores.mean()

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Mean Squared Error:", mse)
        print("R-Score:", r2)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs predicted Values')
        plt.show()

        feature_importance = best_model.named_steps['randomforestregressor'].feature_importances_
        sorted_idx = feature_importance.argsort()
        plt.figure(figsize=(10, 6))
        plt.barh(X_train.columns[sorted_idx], feature_importance[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Feature Importance Plot')
        plt.show()

        print(f'{datetime.now()}: Analytics Automation completed Successfully')

    except Exception as e:
        print(f'Error occurred during Analytics Automation: {str(e)}')

def behavioral_analysis(data):
    try:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        decomposed = seasonal_decompose(data['10Y_Treasury_Yields'].dropna(), model='additive', period=252)
        plt.figure(figsize=(12, 8))
        decomposed.plot()
        plt.show()

        kmeans = KMeans(n_clusters=3, random_state=42)
        data['Cluster'] = kmeans.fit_predict(data[['10Y_Treasury_Yields', 'Federal_Fund_Rate']].dropna())

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='10Y_Treasury_Yields', y='Federal_Fund_Rate', hue='Cluster', palette='viridis')
        plt.title('Clustering of 10Y Treasury Yields and Federal Fund Rate')
        plt.show()

        corr_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

        pearson_corr, pearson_pval = pearsonr(data['10Y_Treasury_Yields'].dropna(), data['Federal_Fund_Rate'].dropna())
        spearman_corr, spearman_pval = spearmanr(data['10Y_Treasury_Yields'].dropna(), data['Federal_Fund_Rate'].dropna())

        print(f"Pearson Correlation: {pearson_corr}, p-value: {pearson_pval}")
        print(f"Spearman Correlation: {spearman_corr}, p-value: {spearman_pval}")

        cluster_0 = data[data['Cluster'] == 0]['10Y_Treasury_Yields']
        cluster_1 = data[data['Cluster'] == 1]['10Y_Treasury_Yields']
        t_stat, p_value = ttest_ind(cluster_0.dropna(), cluster_1.dropna())

        print(f"T-test between Cluster 0 and Cluster 1: T-statistic = {t_stat}, p-value = {p_value}")

        print(f'{datetime.now()}: Behavioral Analysis completed Successfully')

    except Exception as e:
        print(f'Error occurred during Behavioral Analysis: {str(e)}')

def performance_enhancements(data):
    try:
        data['Yield_Spread'] = data['10Y_Treasury_Yields'] - data['Federal_Fund_Rate']
        data['Month'] = data.index.month
        data['Year'] = data.index.year

        param_grid = {
            'n_estimators':[100, 200, 300],
            'max_depth':[10, 20, 30],
            'min_samples_split':[2, 5, 10]
        }

        grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        X = data.drop(columns=['10Y_Treasury_Yields'])
        y = data['10Y_Treasury_Yields']
        grid_search.fit(X, y)
        best_params = grid_search.best_params_

        print(f'Best Parameters from Grid Search: {best_params}')

        plt.figure(figsize=(14, 7))
        sns.lineplot(data=data, x='Year', y='Yield_Spread', hue='Month', palette='tab10')
        plt.title('Yield Spread Over time by Month')
        plt.show()

        model = RandomForestRegressor(**best_params, random_state=42)
        model.fit(X, y)
        feature_importances = model.feature_importances_
        features = X.columns
        feature_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

        plt.figure(figsize = (10, 6))
        sns.barplot(data=feature_df, x='Importance', y='Feature')
        plt.title('Feature Importances')
        plt.show()

        scores = cross_val_score(model, X, y, cv=5)
        print(f'Cross-Validation score: {scores}')
        print(f'Average Cross-Validation Score: {np.mean(scores)}')

        rfe = RFE(estimator=model, n_features_to_select=5)
        rfe.fit(X, y)
        selected_features = X.columns[rfe.support_]

        print(f'Selected features: {selected_features}')

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_type='bar')

        print(f'{datetime.now()}: Performance Enhancements completed Successfully')

    except Exception as e:
        print(f'Error occurred during Performance Enhancements: {str(e)}')

def performance_optimization(data):
    try:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )

        X = data.drop(columns=['10Y_Treasury_Yields'])
        y = data['10Y_Treasury_Yields']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        print("Optimised Mean Square error:", mse)
        print(f'{datetime.now()}: Performance Optimization completed Successfully')

    except Exception as e:
        print(f'Error occurred during Performance Optimization: {str(e)}')

def main():

    data = etl_process()
    if data is None:
      print("ETL process failed, terminating.")
      return

    data = aggregation_and_simplification(data)

    behavioral_analysis(data)

    performance_enhancements(data)

    if 'Date' in data.columns:
      data.drop(columns=['Date'], inplace=True)

    X = data.drop(columns=['10Y_Treasury_Yields'])
    y = data['10Y_Treasury_Yields']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    analytics_automation(X_train, y_train, X_test, y_test)

    performance_optimization(data)

if __name__ == "__main__":
    main()