import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from scipy.optimize import minimize

# Load data
df = pd.read_excel('result_kz.xlsx')  # Update the path to your Excel file
df['GovInvest_GDP_Ratio'] = df['IM'] / df['GDPM']
df['PublicDebt_GDP_Ratio'] = df['GDT'] / df['GDPM']
df['GDP_Growth'] = df['GDPM'].pct_change()

# Drop any NaN values that could interfere with regressions
df.dropna(inplace=True)

# Define the transition function
def transition_function(x, gamma, c):
    # Clip x * gamma to avoid overflow in exp
    z = -gamma * (x - c)
    z = np.clip(z, -np.inf, np.inf)  # You might adjust bounds based on your data
    return 1 / (1 + np.exp(z))

# Adjusted rolling regression to handle edge cases
def rolling_regression(df, window, gamma, c):
    results = []
    for i in range(len(df) - window + 1):
        subset = df.iloc[i:i+window]
        transition = transition_function(subset['PublicDebt_GDP_Ratio'], gamma, c)
        X = sm.add_constant(subset['GovInvest_GDP_Ratio'] * transition)
        try:
            model = sm.OLS(subset['GDP_Growth'], X).fit()
            results.append(model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1])
        except Exception as e:  # Catch all exceptions from regression to avoid interruption
            print(f"Regression error: {e}")
            continue
    return np.array(results)

# Objective function with validation
def objective(params, *args):
    gamma, c = params
    df, window = args
    coefficients = rolling_regression(df, window, gamma, c)
    if coefficients.size == 0 or np.isnan(coefficients).all():  # Check for empty or all-NaN results
        return np.inf  # Return a large penalty to guide optimizer
    valid_coefficients = coefficients[~np.isnan(coefficients)]
    return -np.mean(valid_coefficients) if valid_coefficients.size > 0 else np.inf

# Optimization setup remains the same
# Ensure initial guesses and bounds are reasonable
bounds = [(0.001, 20), (df['PublicDebt_GDP_Ratio'].min(), df['PublicDebt_GDP_Ratio'].max())]  # Example bounds
initial_gamma = 1.0
initial_c = df['PublicDebt_GDP_Ratio'].median()

result = minimize(objective, [initial_gamma, initial_c], args=(df, 5), method='L-BFGS-B', bounds=bounds)

if result.success:
    optimal_gamma, optimal_c = result.x
    # Ensure the result is plausible before using it
    optimal_investment_ratio = rolling_regression(df, 5, optimal_gamma, optimal_c)
    if optimal_investment_ratio.size > 0 and not np.isnan(optimal_investment_ratio).all():
        optimal_investment_ratio_max = np.nanmax(optimal_investment_ratio)
        latest_GDP = df['GDPM'].iloc[-1]
        optimal_investment = optimal_investment_ratio_max * latest_GDP // 1000
        print(f"Optimal Government Investment/GDP Ratio: {optimal_investment_ratio_max:.4f}")
        print(f"Optimal Government Investment Amount: {optimal_investment:.2f}")
    else:
        print("Failed to find an optimal government investment ratio due to invalid regression results.")
else:
    print(f"Optimization failed with message: {result.message}")