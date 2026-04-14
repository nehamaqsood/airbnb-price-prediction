import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

#data selection and dummy variable creation
file_path = r'C:\Users\00OOoo\Desktop\Duke Fuqua\Fall 1\Decision 520\Airbnb_final\airbnb_cleaned.csv'
df = pd.read_csv(file_path)
columns_to_exclude = ['id', 'zipcode', 'last_review', 'host_since', 'first_review']
target_var = 'log_price'
existing_exclude = [col for col in columns_to_exclude if col in df.columns]
X = df.drop(columns=[target_var] + existing_exclude)
y = df[target_var]
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_encoded.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_encoded.columns, index=X_test.index)


# Linear Regression
print("=== STATSMODELS LINEAR REGRESSION SUMMARY ===")
X_train_sm = sm.add_constant(X_train_scaled_df)
model_sm = sm.OLS(y_train, X_train_sm)
results_sm = model_sm.fit()
print(results_sm.summary())
lr_model = LinearRegression()
lr_model.fit(X_train_scaled_df, y_train)
y_pred_train = lr_model.predict(X_train_scaled_df)
y_pred_test = lr_model.predict(X_test_scaled_df)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
print(f"Linear Regression Performance:")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE:  {test_rmse:.4f}")
print(f"Train R²:   {train_r2:.4f}")
print(f"Test R²:    {test_r2:.4f}")
print(f"Train MAE:  {train_mae:.4f}")
print(f"Test MAE:   {test_mae:.4f}")

## Cross-validation
cv_scores = cross_val_score(lr_model, X_train_scaled_df, y_train,
                           cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores.mean())
print(f"5-fold CV RMSE: {cv_rmse:.4f}")

## Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'coefficient': lr_model.coef_,
    'abs_coefficient': np.abs(lr_model.coef_)
})

# Analyze feature categories
def categorize_features(feature_names):
    """Categorize features into logical groups"""
    categories = {
        'location': [f for f in feature_names if 'city_' in f or 'latitude' in f or 'longitude' in f],
        'property_type': [f for f in feature_names if 'property_type_' in f],
        'room_type': [f for f in feature_names if 'room_type_' in f],
        'capacity': [f for f in feature_names if f in ['accommodates', 'bathrooms', 'bedrooms', 'beds']],
        'amenities': [f for f in feature_names if 'amenity_' in f],
        'host': [f for f in feature_names if 'host_' in f],
        'policy': [f for f in feature_names if 'cancellation_policy_' in f or 'cleaning_fee' in f or 'instant_bookable' in f],
        'reviews': [f for f in feature_names if 'review_' in f or 'number_of_reviews' in f]
    }
    return categories

feature_categories = categorize_features(X_encoded.columns)
print("\n=== FEATURE CATEGORY BREAKDOWN ===")
for category, features in feature_categories.items():
    print(f"{category}: {len(features)} features")

# Visualization of key results
plt.figure(figsize=(15, 12))

# 1. Top 15 feature coefficients
plt.subplot(2, 2, 1)
top_15 = feature_importance.nlargest(15, 'abs_coefficient').sort_values('coefficient', ascending=True)
plt.barh(range(len(top_15)), top_15['coefficient'])
plt.yticks(range(len(top_15)), top_15['feature'])
plt.xlabel('Coefficient Value')
plt.title('Top 15 Feature Coefficients\n(Positive = Increase Price, Negative = Decrease Price)')
plt.grid(axis='x', alpha=0.3)

# 2. Actual vs Predicted
plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.6, s=20)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual log_price')
plt.ylabel('Predicted log_price')
plt.title(f'Actual vs Predicted\nTest R² = {test_r2:.3f}')

# 3. Residuals vs Predicted
plt.subplot(2, 2, 3)
residuals = y_test - y_pred_test
plt.scatter(y_pred_test, residuals, alpha=0.6, s=20)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# 4. Distribution of residuals
plt.subplot(2, 2, 4)
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals\n(Mean = {:.4f})'.format(residuals.mean()))

plt.tight_layout()
plt.show()

# Save
feature_importance.to_csv('linear_regression_feature_importance.csv', index=False)
performance_summary = pd.DataFrame({
    'metric': ['Train_RMSE', 'Test_RMSE', 'Train_R2', 'Test_R2', 'Train_MAE', 'Test_MAE', 'CV_RMSE'],
    'value': [train_rmse, test_rmse, train_r2, test_r2, train_mae, test_mae, cv_rmse]
})
performance_summary.to_csv('linear_regression_performance.csv', index=False)




print("=== LASSO REGRESSION ANALYSIS ===")
# 1. Find optimal alpha using cross-validation
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, n_alphas=100)
lasso_cv.fit(X_train_scaled_df, y_train)
optimal_alpha = lasso_cv.alpha_
print(f"Optimal alpha: {optimal_alpha}")

# Plot the cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(lasso_cv.alphas_, lasso_cv.mse_path_, ':')
plt.plot(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=-1),
         'k', label='Average across folds', linewidth=2)
plt.axvline(lasso_cv.alpha_, linestyle='--', color='k',
            label=f'Optimal alpha: {lasso_cv.alpha_:.4f}')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Mean squared error')
plt.title('Lasso Cross-Validation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 2. Fit Lasso with optimal alpha
lasso_model = Lasso(alpha=optimal_alpha, max_iter=10000, random_state=42)
lasso_model.fit(X_train_scaled_df, y_train)

# Predictions
y_pred_lasso_train = lasso_model.predict(X_train_scaled_df)
y_pred_lasso_test = lasso_model.predict(X_test_scaled_df)
lasso_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_lasso_train))
lasso_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso_test))
lasso_train_r2 = r2_score(y_train, y_pred_lasso_train)
lasso_test_r2 = r2_score(y_test, y_pred_lasso_test)
lasso_train_mae = mean_absolute_error(y_train, y_pred_lasso_train)
lasso_test_mae = mean_absolute_error(y_test, y_pred_lasso_test)

print(f"\nLasso Regression Performance:")
print(f"Train RMSE: {lasso_train_rmse:.4f}")
print(f"Test RMSE:  {lasso_test_rmse:.4f}")
print(f"Train R²:   {lasso_train_r2:.4f}")
print(f"Test R²:    {lasso_test_r2:.4f}")
print(f"Train MAE:  {lasso_train_mae:.4f}")
print(f"Test MAE:   {lasso_test_mae:.4f}")

# Cross-validation for Lasso
lasso_cv_scores = cross_val_score(lasso_model, X_train_scaled_df, y_train,
                                 cv=5, scoring='neg_mean_squared_error')
lasso_cv_rmse = np.sqrt(-lasso_cv_scores.mean())
print(f"5-fold CV RMSE: {lasso_cv_rmse:.4f}")

# 3. Feature selection results
lasso_coef = lasso_model.coef_
selected_features_mask = lasso_coef != 0
selected_features = X_encoded.columns[selected_features_mask]
selected_coefficients = lasso_coef[selected_features_mask]

print(f"\n=== FEATURE SELECTION RESULTS ===")
print(f"Total features available: {X_encoded.shape[1]}")
print(f"Features selected by Lasso: {len(selected_features)}")
print(f"Features eliminated (zero coefficients): {X_encoded.shape[1] - len(selected_features)}")
print(
    f"Sparsity: {(X_encoded.shape[1] - len(selected_features)) / X_encoded.shape[1] * 100:.1f}% of features eliminated")

# Create feature importance dataframe for Lasso
lasso_feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'coefficient': lasso_coef,
    'abs_coefficient': np.abs(lasso_coef),
    'selected': lasso_coef != 0})

top_lasso_features = lasso_feature_importance[lasso_feature_importance['selected']].nlargest(20, 'abs_coefficient')[
    ['feature', 'coefficient']]
positive_lasso = lasso_feature_importance[(lasso_feature_importance['selected']) &
                                          (lasso_feature_importance['coefficient'] > 0)].nlargest(10, 'coefficient')
negative_lasso = lasso_feature_importance[(lasso_feature_importance['selected']) &
                                          (lasso_feature_importance['coefficient'] < 0)].nlargest(10, 'abs_coefficient')

print("\n=== TOP 10 POSITIVE INFLUENCES (Lasso Selected) ===")
print(positive_lasso[['feature', 'coefficient']])
print("\n=== TOP 10 NEGATIVE INFLUENCES (Lasso Selected) ===")
print(negative_lasso[['feature', 'coefficient']])


# Analyze which feature categories were most retained
def analyze_lasso_selection_by_category(lasso_feature_importance, feature_categories):
    category_analysis = {}
    for category, features in feature_categories.items():
        category_features = [f for f in features if f in lasso_feature_importance['feature'].values]
        if category_features:
            selected_in_category = lasso_feature_importance[
                (lasso_feature_importance['feature'].isin(category_features)) &
                (lasso_feature_importance['selected'])
                ]
            total_in_category = len(category_features)
            selected_count = len(selected_in_category)
            retention_rate = selected_count / total_in_category * 100

            category_analysis[category] = {
                'total_features': total_in_category,
                'selected_features': selected_count,
                'retention_rate': retention_rate,
                'avg_abs_coef': selected_in_category['abs_coefficient'].mean() if selected_count > 0 else 0
            }
    return category_analysis
lasso_category_analysis = analyze_lasso_selection_by_category(lasso_feature_importance, feature_categories)

print("\n=== LASSO SELECTION BY FEATURE CATEGORY ===")
category_summary = []
for category, stats in lasso_category_analysis.items():
    print(f"{category:15} | {stats['selected_features']:2d}/{stats['total_features']:2d} features | "
          f"Retention: {stats['retention_rate']:5.1f}% | Avg |coef|: {stats['avg_abs_coef']:.4f}")
    category_summary.append({
        'category': category,
        'selected_features': stats['selected_features'],
        'total_features': stats['total_features'],
        'retention_rate': stats['retention_rate'],
        'avg_abs_coef': stats['avg_abs_coef']
    })
category_summary_df = pd.DataFrame(category_summary)

# 4. Comparison with Linear Regression
# Load linear regression results if needed, or use from previous analysis
print("\n=== COMPARISON: LASSO vs LINEAR REGRESSION ===")

comparison = pd.DataFrame({
    'Metric': ['Test RMSE', 'Test R²', 'Test MAE', 'Number of Features', 'CV RMSE'],
    'Linear Regression': [test_rmse, test_r2, test_mae, X_encoded.shape[1], cv_rmse],
    'Lasso Regression': [lasso_test_rmse, lasso_test_r2, lasso_test_mae, len(selected_features), lasso_cv_rmse],
    'Difference': [lasso_test_rmse - test_rmse, lasso_test_r2 - test_r2,
                   lasso_test_mae - test_mae, len(selected_features) - X_encoded.shape[1],
                   lasso_cv_rmse - cv_rmse]
})
print(comparison)

# 5. Visualization of Lasso results
plt.figure(figsize=(15, 10))
# Plot 1: Top Lasso coefficients
plt.subplot(2, 2, 1)
top_lasso_plot = top_lasso_features.sort_values('coefficient', ascending=True)
plt.barh(range(len(top_lasso_plot)), top_lasso_plot['coefficient'])
plt.yticks(range(len(top_lasso_plot)), top_lasso_plot['feature'])
plt.xlabel('Coefficient Value')
plt.title('Top 20 Lasso Coefficients\n(Selected Features Only)')
plt.grid(axis='x', alpha=0.3)
# Plot 2: Category retention rates
plt.subplot(2, 2, 2)
categories_retention = category_summary_df.sort_values('retention_rate', ascending=True)
plt.barh(range(len(categories_retention)), categories_retention['retention_rate'])
plt.yticks(range(len(categories_retention)), categories_retention['category'])
plt.xlabel('Retention Rate (%)')
plt.title('Feature Retention by Category\n(Lasso Selection)')
plt.grid(axis='x', alpha=0.3)
# Plot 3: Actual vs Predicted for Lasso
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred_lasso_test, alpha=0.6, s=20)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual log_price')
plt.ylabel('Predicted log_price')
plt.title(f'Lasso: Actual vs Predicted\nTest R² = {lasso_test_r2:.3f}')
# Plot 4: Coefficient distribution
plt.subplot(2, 2, 4)
non_zero_coef = lasso_coef[lasso_coef != 0]
zero_coef = lasso_coef[lasso_coef == 0]

plt.hist(non_zero_coef, bins=30, alpha=0.7, color='blue', label=f'Non-zero ({len(non_zero_coef)})')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Coefficient Value')
plt.ylabel('Frequency')
plt.title('Lasso Coefficient Distribution')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 6. Detailed analysis of what Lasso eliminated
print("\n=== FEATURES ELIMINATED BY LASSO ===")

# Features with highest potential impact but eliminated
eliminated_features = lasso_feature_importance[~lasso_feature_importance['selected']]
print(f"Lasso eliminated {len(eliminated_features)} features")
try:
    lr_feature_importance = pd.read_csv('linear_regression_feature_importance.csv')
    merged = lr_feature_importance.merge(
        lasso_feature_importance[['feature', 'selected']],
        on='feature'
    )
    eliminated_important = merged[
        (~merged['selected']) &
        (np.abs(merged['coefficient']) > 0.1)
        ].nlargest(10, 'coefficient')

    if len(eliminated_important) > 0:
        print("\nNotable features eliminated by Lasso:")
        print(eliminated_important[['feature', 'coefficient']])
except Exception as e:
    print(f"not able to analyze: {e}")
print("\n=== ELIMINATION BY CATEGORY ===")
for category, features in feature_categories.items():
    category_features = [f for f in features if f in lasso_feature_importance['feature'].values]
    if category_features:
        eliminated_count = len([f for f in category_features if f in eliminated_features['feature'].values])
        total_in_category = len(category_features)
        if total_in_category > 0:
            print(f"{category:15} | Eliminated: {eliminated_count:2d}/{total_in_category:2d} "
                  f"({eliminated_count / total_in_category * 100:.1f}%)")

# 7. Save
print("\n=== SAVING RESULTS ===")
lasso_feature_importance.to_csv('lasso_feature_importance.csv', index=False)
selected_features_df = pd.DataFrame({
    'selected_features': selected_features,
    'coefficients': selected_coefficients
})
selected_features_df.to_csv('lasso_selected_features.csv', index=False)
lasso_performance = pd.DataFrame({
    'metric': ['Optimal_Alpha', 'Train_RMSE', 'Test_RMSE', 'Train_R2', 'Test_R2',
               'Train_MAE', 'Test_MAE', 'CV_RMSE', 'Num_Features_Selected', 'Total_Features'],
    'value': [optimal_alpha, lasso_train_rmse, lasso_test_rmse, lasso_train_r2, lasso_test_r2,
              lasso_train_mae, lasso_test_mae, lasso_cv_rmse, len(selected_features), X_encoded.shape[1]]
})
lasso_performance.to_csv('lasso_performance.csv', index=False)
category_summary_df.to_csv('lasso_category_analysis.csv', index=False)

print(f"\n=== KEY INSIGHTS ===")
print(f"• Lasso selected {len(selected_features)} out of {X_encoded.shape[1]} features")
print(f"• Test R²: {lasso_test_r2:.4f} (vs Linear: {test_r2:.4f})")
print(f"• Most important price drivers identified and validated")
print(f"• Model simplified by eliminating {X_encoded.shape[1] - len(selected_features)} features")



print("=== INTERACTION TERMS LINEAR REGRESSION ===")
selected_features_df = pd.read_csv('lasso_selected_features.csv')
selected_features = selected_features_df['selected_features'].tolist()
X_selected = X_encoded[selected_features]
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
    X_selected, y, test_size=0.2, random_state=42)
scaler_sel = StandardScaler()
X_train_sel_scaled = scaler_sel.fit_transform(X_train_sel)
X_test_sel_scaled = scaler_sel.transform(X_test_sel)
X_train_sel_scaled_df = pd.DataFrame(X_train_sel_scaled, columns=selected_features, index=X_train_sel.index)
X_test_sel_scaled_df = pd.DataFrame(X_test_sel_scaled, columns=selected_features, index=X_test_sel.index)

# 1. Identify potential interaction candidates
def identify_interaction_candidates(X_df, top_n=15):
    feature_variance = X_df.var().sort_values(ascending=False)
    top_variance_features = feature_variance.head(top_n).index.tolist()
    categorical_features = [f for f in X_df.columns if any(x in f for x in ['city_', 'property_type_', 'room_type_'])]
    numerical_features = [f for f in X_df.columns if f not in categorical_features and
                          f in ['accommodates', 'bathrooms', 'bedrooms', 'beds',
                                'number_of_reviews', 'review_scores_rating']]
    amenity_features = [f for f in X_df.columns if 'amenity_' in f]
    return {
        'categorical': categorical_features,
        'numerical': numerical_features,
        'amenities': amenity_features,
        'high_variance': top_variance_features
    }
interaction_candidates = identify_interaction_candidates(X_train_sel_scaled_df)

# 2. Create meaningful interaction terms
def create_interaction_terms(X_df, interaction_candidates, max_interactions=20):
    interaction_terms = {}
    # Strategy 1: Location × Property Type interactions
    city_features = [f for f in interaction_candidates['categorical'] if 'city_' in f]
    property_features = [f for f in interaction_candidates['categorical'] if 'property_type_' in f]
    if city_features and property_features:
        # Take top 3 cities and top 3 property types to avoid too many terms
        top_cities = city_features[:3]
        top_properties = property_features[:3]
        for city in top_cities:
            for prop in top_properties:
                interaction_name = f"{city}_x_{prop}"
                interaction_terms[interaction_name] = X_df[city] * X_df[prop]
    # Strategy 2: Room Type × Capacity interactions
    room_features = [f for f in interaction_candidates['categorical'] if 'room_type_' in f]
    capacity_features = [f for f in interaction_candidates['numerical'] if
                         f in ['accommodates', 'bathrooms', 'bedrooms']]
    if room_features and capacity_features:
        for room in room_features:
            for capacity in capacity_features[:2]:  # Limit to top 2 capacity features
                interaction_name = f"{room}_x_{capacity}"
                interaction_terms[interaction_name] = X_df[room] * X_df[capacity]
    # Strategy 3: Key Amenity × Location interactions
    top_amenities = [f for f in interaction_candidates['amenities']][:3]  # Top 3 amenities
    if top_amenities and city_features:
        for amenity in top_amenities:
            for city in top_cities:
                interaction_name = f"{amenity}_x_{city}"
                interaction_terms[interaction_name] = X_df[amenity] * X_df[city]
    # Strategy 4: Review Score × Property Type interactions
    review_features = [f for f in interaction_candidates['numerical'] if 'review' in f.lower()]
    if review_features and property_features:
        for review in review_features[:2]:
            for prop in top_properties:
                interaction_name = f"{review}_x_{prop}"
                interaction_terms[interaction_name] = X_df[review] * X_df[prop]
    print(f"Created {len(interaction_terms)} interaction terms")
    return interaction_terms


# Create interaction terms for training and test sets
train_interactions = create_interaction_terms(X_train_sel_scaled_df, interaction_candidates)
test_interactions = create_interaction_terms(X_test_sel_scaled_df, interaction_candidates)
X_train_with_interactions = X_train_sel_scaled_df.copy()
X_test_with_interactions = X_test_sel_scaled_df.copy()
for term_name, term_values in train_interactions.items():
    X_train_with_interactions[term_name] = term_values
for term_name, term_values in test_interactions.items():
    X_test_with_interactions[term_name] = term_values
print(f"Dataset shape with interactions: {X_train_with_interactions.shape}")
print("New interaction terms:", list(train_interactions.keys()))

# 3. Fit Linear Regression with Interaction Terms
print("\n=== LINEAR REGRESSION WITH INTERACTION TERMS ===")
# Using statsmodels for detailed coefficient analysis
X_train_sm_interactions = sm.add_constant(X_train_with_interactions)
model_sm_interactions = sm.OLS(y_train_sel, X_train_sm_interactions)
results_sm_interactions = model_sm_interactions.fit()
print(results_sm_interactions.summary())

# Sklearn model for predictions
lr_interactions = LinearRegression()
lr_interactions.fit(X_train_with_interactions, y_train_sel)
y_pred_train_interactions = lr_interactions.predict(X_train_with_interactions)
y_pred_test_interactions = lr_interactions.predict(X_test_with_interactions)
train_rmse_interactions = np.sqrt(mean_squared_error(y_train_sel, y_pred_train_interactions))
test_rmse_interactions = np.sqrt(mean_squared_error(y_test_sel, y_pred_test_interactions))
train_r2_interactions = r2_score(y_train_sel, y_pred_train_interactions)
test_r2_interactions = r2_score(y_test_sel, y_pred_test_interactions)
train_mae_interactions = mean_absolute_error(y_train_sel, y_pred_train_interactions)
test_mae_interactions = mean_absolute_error(y_test_sel, y_pred_test_interactions)

print(f"\nLinear Regression with Interactions Performance:")
print(f"Train RMSE: {train_rmse_interactions:.4f}")
print(f"Test RMSE:  {test_rmse_interactions:.4f}")
print(f"Train R²:   {train_r2_interactions:.4f}")
print(f"Test R²:    {test_r2_interactions:.4f}")
print(f"Train MAE:  {train_mae_interactions:.4f}")
print(f"Test MAE:   {test_mae_interactions:.4f}")

# Cross-validation
cv_scores_interactions = cross_val_score(lr_interactions, X_train_with_interactions, y_train_sel,
                                        cv=5, scoring='neg_mean_squared_error')
cv_rmse_interactions = np.sqrt(-cv_scores_interactions.mean())
print(f"5-fold CV RMSE: {cv_rmse_interactions:.4f}")

# 4. Analyze Interaction Effects
print("\n=== INTERACTION EFFECTS ANALYSIS ===")

# Extract coefficients for interaction terms
interaction_coefficients = {}
all_coefficients = lr_interactions.coef_
feature_names = X_train_with_interactions.columns
for i, (coef, feature) in enumerate(zip(all_coefficients, feature_names)):
    if '_x_' in feature:  # This is an interaction term
        interaction_coefficients[feature] = coef

# Sort interaction terms by absolute coefficient value
sorted_interactions = sorted(interaction_coefficients.items(),
                            key=lambda x: abs(x[1]), reverse=True)
print("Interaction Terms by Impact (Absolute Value):")
for interaction, coef in sorted_interactions:
    print(f"  {interaction:40} | Coef: {coef:7.4f}")

# Analyze significant interactions (p-value < 0.05 from statsmodels)
pvalues = results_sm_interactions.pvalues
significant_interactions = []

for feature in feature_names:
    if '_x_' in feature and pvalues[feature] < 0.05:
        significant_interactions.append({
            'interaction': feature,
            'coefficient': results_sm_interactions.params[feature],
            'p_value': pvalues[feature]
        })

print(f"\nStatistically Significant Interactions (p < 0.05): {len(significant_interactions)}")
for sig in sorted(significant_interactions, key=lambda x: abs(x['coefficient']), reverse=True)[:10]:
    print(f"  {sig['interaction']:40} | Coef: {sig['coefficient']:7.4f} | p-value: {sig['p_value']:.4f}")

# 5. Compare with previous models
print("\n=== MODEL COMPARISON ===")

# Load previous model performances
lr_performance = pd.read_csv('linear_regression_performance.csv')
lasso_performance = pd.read_csv('lasso_performance.csv')

# Create comparison table
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Lasso Regression', 'Linear + Interactions'],
    'Test_RMSE': [lr_performance[lr_performance['metric'] == 'Test_RMSE']['value'].iloc[0],
                  lasso_performance[lasso_performance['metric'] == 'Test_RMSE']['value'].iloc[0],
                  test_rmse_interactions],
    'Test_R2': [lr_performance[lr_performance['metric'] == 'Test_R2']['value'].iloc[0],
                lasso_performance[lasso_performance['metric'] == 'Test_R2']['value'].iloc[0],
                test_r2_interactions],
    'Test_MAE': [lr_performance[lr_performance['metric'] == 'Test_MAE']['value'].iloc[0],
                 lasso_performance[lasso_performance['metric'] == 'Test_MAE']['value'].iloc[0],
                 test_mae_interactions],
    'Num_Features': [X_encoded.shape[1],
                     lasso_performance[lasso_performance['metric'] == 'Num_Features_Selected']['value'].iloc[0],
                     X_train_with_interactions.shape[1]],
    'CV_RMSE': [lr_performance[lr_performance['metric'] == 'CV_RMSE']['value'].iloc[0],
                lasso_performance[lasso_performance['metric'] == 'CV_RMSE']['value'].iloc[0],
                cv_rmse_interactions]
})

print(comparison)

# Calculate improvements
lr_base_r2 = lr_performance[lr_performance['metric'] == 'Test_R2']['value'].iloc[0]
improvement_vs_lr = (test_r2_interactions - lr_base_r2) * 100
improvement_vs_lasso = (test_r2_interactions - lasso_performance[lasso_performance['metric'] == 'Test_R2']['value'].iloc[0]) * 100
print(f"\nR² Improvement vs Linear Regression: {improvement_vs_lr:+.2f}%")
print(f"R² Improvement vs Lasso Regression: {improvement_vs_lasso:+.2f}%")

# 6. Visualize Interaction Effects
plt.figure(figsize=(15, 12))
# Plot 1: Top interaction coefficients
plt.subplot(2, 2, 1)
top_interactions = sorted_interactions[:10]
interaction_names = [x[0] for x in top_interactions]
interaction_coefs = [x[1] for x in top_interactions]
y_pos = range(len(interaction_names))
plt.barh(y_pos, interaction_coefs)
plt.yticks(y_pos, [name.replace('_x_', ' ×\n') for name in interaction_names])
plt.xlabel('Coefficient Value')
plt.title('Top 10 Interaction Effects\n(Larger = Stronger Effect)')
plt.grid(axis='x', alpha=0.3)
# Plot 2: Model comparison
plt.subplot(2, 2, 2)
models = comparison['Model']
test_r2_values = comparison['Test_R2']
plt.bar(models, test_r2_values, color=['skyblue', 'lightcoral', 'lightgreen'])
plt.ylabel('Test R²')
plt.title('Model Comparison: Test R²')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(test_r2_values):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
# Plot 3: Actual vs Predicted for interaction model
plt.subplot(2, 2, 3)
plt.scatter(y_test_sel, y_pred_test_interactions, alpha=0.6, s=20)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual log_price')
plt.ylabel('Predicted log_price')
plt.title(f'Interactions Model: Actual vs Predicted\nTest R² = {test_r2_interactions:.3f}')
# Plot 4: Feature count comparison
plt.subplot(2, 2, 4)
feature_counts = comparison['Num_Features']
plt.bar(models, feature_counts, color=['skyblue', 'lightcoral', 'lightgreen'])
plt.ylabel('Number of Features')
plt.title('Model Complexity: Number of Features')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(feature_counts):
    plt.text(i, v + 5, f'{v:.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 7. Interpret key interaction effects
print("\n=== BUSINESS INTERPRETATION OF KEY INTERACTIONS ===")
def interpret_interaction(interaction_term, coefficient):
    """Provide business interpretation for interaction effects"""
    parts = interaction_term.split('_x_')
    if len(parts) == 2:
        feature1, feature2 = parts

        # Clean up feature names for display
        feature1_clean = feature1.replace('city_', '').replace('property_type_', '').replace('room_type_', '').replace(
            'amenity_', '')
        feature2_clean = feature2.replace('city_', '').replace('property_type_', '').replace('room_type_', '').replace(
            'amenity_', '')

        direction = "increases" if coefficient > 0 else "decreases"
        strength = "strongly" if abs(coefficient) > 0.1 else "moderately"

        interpretation = f"• {feature1_clean} × {feature2_clean}: {strength} {direction} price (coef: {coefficient:.3f})"
        return interpretation
    return ""

# 8. Save
print("\n=== SAVING INTERACTION MODEL RESULTS ===")
interaction_performance = pd.DataFrame({
    'metric': ['Train_RMSE', 'Test_RMSE', 'Train_R2', 'Test_R2',
               'Train_MAE', 'Test_MAE', 'CV_RMSE', 'Num_Features', 'Num_Interactions'],
    'value': [train_rmse_interactions, test_rmse_interactions, train_r2_interactions, test_r2_interactions,
              train_mae_interactions, test_mae_interactions, cv_rmse_interactions,
              X_train_with_interactions.shape[1], len(interaction_coefficients)]
})
interaction_performance.to_csv('interaction_model_performance.csv', index=False)
interaction_coef_df = pd.DataFrame({
    'interaction_term': list(interaction_coefficients.keys()),
    'coefficient': list(interaction_coefficients.values()),
    'abs_coefficient': [abs(x) for x in interaction_coefficients.values()]
}).sort_values('abs_coefficient', ascending=False)
interaction_coef_df.to_csv('interaction_coefficients.csv', index=False)
comparison.to_csv('model_comparison_all.csv', index=False)

print(f"\n=== KEY FINDINGS ===")
print(f"• Added {len(interaction_coefficients)} interaction terms")
print(f"• Test R²: {test_r2_interactions:.4f} (vs {lr_base_r2:.4f} baseline)")
print(f"• {len(significant_interactions)} statistically significant interactions found")
print(f"• Top interactions reveal nuanced pricing patterns")



print("=== TARGET 2: PRICE RATIONALITY ANALYSIS WITH RANDOM FOREST ===")

# Use the original cleaned dataset (excluding the same columns as before)
columns_to_exclude = ['id', 'zipcode', 'last_review', 'host_since', 'first_review']
target_var = 'log_price'
X = df.drop(columns=[target_var] + [col for col in columns_to_exclude if col in df.columns])
y = df[target_var]
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X_encoded_full = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_encoded_full, y, test_size=0.2, random_state=42
)

# Initialize Random Forest - let it find the best features itself
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',  # Let RF decide the best feature subset
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_rf, y_train_rf)
y_pred_train_rf = rf_model.predict(X_train_rf)
y_pred_test_rf = rf_model.predict(X_test_rf)
train_rmse_rf = np.sqrt(mean_squared_error(y_train_rf, y_pred_train_rf))
test_rmse_rf = np.sqrt(mean_squared_error(y_test_rf, y_pred_test_rf))
train_r2_rf = r2_score(y_train_rf, y_pred_train_rf)
test_r2_rf = r2_score(y_test_rf, y_pred_test_rf)
train_mae_rf = mean_absolute_error(y_train_rf, y_pred_train_rf)
test_mae_rf = mean_absolute_error(y_test_rf, y_pred_test_rf)

print("Random Forest Performance (Full Dataset):")
print(f"Train RMSE: {train_rmse_rf:.4f}")
print(f"Test RMSE:  {test_rmse_rf:.4f}")
print(f"Train R²:   {train_r2_rf:.4f}")
print(f"Test R²:    {test_r2_rf:.4f}")
print(f"Train MAE:  {train_mae_rf:.4f}")
print(f"Test MAE:   {test_mae_rf:.4f}")

# Cross-validation
cv_scores_rf = cross_val_score(rf_model, X_train_rf, y_train_rf,
                              cv=5, scoring='neg_mean_squared_error')
cv_rmse_rf = np.sqrt(-cv_scores_rf.mean())
print(f"5-fold CV RMSE: {cv_rmse_rf:.4f}")

# 3. Feature importance analysis from Random Forest itself
feature_importance_rf = pd.DataFrame({
    'feature': X_encoded_full.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_20_rf = feature_importance_rf.head(20)
plt.barh(range(len(top_20_rf)), top_20_rf['importance'])
plt.yticks(range(len(top_20_rf)), top_20_rf['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importance - Random Forest (Full Dataset)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 4. Price rationality analysis
residuals_test = y_test_rf - y_pred_test_rf
residuals_train = y_train_rf - y_pred_train_rf
original_prices_test = np.exp(y_test_rf)
predicted_prices_test = np.exp(y_pred_test_rf)
price_residuals_ratio = original_prices_test / predicted_prices_test

# Use percentiles for more robust thresholding
lower_percentile = np.percentile(residuals_test, 10)
upper_percentile = np.percentile(residuals_test, 90)
undervalued_mask = residuals_test <= lower_percentile
overvalued_mask = residuals_test >= upper_percentile
undervalued_count = undervalued_mask.sum()
overvalued_count = overvalued_mask.sum()
normal_valued_count = len(residuals_test) - undervalued_count - overvalued_count

print(f"Undervalued properties (bottom 10%): {undervalued_count} ({undervalued_count/len(residuals_test)*100:.1f}%)")
print(f"Overvalued properties (top 10%): {overvalued_count} ({overvalued_count/len(residuals_test)*100:.1f}%)")
print(f"Fairly valued properties: {normal_valued_count} ({normal_valued_count/len(residuals_test)*100:.1f}%)")
print(f"Undervalued threshold: residual < {lower_percentile:.4f}")
print(f"Overvalued threshold: residual > {upper_percentile:.4f}")

# Create opportunity dataframe
opportunity_df = pd.DataFrame({
    'actual_log_price': y_test_rf,
    'predicted_log_price': y_pred_test_rf,
    'residual': residuals_test,
    'actual_price': original_prices_test,
    'predicted_price': predicted_prices_test,
    'price_ratio': price_residuals_ratio,
    'opportunity_type': 'Fair'})
opportunity_df.loc[undervalued_mask, 'opportunity_type'] = 'Undervalued'
opportunity_df.loc[overvalued_mask, 'opportunity_type'] = 'Overvalued'

# Add original features for analysis
opportunity_df = pd.concat([opportunity_df, X_test_rf.reset_index(drop=True)], axis=1)

# 5. Detailed opportunity analysis
top_undervalued = opportunity_df[opportunity_df['opportunity_type'] == 'Undervalued'].nsmallest(20, 'residual')
top_overvalued = opportunity_df[opportunity_df['opportunity_type'] == 'Overvalued'].nlargest(20, 'residual')
print("Top 10 Undervalued Properties (Biggest Opportunities):")
print(top_undervalued[['actual_price', 'predicted_price', 'price_ratio', 'residual']].head(10))
print("\nTop 10 Overvalued Properties (Most Overpriced):")
print(top_overvalued[['actual_price', 'predicted_price', 'price_ratio', 'residual']].head(10))
avg_undervaluation = (top_undervalued['predicted_price'] - top_undervalued['actual_price']).mean()
avg_overvaluation = (top_overvalued['actual_price'] - top_overvalued['predicted_price']).mean()
print(f"\nAverage undervaluation in top opportunities: ${avg_undervaluation:.2f}")
print(f"Average overvaluation in most overpriced: ${avg_overvaluation:.2f}")

# Analyze what features characterize undervalued vs overvalued properties
def analyze_opportunity_characteristics(opportunity_df, feature_importance_rf, top_n=10):
    top_features = feature_importance_rf.head(top_n)['feature'].tolist()
    segment_analysis = {}
    for segment in ['Undervalued', 'Overvalued', 'Fair']:
        segment_data = opportunity_df[opportunity_df['opportunity_type'] == segment]
        if len(segment_data) > 0:
            analysis = {'count': len(segment_data)}
            for feature in top_features:
                if feature in segment_data.columns:
                    analysis[f'avg_{feature}'] = segment_data[feature].mean()
            analysis.update({
                'avg_actual_price': segment_data['actual_price'].mean(),
                'avg_predicted_price': segment_data['predicted_price'].mean(),
                'avg_price_ratio': segment_data['price_ratio'].mean(),
            })
            segment_analysis[segment] = analysis

    return segment_analysis, top_features
segment_analysis, top_features = analyze_opportunity_characteristics(opportunity_df, feature_importance_rf)
print(f"\nSegment Analysis (based on top {len(top_features)} RF features):")
for segment, stats in segment_analysis.items():
    print(f"\n{segment}:")
    print(f"  Count: {stats['count']}")
    print(f"  Avg Actual Price: ${stats['avg_actual_price']:.2f}")
    print(f"  Avg Predicted Price: ${stats['avg_predicted_price']:.2f}")
    print(f"  Price Ratio: {stats['avg_price_ratio']:.2f}")