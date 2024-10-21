import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('train.csv')
df_filtered = df[df['bought'] != 0]
df_filtered['time'] = pd.to_datetime(df_filtered['time'])
df_filtered['day_of_week'] = df_filtered['time'].dt.day_name()
df_filtered['month'] = df_filtered['time'].dt.month
df_filtered['day'] = df_filtered['time'].dt.day
df_filtered['weekday'] = df_filtered['time'].dt.weekday < 5  # True if it's a weekday
df_filtered = df_filtered.sort_values(by=['user', 'time'])
df_filtered['days_since_last_purchase'] = df_filtered.groupby('user')['time'].diff().dt.days
df_filtered['days_since_last_purchase'] = df_filtered['days_since_last_purchase'].fillna(0)
output_file_path = 'filtered_enhanced_train.csv'
df_filtered.to_csv(output_file_path, index=False)

print(df_filtered.head())

# Visualization: Average Days Since Last Purchase for Each User
avg_days_since_last_purchase = df_filtered.groupby('user')['days_since_last_purchase'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='user', y='days_since_last_purchase', data=avg_days_since_last_purchase, palette='viridis')
plt.title('Average Days Since Last Purchase for Each User')
plt.xlabel('User')
plt.ylabel('Average Days Since Last Purchase')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Visualization: Days Since Last Purchase for Each User over Time
df_filtered = df_filtered.sort_values(by=['user', 'time'])
plt.figure(figsize=(14, 8))
for user in df_filtered['user'].unique():
    user_data = df_filtered[df_filtered['user'] == user]
    plt.plot(user_data['time'], user_data['days_since_last_purchase'], marker='o', label=f'User {user}')

plt.title('Days Since Last Purchase for Each User')
plt.xlabel('Time')
plt.ylabel('Days Since Last Purchase')
plt.xticks(rotation=45)
plt.legend(title='Users', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()

df_filtered['time'] = pd.to_datetime(df_filtered['time'])
df_filtered = df_filtered.sort_values(by=['user', 'time'])
unique_users = df_filtered['user'].unique()
print("Available users:", unique_users)
selected_user = int(input("Please enter the user ID you want to visualize: "))

if selected_user in unique_users:
    user_data = df_filtered[df_filtered['user'] == selected_user]
    user_data['days_since_start'] = (user_data['time'] - user_data['time'].min()).dt.days

    X = user_data['days_since_start'].values.reshape(-1, 1)
    y = user_data['days_since_last_purchase'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    average_days_since_last_purchase = user_data['days_since_last_purchase'].mean()
    lower_bound = average_days_since_last_purchase * (1 - 0.17)
    upper_bound = average_days_since_last_purchase * (1 + 0.17)

    plt.figure(figsize=(14, 8))
    plt.plot(user_data['time'], user_data['days_since_last_purchase'], marker='o', label='Actual Data', color='blue')
    plt.plot(user_data['time'], y_pred, label='Linear Regression Fit', color='red')
    plt.axhline(y=average_days_since_last_purchase, color='green', linestyle='--', label='Average Days Since Last Purchase')
    plt.fill_between(user_data['time'], lower_bound, upper_bound, color='orange', alpha=0.3, label='±0.17 Area')

    plt.title(f'Days Since Last Purchase for User {selected_user} with Linear Regression')
    plt.xlabel('Time')
    plt.ylabel('Days Since Last Purchase')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Invalid user ID selected.")
