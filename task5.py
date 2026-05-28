# Import pandas
import pandas as pd

# Load Titanic dataset
df = pd.read_csv("train.csv")

# Show first 5 rows
print("Titanic Dataset")
print(df.head())

# ---------------------------------------------------
# Step 1: Filter passengers
# Adults = Age >= 18
# Fare greater than 30
# ---------------------------------------------------

filtered_data = df[(df["Age"] >= 18) & (df["Fare"] > 30)]

print("\nAdults who paid fare greater than 30")
print(filtered_data.head())

# ---------------------------------------------------
# Step 2: Create new feature - Family Size
# Family Size = SibSp + Parch + 1
# ---------------------------------------------------

df["Family_Size"] = df["SibSp"] + df["Parch"] + 1

# ---------------------------------------------------
# Step 3: Create new feature - Fare Per Person
# Fare Per Person = Fare / Family Size
# ---------------------------------------------------

df["Fare_Per_Person"] = df["Fare"] / df["Family_Size"]

print("\nDataset with new features")
print(df[["Family_Size", "Fare_Per_Person"]].head())

# ---------------------------------------------------
# Step 4: Groupby Analysis
# Average fare and survival rate
# based on Passenger Class and Gender
# ---------------------------------------------------

group_analysis = df.groupby(["Pclass", "Sex"]).agg({
    "Fare": "mean",
    "Survived": "mean"
})

print("\nAverage Fare and Survival Rate")
print(group_analysis)

# ---------------------------------------------------
# Step 5: Save cleaned dataset
# ---------------------------------------------------

df.to_csv("task5_output.csv", index=False)

print("\nTask 5 completed successfully")