# ============================================================
#   TMDB Movies Data Analytics Project
#   Dataset : tmdb_5000_movies.csv  &  tmdb_5000_credits.csv
#   College Data Analytics Project
# ============================================================

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1 : Import Required Libraries
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import ast
import warnings
import matplotlib
matplotlib.use("Agg")          # non-interactive backend â€” saves to file, no GUI window
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 5)})

print("=" * 60)
print("   TMDB Movies â€” Data Analytics Project")
print("=" * 60)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2 : Load the Datasets
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 2] Loading datasets â€¦")

movies  = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

print(f"  Movies  dataset : {movies.shape[0]:,} rows Ã— {movies.shape[1]} columns")
print(f"  Credits dataset : {credits.shape[0]:,} rows Ã— {credits.shape[1]} columns")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3 : Merge the Two Datasets
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 3] Merging movies and credits on 'title' â€¦")

credits.rename(columns={"movie_id": "id"}, inplace=True)
df = movies.merge(credits, on="id", suffixes=("", "_credits"))
df.drop(columns=["title_credits"], inplace=True)   # remove duplicate title col

print(f"  Merged shape : {df.shape[0]:,} rows Ã— {df.shape[1]} columns")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4 : Exploratory Data Analysis (EDA)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 4] Exploratory Data Analysis â€¦")

print("\n  --- First 5 rows ---")
print(df[["title", "budget", "revenue", "vote_average",
          "popularity", "runtime"]].head())

print("\n  --- Missing values ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n  --- Basic statistics ---")
print(df[["budget", "revenue", "vote_average",
          "popularity", "runtime"]].describe().round(2))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5 : Data Cleaning Pipeline
#          (uses sklearn Pipeline for numeric preprocessing)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 5] Cleaning data using Pipeline â€¦")

# 5a. Drop columns that are mostly empty or irrelevant
df.drop(columns=["homepage", "tagline", "spoken_languages",
                  "production_countries", "status"], inplace=True)

# 5b. Drop rows with missing overview or release_date
df.dropna(subset=["overview", "release_date"], inplace=True)

# 5c. Extract year from release_date
df["release_year"] = pd.to_datetime(df["release_date"],
                                    errors="coerce").dt.year

# 5d. Replace 0 budget / revenue with NaN (treat as missing)
df["budget"]  = df["budget"].replace(0, np.nan)
df["revenue"] = df["revenue"].replace(0, np.nan)

# 5e. Numeric cleaning Pipeline  (impute â†’ scale)
numeric_cols = ["budget", "revenue", "runtime", "popularity",
                "vote_count", "vote_average"]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

scaled_array = numeric_pipeline.fit_transform(df[numeric_cols])
df_scaled    = pd.DataFrame(scaled_array,
                             columns=[c + "_scaled" for c in numeric_cols],
                             index=df.index)
df = pd.concat([df, df_scaled], axis=1)

print("  Numeric Pipeline applied  (imputation + standardisation)")
print(f"  Clean dataset shape : {df.shape}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 6 : Feature Extraction from JSON columns
#          (genres, keywords, cast, crew)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 6] Extracting features from JSON-like columns â€¦")

def extract_names(text, field="name", limit=3):
    """Parse a JSON list string and return up to `limit` values."""
    try:
        items = ast.literal_eval(text)
        return [item[field] for item in items[:limit]]
    except Exception:
        return []

def get_director(crew_text):
    """Return the director's name from the crew JSON string."""
    try:
        crew = ast.literal_eval(crew_text)
        for member in crew:
            if member.get("job") == "Director":
                return member["name"]
    except Exception:
        pass
    return np.nan

df["genres_list"]   = df["genres"].apply(extract_names)
df["keywords_list"] = df["keywords"].apply(extract_names)
df["cast_list"]     = df["cast"].apply(extract_names)
df["director"]      = df["crew"].apply(get_director)

# Primary genre (first genre) for grouping
df["primary_genre"] = df["genres_list"].apply(
    lambda x: x[0] if x else "Unknown")

print("  Extracted: genres, keywords, cast, director, primary_genre")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 7 : Analysis â€” Genre-wise Statistics
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 7] Genre-wise analysis â€¦")

genre_stats = (df[df["primary_genre"] != "Unknown"]
               .groupby("primary_genre")
               .agg(
                   movie_count  = ("title",        "count"),
                   avg_rating   = ("vote_average", "mean"),
                   avg_revenue  = ("revenue",      "mean"),
                   avg_budget   = ("budget",       "mean"),
               )
               .sort_values("movie_count", ascending=False)
               .head(12)
               .round(2))

print(genre_stats)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 8 : Visualisation â€” 4 Charts
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 8] Creating visualisations â€¦")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("TMDB Movies â€” Data Analytics Dashboard",
             fontsize=16, fontweight="bold", y=1.01)


# --- Chart 1 : Top Genres by Movie Count (Bar) ---
ax1 = axes[0, 0]
genre_stats["movie_count"].plot(kind="bar", ax=ax1,
                                 color=sns.color_palette("Set2", 12),
                                 edgecolor="white")
ax1.set_title("Top Genres by Number of Movies", fontweight="bold")
ax1.set_xlabel("Genre")
ax1.set_ylabel("Movie Count")
ax1.tick_params(axis="x", rotation=45)
for bar in ax1.patches:
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 5,
             int(bar.get_height()),
             ha="center", va="bottom", fontsize=8)


# --- Chart 2 : Average Rating by Genre (Horizontal Bar) ---
ax2 = axes[0, 1]
genre_stats["avg_rating"].sort_values().plot(
    kind="barh", ax=ax2,
    color=sns.color_palette("coolwarm", 12))
ax2.set_title("Average Rating by Genre", fontweight="bold")
ax2.set_xlabel("Average Vote Score")
ax2.axvline(df["vote_average"].mean(), color="red",
            linestyle="--", linewidth=1.2, label="Overall Mean")
ax2.legend(fontsize=8)


# --- Chart 3 : Budget vs Revenue Scatter ---
ax3 = axes[1, 0]
plot_df = df.dropna(subset=["budget", "revenue"])
ax3.scatter(plot_df["budget"] / 1e6, plot_df["revenue"] / 1e6,
            alpha=0.4, s=20, color="steelblue", edgecolors="none")
ax3.set_title("Budget vs Revenue (in Millions USD)", fontweight="bold")
ax3.set_xlabel("Budget ($M)")
ax3.set_ylabel("Revenue ($M)")
# Trend line
z = np.polyfit(plot_df["budget"], plot_df["revenue"], 1)
p = np.poly1d(z)
x_line = np.linspace(plot_df["budget"].min(), plot_df["budget"].max(), 200)
ax3.plot(x_line / 1e6, p(x_line) / 1e6, "r--", linewidth=1.5,
         label="Trend")
ax3.legend(fontsize=8)


# --- Chart 4 : Vote Average Distribution (Histogram + KDE) ---
ax4 = axes[1, 1]
sns.histplot(df["vote_average"], bins=30, kde=True,
             ax=ax4, color="mediumpurple", edgecolor="white")
ax4.set_title("Distribution of Vote Averages", fontweight="bold")
ax4.set_xlabel("Vote Average")
ax4.set_ylabel("Frequency")
ax4.axvline(df["vote_average"].mean(), color="red",
            linestyle="--", label=f"Mean = {df['vote_average'].mean():.2f}")
ax4.legend(fontsize=8)


plt.tight_layout()
plt.savefig("data/analytics_dashboard.png", bbox_inches="tight")

print("  Dashboard saved â†’ data/analytics_dashboard.png")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 9 : Correlation Analysis
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 9] Correlation analysis â€¦")

corr_cols = ["budget", "revenue", "popularity",
             "runtime", "vote_average", "vote_count"]
corr_matrix = df[corr_cols].corr().round(2)
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f",
            cmap="RdYlGn", linewidths=0.5, square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap â€” Key Numeric Features",
          fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("data/correlation_heatmap.png", bbox_inches="tight")

print("  Heatmap saved â†’ data/correlation_heatmap.png")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 10 : Machine Learning â€” Predict Revenue
#           using a Linear Regression Pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[STEP 10] Machine Learning â€” Revenue Prediction Pipeline â€¦")

# Prepare ML dataset (drop rows where budget or revenue is missing)
ml_df = df.dropna(subset=["budget", "revenue",
                           "runtime", "popularity", "vote_count"]).copy()

# Encode primary_genre as a numeric label
le = LabelEncoder()
ml_df["genre_encoded"] = le.fit_transform(ml_df["primary_genre"])

# Features and target
FEATURES = ["budget", "runtime", "popularity", "vote_count", "genre_encoded"]
TARGET   = "revenue"

X = ml_df[FEATURES]
y = ml_df[TARGET]

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ML Pipeline : impute â†’ scale â†’ model
ml_pipeline = Pipeline(steps=[
    ("imputer",  SimpleImputer(strategy="median")),
    ("scaler",   StandardScaler()),
    ("model",    LinearRegression())
])

ml_pipeline.fit(X_train, y_train)
y_pred = ml_pipeline.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"  Training samples : {len(X_train):,}")
print(f"  Testing  samples : {len(X_test):,}")
print(f"  RMSE             : ${rmse:,.0f}")
print(f"  RÂ² Score         : {r2:.4f}")

# Actual vs Predicted plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test / 1e6, y_pred / 1e6,
            alpha=0.5, s=20, color="darkorange", edgecolors="none")
plt.plot([0, y_test.max() / 1e6], [0, y_test.max() / 1e6],
         "b--", linewidth=1.5, label="Perfect Prediction")
plt.xlabel("Actual Revenue ($M)")
plt.ylabel("Predicted Revenue ($M)")
plt.title("Actual vs Predicted Revenue â€” Linear Regression",
          fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("data/revenue_prediction.png", bbox_inches="tight")

print("  Plot saved â†’ data/revenue_prediction.png")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 11 : Insights Summary
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 60)
print("   KEY INSIGHTS")
print("=" * 60)

top_genre    = genre_stats["movie_count"].idxmax()
top_rated    = genre_stats["avg_rating"].idxmax()
best_revenue = genre_stats["avg_revenue"].idxmax()
top_director = (df.groupby("director")["vote_average"]
                .mean()
                .sort_values(ascending=False)
                .head(1))

print(f"  1. Most produced genre      : {top_genre}")
print(f"  2. Highest rated genre      : {top_rated}")
print(f"  3. Highest avg revenue genre: {best_revenue}")
print(f"  4. Top director (avg rating): {top_director.index[0]}"
      f" ({top_director.values[0]:.2f})")
print(f"  5. Overall avg vote score   : {df['vote_average'].mean():.2f}")
print(f"  6. Revenue prediction RÂ²    : {r2:.4f} "
      f"({'Good' if r2 > 0.5 else 'Moderate'} fit)")

print("\n  Output files saved in  data/")
print("    â€¢ analytics_dashboard.png")
print("    â€¢ correlation_heatmap.png")
print("    â€¢ revenue_prediction.png")
print("\n[Project Complete]")
