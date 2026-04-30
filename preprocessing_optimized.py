"""
CISC 886 - Cloud Computing
Section 4: Data Preprocessing with Apache Spark on EMR

Optimized preprocessing script for Bitext Customer Support Dataset.

What this script does:
1. Reads the raw CSV file from S3.
2. Cleans instruction/response/category/intent columns.
3. Adds token-length and character-length features.
4. Saves EDA outputs to S3.
5. Splits data into train/validation/test.
6. Saves preprocessed train/validation/test CSV files to S3.

Main improvements over the original version:
- Uses explicit CSV options for safer multiline/quoted CSV reading.
- Caches the cleaned dataframe to avoid repeated recomputation.
- Reduces unnecessary repeated Spark actions.
- Uses coalesce(1) because this dataset is small, around 27K rows.
- Adds clearer progress logs.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    length,
    split,
    size,
    trim,
    lower,
    regexp_replace,
)
from pyspark.storagelevel import StorageLevel


# -------------------------------------------------------
# 1. Initialize Spark Session
# -------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("25wqgk-customer-support-preprocessing")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# -------------------------------------------------------
# 2. Configuration
# -------------------------------------------------------

BUCKET = "s3://netid-25wqgk-cloud-storage-project"

INPUT_PATH = (
    f"{BUCKET}/row data/Bitext_customer_support.csv"
)

OUTPUT_PATH = f"{BUCKET}/output"

REQUIRED_COLUMNS = ["instruction", "response", "intent", "category"]


# -------------------------------------------------------
# 3. Load Raw Dataset
# -------------------------------------------------------

print("=== Loading dataset ===")
print(f"Input path: {INPUT_PATH}")

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("multiLine", "true")
    .option("quote", '"')
    .option("escape", '"')
    .csv(INPUT_PATH)
)

print("=== Dataset loaded ===")
print(f"Columns: {df.columns}")

missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

raw_count = df.count()
print(f"Raw row count: {raw_count}")

print("=== Sample raw rows ===")
df.select(REQUIRED_COLUMNS).show(5, truncate=True)


# -------------------------------------------------------
# 4. Data Cleaning
# -------------------------------------------------------

print("=== Cleaning data ===")

df_clean = df.dropna(subset=REQUIRED_COLUMNS)

df_clean = df_clean.withColumn("instruction", trim(col("instruction")))
df_clean = df_clean.withColumn("response", trim(col("response")))
df_clean = df_clean.withColumn("category", lower(trim(col("category"))))
df_clean = df_clean.withColumn("intent", lower(trim(col("intent"))))

# Remove unusual characters from customer instruction only.
# Keep common punctuation and curly braces for placeholders.
df_clean = df_clean.withColumn(
    "instruction",
    regexp_replace(col("instruction"), r"[^\w\s\?\.\,\!\'\-\{\}]", " ")
)

# Remove rows that became empty after cleaning.
df_clean = df_clean.filter(
    (length(col("instruction")) > 0) &
    (length(col("response")) > 0) &
    (length(col("intent")) > 0) &
    (length(col("category")) > 0)
)


# -------------------------------------------------------
# 5. Feature Engineering
# -------------------------------------------------------

print("=== Adding length features ===")

df_clean = df_clean.withColumn(
    "instruction_length",
    size(split(trim(col("instruction")), r"\s+"))
)

df_clean = df_clean.withColumn(
    "response_length",
    size(split(trim(col("response")), r"\s+"))
)

df_clean = df_clean.withColumn(
    "instruction_char_length",
    length(col("instruction"))
)

df_clean = df_clean.withColumn(
    "response_char_length",
    length(col("response"))
)


# -------------------------------------------------------
# 6. Cache Cleaned Data
# -------------------------------------------------------

print("=== Caching cleaned dataframe ===")

df_clean = df_clean.persist(StorageLevel.MEMORY_AND_DISK)

clean_count = df_clean.count()
print(f"After cleaning: {clean_count} rows")

print("=== Sample cleaned rows ===")
df_clean.select(
    "instruction",
    "response",
    "intent",
    "category",
    "instruction_length",
    "response_length",
).show(5, truncate=True)


# -------------------------------------------------------
# 7. EDA Outputs
# -------------------------------------------------------

print("=== EDA 1: Token length summary ===")

length_summary = df_clean.select(
    "instruction_length",
    "response_length",
    "instruction_char_length",
    "response_char_length",
).describe()

length_summary.show(truncate=False)

length_summary.coalesce(1).write.mode("overwrite").csv(
    f"{OUTPUT_PATH}/eda_length_summary",
    header=True,
)

print(f"Saved length summary to {OUTPUT_PATH}/eda_length_summary")


print("=== EDA 2: Token length rows ===")

df_clean.select(
    "instruction_length",
    "response_length",
    "instruction_char_length",
    "response_char_length",
).coalesce(1).write.mode("overwrite").csv(
    f"{OUTPUT_PATH}/eda_token_lengths",
    header=True,
)

print(f"Saved token length rows to {OUTPUT_PATH}/eda_token_lengths")


print("=== EDA 3: Category distribution ===")

category_dist = (
    df_clean.groupBy("category")
    .count()
    .orderBy(col("count").desc())
)

category_dist.show(20, truncate=False)

category_dist.coalesce(1).write.mode("overwrite").csv(
    f"{OUTPUT_PATH}/eda_category_distribution",
    header=True,
)

print(f"Saved category distribution to {OUTPUT_PATH}/eda_category_distribution")


print("=== EDA 4: Intent distribution ===")

intent_dist = (
    df_clean.groupBy("intent")
    .count()
    .orderBy(col("count").desc())
)

intent_dist.show(20, truncate=False)

intent_dist.coalesce(1).write.mode("overwrite").csv(
    f"{OUTPUT_PATH}/eda_intent_distribution",
    header=True,
)

print(f"Saved intent distribution to {OUTPUT_PATH}/eda_intent_distribution")


# -------------------------------------------------------
# 8. Train / Validation / Test Split
# -------------------------------------------------------

print("=== Splitting dataset into train/validation/test ===")

train, val, test = df_clean.randomSplit([0.8, 0.1, 0.1], seed=42)

# Cache split dataframes before counting and writing.
train = train.persist(StorageLevel.MEMORY_AND_DISK)
val = val.persist(StorageLevel.MEMORY_AND_DISK)
test = test.persist(StorageLevel.MEMORY_AND_DISK)

train_count = train.count()
val_count = val.count()
test_count = test.count()

print(f"Train size:      {train_count}")
print(f"Validation size: {val_count}")
print(f"Test size:       {test_count}")

split_summary = spark.createDataFrame(
    [
        ("train", train_count),
        ("validation", val_count),
        ("test", test_count),
    ],
    ["split", "count"],
)

split_summary.coalesce(1).write.mode("overwrite").csv(
    f"{OUTPUT_PATH}/eda_split_counts",
    header=True,
)

print(f"Saved split counts to {OUTPUT_PATH}/eda_split_counts")


# -------------------------------------------------------
# 9. Save Preprocessed Output
# -------------------------------------------------------

print("=== Saving train/validation/test outputs to S3 ===")

cols_to_save = [
    "instruction",
    "response",
    "intent",
    "category",
    "instruction_length",
    "response_length",
    "instruction_char_length",
    "response_char_length",
]

train.select(cols_to_save).coalesce(1).write.mode("overwrite").csv(
    f"{OUTPUT_PATH}/train",
    header=True,
)

val.select(cols_to_save).coalesce(1).write.mode("overwrite").csv(
    f"{OUTPUT_PATH}/val",
    header=True,
)

test.select(cols_to_save).coalesce(1).write.mode("overwrite").csv(
    f"{OUTPUT_PATH}/test",
    header=True,
)

print("=== All outputs saved successfully ===")
print(f"Train output:      {OUTPUT_PATH}/train")
print(f"Validation output: {OUTPUT_PATH}/val")
print(f"Test output:       {OUTPUT_PATH}/test")


# -------------------------------------------------------
# 10. Cleanup
# -------------------------------------------------------

train.unpersist()
val.unpersist()
test.unpersist()
df_clean.unpersist()

spark.stop()

print("=== Spark session stopped ===")
