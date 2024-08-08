# Databricks notebook source
# MAGIC %md
# MAGIC Use Python, Scala or Java to produce a CSV file with top five word 3-grams (n-grams, use lowercase and remove punctuation) in the commit messages for each author name in event type “PushEvent” within the file 10K.github.jsonl.bz2.
# MAGIC
# MAGIC Output example:
# MAGIC - 'author' 'first 3-gram' 'second 3-gram' 'third 3-gram' 'fourth 3-gram' 'fifth 3-gram'
# MAGIC - 'erfankashani' 'merge pull request' 'pull request #4' 'request #4 from' 'rack from 207' 'from 207 to'
# MAGIC
# MAGIC

# COMMAND ----------

from databricks.connect import DatabricksSession
spark = DatabricksSession.builder.getOrCreate()

import pyspark.sql.functions as F
import string

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
username = str(context.userName().get())

# COMMAND ----------

# Read json from the compressed archive
df = spark.read.json("/Volumes/databricks_learn/schema_learn/sample_data/ngram/10K.github.json.bz2")

df = df.filter(df.type == "PushEvent")

# For each commit create new row with column "commit"
df = df.withColumn("commit", F.explode(df.payload.commits))

# Replace punctuation with "" in "translate" by using built-in list string.punctuation and make it lower case.
df = df.withColumn("message", F.lower(F.translate(df.commit.message, string.punctuation, "")))

# Split the df.message col into an array of words by white space and new line character
df = df.withColumn("array_words", F.split(df.message, "\s+"))

# Remove empty strings
df = df.withColumn("array_words", F.array_remove(df.array_words, ""))

# COMMAND ----------

from pyspark.sql import Row
import json
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import NGram

NGRAM_QUOTA = 3

# taking array_words as ngrams source and output "ngrams" after transformation
ngram = NGram(n=NGRAM_QUOTA)
ngram.setInputCol("array_words")
ngram.setOutputCol("ngrams")

# transform ngram into a dataframe
df_ngram = ngram.transform(df)

# NOT NEEDED bcs of split \s+ done before. #df_ngram_filtered = df_ngram.filter((df_ngram.ngrams.isNotNull()) & (F.size(df_ngram.ngrams) > 0))

# merging by Author name and sending multiple ngrams lists into a single one
df_ngram_by_author = df_ngram.groupBy("commit.author.name").agg(
    F.flatten(F.collect_list("ngrams")).alias("merged_ngrams")
)

# Create new row for each ngram
df_ngram_by_author = df_ngram_by_author.withColumn("each_ngram", F.explode(df_ngram_by_author.merged_ngrams))

# Count combinations of "name" and "each_ngram" 
df_ngram_by_author = df_ngram_by_author.groupBy("name", "each_ngram").count()

# Create window to allow calculations against rows with same "name", i.e. each group of rows with the same name is a separate partiton. Note it is ordered by descending, meaning higher values to have higher rating.
window_spec = Window.partitionBy("name").orderBy(F.desc("count"))

# Create a new column "rank" with "row_number()"" which increasignly creates number for each row in partition starting from 1, where as "over(window_spec)" point to the logic of partitioning and the field which we reference to assign rank in descending order.
df_top_ngrams = df_ngram_by_author.withColumn("ranking", F.row_number().over(window_spec))

# Filter top 5 ngrams
df_top_ngrams = df_top_ngrams.filter(df_top_ngrams.ranking <= 5)

df_top_ngrams = df_top_ngrams.groupBy(df_top_ngrams.name).agg(
    F.collect_list("each_ngram").alias("ngrams")
)

top_ngram_per_author_name = "top_ngram_per_author"
# Store it in catalog
json_doc_catalog = df_top_ngrams.write.json(f"/Volumes/databricks_learn/schema_learn/sample_data/{top_ngram_per_author_name}", mode="overwrite")
# Store in via VSCode Databricks Connect in .ide folder in remote user workspace (of executed from VSCode)
#json_doc_ide = df_top_ngrams.write.json(f"./sample/{top_ngram_per_author_name}", mode="overwrite")
# Store it in Databricks user workspace connected to Git repo
json_doc_repo = df_top_ngrams.write.json(f"/Workspace/Users/${username}/databricks-info-export/ngrams/{top_ngram_per_author_name}", mode="overwrite")
