import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, udf, explode
import os
from pyspark.ml import Pipeline
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, LemmatizerModel, PerceptronModel
from sparknlp.base import DocumentAssembler, Finisher
from pyspark.sql.types import ArrayType, StringType


df_pd = pd.read_csv("Data/csv/reuters_all.csv")
df_pd = df_pd.dropna(subset=['body'])
df_pd = df_pd.reset_index(drop=True)


os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SemanticsBasedDistributedDocumentClustering_SparkNLP") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.3.3") \
    .config("spark.executorEnv.NLTK_DATA", "/Users/tuankietnguyen/nltk_data") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
df = spark.createDataFrame(df_pd)

df = df.withColumn("body", concat_ws(" ", col("title"), col("body")))



document_assembler = DocumentAssembler().setInputCol("body").setOutputCol("document")

tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("raw_tokens")

stopwords_cleaner = StopWordsCleaner.pretrained("stopwords_en", "en") \
    .setInputCols(["raw_tokens"]) \
    .setOutputCol("clean_tokens")

normalizer = Normalizer() \
    .setInputCols(["clean_tokens"]) \
    .setOutputCol("normalized_tokens") \
    .setLowercase(True) \
    .setCleanupPatterns(["[^A-Za-z]"])

pos_tagger = PerceptronModel.pretrained("pos_anc", "en") \
    .setInputCols(["document", "normalized_tokens"]) \
    .setOutputCol("pos_tags")
    
lemmatizer = LemmatizerModel.pretrained("lemma_antbnc", "en") \
    .setInputCols(["normalized_tokens"]) \
    .setOutputCol("lemmas")

finisher = Finisher() \
    .setInputCols(["lemmas", "pos_tags"]) \
    .setOutputCols(["fin_tokens", "final_pos"])

nlp_pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    stopwords_cleaner,
    normalizer,
    pos_tagger,
    lemmatizer,
    finisher
])

print("⏳ Bắt đầu tiền xử lý với Spark NLP...")
df_processed = nlp_pipeline.fit(df).transform(df)
df_processed.cache()
print("✅ Xong bước NLP")



@udf(ArrayType(StringType()))
def wsd_udf(tokens, pos_tags):
    from nltk.corpus import wordnet as wn
    from nltk.wsd import lesk
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('V'):
            return wn.VERB
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        else:
            return None
    result = []
    for word, tag in zip(tokens, pos_tags):
        pos = get_wordnet_pos(tag)
        syn = lesk(tokens, word, pos)
        if syn :
            result.append(syn.name())
    return result

df_wsd = (
    df_processed
    .withColumn("senses", wsd_udf(col("fin_tokens"), col("final_pos")))
    .drop("fin_tokens", "final_pos") 
)
print(df_wsd.columns)

@udf(ArrayType(StringType()))
def expand_syn_hyper(sids):
    from nltk.corpus import wordnet as wn
    if not sids:
        return []

    out = set()
    for sid in sids:
        if not sid:
            continue
        if "." in sid:
            try:
                syn = wn.synset(sid)
                out.update(l.name() for l in syn.lemmas())
                continue        
            except Exception:
                pass             
        out.add(sid)

    return list(out)

df_expanded = (
    df_wsd.select("id", "topics", "senses")
    .withColumn("sem_tokens", expand_syn_hyper(col("senses")))
    .drop("senses") 
)
print(df_expanded.columns)


def synset2concepts(sid, max_depth=1):
    from nltk.corpus import wordnet as wn
    try:
        syn = wn.synset(sid)
    except:
        return ["unknown"]
    
    paths = syn.hypernym_paths()
    if not paths:
        return [syn.lexname()]
    
    path = paths[0]
    concepts = []

    for i in range(0, min(len(path), max_depth + 1)):
        lemma = path[-(i + 1)].lemma_names()[0]
        if lemma not in concepts:
            concepts.append(lemma)
    return concepts

unique_synsets = (
    df_wsd
    .select(explode(col("senses")).alias("sid"))
    .where(col("sid").isNotNull())
    .distinct()
    .toPandas()["sid"]
    .tolist()
)

concept_map = {}
for sid in unique_synsets:
    key = sid.split('.')[0] 
    concepts = synset2concepts(sid)
    concept_map[key] = concepts
    
b_concept_map = spark.sparkContext.broadcast(concept_map)

@udf(ArrayType(StringType()))
def map_to_concepts(tokens):
    out = []
    for t in tokens:
        concepts = b_concept_map.value.get(t, [t]) 
        out.extend(concepts)
    return list(dict.fromkeys(out)) 

df_semantic = df_expanded.withColumn(
    "concept_tokens", map_to_concepts(col("sem_tokens"))
).drop("sem_tokens")
print(df_semantic.columns)
df_semantic.select("concept_tokens").show(5, truncate=False)

from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
import re
import snowballstemmer         

eng_stemmer = snowballstemmer.stemmer('english')

@udf(ArrayType(StringType()))
def denoise_and_stem_tokens_udf(tokens):
    if not tokens:
        return []
    
    cleaned = []
    for t in tokens:
        t = t.strip().lower()
        if len(t) < 8:
            continue
        if re.search(r'\d', t): 
            continue
        stemmed = eng_stemmer.stemWord(t)
        cleaned.append(stemmed)
    
    return list(dict.fromkeys(cleaned))  # loại trùng và giữ thứ tự

# Áp dụng lên DataFrame
df_final = df_semantic.withColumn("final_tokens", denoise_and_stem_tokens_udf(col("concept_tokens")))
df_final.select("final_tokens").show(5, truncate=False)


from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.sql.functions import col, split, explode, log2, countDistinct
def calculate_purity_entropy(df, label_col="topic", pred_col="prediction"):
    df_unique = df.select("id", pred_col, label_col).dropDuplicates(["id"])
    total = df_unique.select("id").distinct().count()
    if total == 0:
        return None, None
    counts = df.groupBy(pred_col, label_col).count()
    cluster_sizes = df.select(pred_col, "id").dropDuplicates(["id", pred_col]) \
    .groupBy(pred_col) \
    .agg(countDistinct("id").alias("cluster_total"))
    joined = counts.join(cluster_sizes, pred_col)

    max_counts = joined.groupBy(pred_col).agg({"count": "max"}).withColumnRenamed("max(count)", "max_count")
    total_purity = max_counts.agg({"max_count": "sum"}).collect()[0][0]
    purity = total_purity / total

    entropy_df = joined.withColumn("p", col("count") / col("cluster_total"))
    entropy_df = entropy_df.withColumn("p_log_p", -col("p") * log2(col("p")))
    entropy = entropy_df.groupBy(pred_col).agg({"p_log_p": "sum"}).agg({"sum(p_log_p)": "avg"}).collect()[0][0]

    return purity, entropy

results = []

variant = "Pos + worknet + stem"
from pyspark.ml.feature import IDF, HashingTF
numFeatures_stage = [9820]
for i in numFeatures_stage:
    print(f"\n============================== numFeatures= {i} ==============================")
    hashingTF = HashingTF(inputCol="final_tokens", outputCol="rawFeatures", numFeatures=i)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    df_tf = hashingTF.transform(df_final)
    idf_model = idf.fit(df_tf)
    df_features = idf_model.transform(df_tf)
    df_features = df_features.select("id", "topics", "features")
    print(df_features.columns)
    # df_features.select("features").show(5, truncate=False)


    k_stage = [5,6,7,8]
    for k in k_stage:
        print(f"\n============================== k= {k} ==============================")
        
        print(f"\nKMeans")
        km = KMeans(k=k, featuresCol="features", predictionCol="prediction", seed=42)
        pred_km = km.fit(df_features).transform(df_features)
        
        pred_km_filtered = pred_km.filter(col("topics").isNotNull() & (col("topics") != "NaN"))
        pred_km_filtered = pred_km_filtered.withColumn("topics_array", split(col("topics"), ";"))
        pred_km_exploded = pred_km_filtered.withColumn("topic", explode(col("topics_array")))
        
        purity_km, entropy_km = calculate_purity_entropy(pred_km_exploded)
        if purity_km is not None:
            print(f"Purity: {purity_km:.4f}")
            print(f"Entropy: {entropy_km:.4f}")
            results.append({
                "k": k,
                "method": "KMeans",
                "variant": variant,
                "purity": purity_km,
                "entropy": entropy_km
            })

        print(f"\nBisectingKMeans")
        bkm = BisectingKMeans(k=k, featuresCol="features", predictionCol="prediction", seed=42)
        pred_bkm = bkm.fit(df_features).transform(df_features)
        
        pred_bkm_filtered = pred_bkm.filter(col("topics").isNotNull() & (col("topics") != "NaN"))
        pred_bkm_filtered = pred_bkm_filtered.withColumn("topics_array", split(col("topics"), ";"))
        pred_bkm_exploded = pred_bkm_filtered.withColumn("topic", explode(col("topics_array")))
        
        purity_bkm, entropy_bkm = calculate_purity_entropy(pred_bkm_exploded)
        if purity_bkm is not None:
            print(f"Purity: {purity_bkm:.4f}")
            print(f"Entropy: {entropy_bkm:.4f}")
            results.append({
                "k": k,
                "method": "BisectingKMeans",
                "variant": variant,
                "purity": purity_bkm,
                "entropy": entropy_bkm
            })
spark.stop()
df_results = pd.DataFrame(results)
df_results.to_csv("../Data/csv/results.csv", index=False, mode="a", header=False)
print("✅ Đã lưu kết quả vào 'results.csv'")