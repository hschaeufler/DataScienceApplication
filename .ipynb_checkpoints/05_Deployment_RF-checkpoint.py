#!/usr/bin/env python
# coding: utf-8

# # Data Modelling

# In[1]:


from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from helpers.helper_functions import translate_to_file_string
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics

inputFile = translate_to_file_string("./data/Data_Preparation_Result.csv")

def prettyPrint(dm, collArray) :
    rows = dm.toArray().tolist()
    dfDM = spark.createDataFrame(rows,collArray)
    newDf = dfDM.toPandas()
    from IPython.display import display, HTML
    return HTML(newDf.to_html(index=False))


# ## Create Spark Session

# In[2]:


#create a SparkSession
spark = (SparkSession
       .builder
       .appName("DataModelling")
       .getOrCreate())
# create a DataFrame using an ifered Schema 
df = spark.read.option("header", "true")        .option("inferSchema", "true")        .option("delimiter", ";")        .csv(inputFile)   
print(df.printSchema())


# ## Vorbereitung der Daten

# ### Filtern der Datensätze
# Für das Training dieses Modells ist es sinnvoll nur die Fälle zu betrachten, bei den der Ausgang der Corona-Erkrankung bereits bekannt ist ("GENESEN" oder "GESTORBEN"). Daher werden die Fälle mit noch erkrankten Personen herausgefiltert. Ebenfalls muss der FallStatusIndex neu vergeben werden, damit dieses Feature nur noch die Werte 0 oder 1 enthält.

# In[3]:


dfNeu = df.filter(df.FallStatus != "NICHTEINGETRETEN").drop("FallStatusIndex")


# ### FallStatusIndex

# In[4]:


indexer = StringIndexer(inputCol="FallStatus", outputCol="FallStatusIndex")
dfReindexed = indexer.fit(dfNeu).transform(dfNeu)


# ### Ziehen eines Samples
# Da der Datensatz sehr groß ist,kann es evt. notwendig sein, nur mit einem kleineren Umfang zu trainieren. Mit Fraction kann an dieser Stelle der Umfang der Stichprobe angepasst werden.

# In[5]:


dfsample = dfReindexed.sample(withReplacement=False, fraction=1.0, seed=12334556)


# ### Undersampling
# Ähnlich dem Fraud-Detection-Beispiel von Tara Boyle (2019) ist die Klasse der an Corona-Verstorbenen im vorliegenden Datensatz unterrepresentiert, weshalb man an dieser Stelle von einer Data Imbalance spricht. Dies sieht man wenn man die Anzahl der Todesfälle mit den Anzahl der Genesenen vergleicht.

# In[6]:


# Vergleich der Fallzahlen
dfsample.groupBy("FallStatus").count().show()


# Die meisten Machine Learning Algorithmen arbeiten am Besten wenn die Nummer der Samples in allen Klassen ungefähr die selbe größe haben. Dies konnte auch im Zuge dieser Arbeit bei den unterschiedlichen Regressions-Modellen festgestellt werden. Da die einzelnen Modelle versuchen den Fehler zu reduzieren, haben alle Modelle am Ende für einen Datensatz nur die Klasse Genesen geliefert, da hier die Wahrscheinlichkeit am größten war korrekt zu liegen. 
# Um diese Problem zu lösen gibt es zwei Möglichkeiten: Under- und Oversampling. Beides fällt unter den Begriff Resampling
# Beim Undersampling werden aus der Klasse mit den meisten Instanzen, Datensätze gelöscht, wohingegen beim Oversampling, der Klasse mit den wenigsten Isntanzen, neue Werte hinzugefügt werden. (Will Badr 2019; Tara Boyle 2019)
# Da in diesem Fall ausreichend Datensätze vorhanden sind, bietet sich Ersteres an.

# In[7]:


# Ermittlung der Anzahl dr Verstorbenen
dfGestorben = dfsample.filter(dfsample.FallStatus == "GESTORBEN")
anzahlGestorben = dfGestorben.count()
print("Anzahl Gestorben : %s" % anzahlGestorben)


# In[8]:


# Ermittlung des Verhätlnisses von Verstorben und Gensen
dfGenesen = dfsample.filter(dfsample.FallStatus == "GENESEN")
anzahlGenesen = dfGenesen.count()
print("Anzahl Genesen : %s" % anzahlGenesen)

ratio = anzahlGestorben / anzahlGenesen
print("Verhältnis : %s" % ratio)


# In[9]:


# Ziehen eines Samples mit der näherungsweise selben Anzahl wie Verstorbene
dfGenesenSample = dfGenesen.sample(fraction=ratio, seed=12345)


# In[10]:


dfGesamtSample = dfGestorben.union(dfGenesenSample)
# Kontrolle
dfGesamtSample.groupBy("FallStatus").count().show()


# ### Splitten in Trainings und Testdaten

# In[11]:


splits = dfGesamtSample.randomSplit([0.8, 0.2], 345678)
trainingData = splits[0]
testData = splits[1]


# ### Aufbau des Feature-Vectors

# In[12]:


assembler =  VectorAssembler(outputCol="features", inputCols=["GeschlechtIndex","AltersgruppeIndex", "LandkreisIndex","BundeslandIndex"])


# ## Modellierung
# ### RandomForestClassifier
# RandomForests sind eine besondere Form von Entscheidungsbäumen. Es werden mehrere Bäume kombiniert um bessere Entscheidungen treffen zu können. (Apache Spark 2021c)

# In[13]:


rfc = RandomForestClassifier(featuresCol="features", labelCol="FallStatusIndex")


# ### Pipeline

# In[14]:


pipeline = Pipeline(stages=[assembler,rfc])


# ### Evaluator
# Für die spätere Cross-Validaton wird ein Evaluator benötigt. Letzterer ist zu wählen, abhängig von dem jeweilligen Modell und Anwendungsfall. (Apache Spark 2020a) In diesem Fall wird ein BinaryClassificationEvaluator angewendet. Dieser eignet sich besonders für binäre Werte. (Apache Spark 2021a) Da Geschlecht, in diesem Fall, der FallStatus 0 oder 1 annehmen kann, bietet er sich hier besonders an.

# In[15]:


# Definition des Evaluators
evaluator= BinaryClassificationEvaluator(labelCol="FallStatusIndex",rawPredictionCol="rawPrediction", metricName="areaUnderPR")


# ### Parametertuning
# Eine wichtige Aufgabe beim Machine Learning ist die Auswahl des geeigneten Modells bzw. die passenden Paramter für ein Modell herauszufinden. Letzteres wird auch Parametertuning genannt. Die in Pyspark enthaltene MLLib bietet speziell hierfür ein entsprechende Tooling. Und zwar kann ein CrossValidator bzw. ein TrainValidationSplit verwendet werden. Voraussetzung sind ein Estimator (ein Modell oder eine Pipeline), ein Paramter-Grid und eine Evaluator. Dies ist auch im Zusammenhang mit dem Thema Cross-Validation zu sehen. (Apache Spark 2020a)

# In[16]:


paramGrid = ParamGridBuilder()    .addGrid(rfc.numTrees, [5,20,50])     .addGrid(rfc.maxBins, [5,32])    .addGrid(evaluator.metricName, ["areaUnderPR", "areaUnderROC"])    .build()


# ### Cross-Validation

# In[17]:


# Definition des Cross-Validators 
# num-Folds gibt an in wie viele Datensatz-Paare die Datensätze aufgeteilt werden.
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=2,
                          parallelism=2)


# #### Training

# In[18]:


# Anpassung des Modells und Auswahl der besten Parameter
cvModel = crossval.fit(trainingData)


# In[19]:


# Ermitteln der Paramter
print(cvModel.explainParams())


# ### Testen des Modells

# In[20]:


predictions = cvModel.transform(testData)
predictions.show()


# In[21]:


# Kontrolle der Predictions
predictions.groupBy("FallStatusIndex", "prediction").count().show()


# ## Modell - Evaluation

# Area Under PR / Area under ROC

# In[22]:


accuracy = evaluator.evaluate(predictions)
print("Test Error",(1.0 - accuracy))


# ### BinaryClassificationMetrics

# Bei dem untersuchten Label (FallStatus mit den Ausprägungen Verstorben und Genesen) handelt es sich um ein einen BinaryClasificator. Er kann die Werte 0 und 1 annehmen. Für die Modellevaluation sind daher die BinaryClassificationMetrics zu verwenden. (Apache Spark 2021i)

# In[23]:


predictionAndLabels = predictions.select("prediction", "FallStatusIndex").rdd.map(lambda p: [p[0], p[1]]) # Map to RDD prediction|label


# In[24]:


# Instanzieire das BinaryClassificationMetrics-Objekt
metrics = BinaryClassificationMetrics(predictionAndLabels)

# Fläche unter der Precision-recall Curve

print("Area under PR = %s" % metrics.areaUnderPR)

# Fläche unter der ROC curve
print("Area under ROC = %s" % metrics.areaUnderROC)


# ### Multiclass classification Metrics
# In den meißten Fällen können auch Multiclass Classification Metrics bei Binary Classifaction Problemen angewandt werden.

# In[25]:


predictionAndLabels = predictions.select("prediction", "FallStatusIndex").rdd.map(lambda p: (p[0], p[1])) # Map to RDD prediction|label
# Instantiate metrics object
mcMetrics = MulticlassMetrics(predictionAndLabels)


# In[26]:


prettyPrint(mcMetrics.confusionMatrix(),["positiv", "negativ"])


# In[27]:


# Overall statistics
precision = mcMetrics.precision(1.0)
recall = mcMetrics.recall(1.0)
f1Score = mcMetrics.fMeasure(1.0)
print("Summary Stats")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)


labels = predictions.select("FallStatusIndex").rdd.map(lambda lp: lp.FallStatusIndex).distinct().collect()
for label in sorted(labels):
    print("Class %s precision = %s" % (label, mcMetrics.precision(label)))
    print("Class %s recall = %s" % (label, mcMetrics.recall(label)))
    print("Class %s F1 Measure = %s" % (label, mcMetrics.fMeasure(label, beta=1.0)))

# Weighted stats
print("Weighted recall = %s" % mcMetrics.weightedRecall)
print("Weighted precision = %s" % mcMetrics.weightedPrecision)
print("Weighted F(1) Score = %s" % mcMetrics.weightedFMeasure())
print("Weighted F(0.5) Score = %s" % mcMetrics.weightedFMeasure(beta=0.5))
print("Weighted false positive rate = %s" % mcMetrics.weightedFalsePositiveRate)

