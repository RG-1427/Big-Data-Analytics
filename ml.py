#Packages used
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

#Imports spark section
import matplotlib
matplotlib.use('Agg')
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col
from pyspark.ml.evaluation import ClusteringEvaluator

def py():
    #Reading the file
    df = pd.read_csv("data.csv")

    #Variables for the calculations
    tcpOutDisturbances = []
    tcpInpDisturbances = []
    start_time = time.time()

    #Creating a list containing the variables
    for i in range(len(df)):
        out = ((df['TCPOutputDelay'].iloc[i] + df['TCPOutputJitter'].iloc[i] + df['TCPOutputPloss'].iloc[i])
                  / df['TCPOutputPacket'].iloc[i])
        tcpOutDisturbances.append(out)
        inp = ((df['TCPInputDelay'].iloc[i] + df['TCPInputJitter'].iloc[i] + df['TCPInputPloss'].iloc[i])
                  / df['TCPInputPacket'].iloc[i])
        tcpInpDisturbances.append(inp)

    #List of the data
    data = list(zip(tcpOutDisturbances, tcpInpDisturbances))

    #Plotting the clustering of the data
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(data)
    scatter = plt.scatter(tcpOutDisturbances, tcpInpDisturbances, c=kmeans.labels_, label=kmeans.labels_)
    plt.legend(*scatter.legend_elements(), title='clusters', loc='upper right')
    plt.title('TCP Disturbances Clustering Scatter')
    plt.xlabel('TCP Output Disturbances')
    plt.ylabel('TCP Input Disturbances')
    plt.show()

    #Displaying results
    df['kmean'] = kmeans.labels_
    print(df['kmean'].value_counts())
    print("Time taken = ", time.time() - start_time)

    #Reseting the variables
    tcpOutDisturbances.clear()
    tcpInpDisturbances.clear()
    start_time = time.time()

    #Calculation with dimensionality reduction
    for i in range(len(df)):
        out = ((df['TCPOutputDelay'].iloc[i])
                  / df['TCPOutputPacket'].iloc[i])
        tcpOutDisturbances.append(out)
        inp = ((df['TCPInputDelay'].iloc[i])
                  / df['TCPInputPacket'].iloc[i])
        tcpInpDisturbances.append(inp)

    #List of the data
    data = list(zip(tcpOutDisturbances, tcpInpDisturbances))

    #Plotting the clustering of the data
    kmeans = KMeans(n_clusters=6)
    kmeans.fit(data)
    scatter = plt.scatter(tcpOutDisturbances, tcpInpDisturbances, c=kmeans.labels_, label=kmeans.labels_)
    plt.legend(*scatter.legend_elements(), title='clusters', loc='upper right')
    plt.title('TCP Disturbances Clustering Scatter')
    plt.xlabel('TCP Output Disturbances')
    plt.ylabel('TCP Input Disturbances')
    plt.show()

    #Displaying results
    df['kmean'] = kmeans.labels_
    print(df['kmean'].value_counts())
    print("Time taken = ", time.time() - start_time)

def spark1():
    # Reading the file differently when starting the spark session
    spark = SparkSession.builder.appName('Clustering1').getOrCreate()
    df = spark.read.format("csv").option("header", True).option("separator", ",").load(
        "hdfs://192.168.56.102:9000/data.csv")

    #Reading the data
    start_time = time.time()
    df = df.withColumn('TCPOutputDisturbance',
                       (col('TCPOutputDelay') + col('TCPOutputJitter') + col('TCPOutputPloss')) / col(
                           'TCPOutputPacket'))
    df = df.withColumn('TCPInputDisturbance',
                       (col('TCPInputDelay') + col('TCPInputJitter') + col('TCPInputPloss')) / col('TCPInputPacket'))

    # Select relevant columns
    assembler = VectorAssembler(inputCols=['TCPOutputDisturbance', 'TCPInputDisturbance'], outputCol='features')
    df = assembler.transform(df)

    #Scaling the data
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scalerModel = scaler.fit(df)
    scaledDf = scalerModel.transform(df)

    # Using the data to determine how many clusters are needed
    inertias = []
    for i in range(2, 11):
        kmeans = KMeans(k=i, seed=1)
        model = kmeans.fit(scaledDf)

        # Make predictions
        predictions = model.transform(scaledDf)

        # Evaluate clustering by using the ClusteringEvaluator
        evaluator = ClusteringEvaluator()
        inertia = evaluator.evaluate(predictions)
        inertias.append(inertia)

    # Plotting the data to decide on the cluster number
    plt.plot(range(2, 11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig('/home/hduser/py_scripts/inertia.png')
    plt.clf()

    # Clustering the data
    kmeans = KMeans(k=5, seed=1)
    model = kmeans.fit(df)
    df = model.transform(df)

    # Collect a sample of data to the driver for plotting
    sample_df = df.select('TCPOutputDisturbance', 'TCPInputDisturbance', 'prediction').sample(False, 0.1).toPandas()

    # Plotting the clustering results
    scatter = plt.scatter(sample_df['TCPOutputDisturbance'], sample_df['TCPInputDisturbance'],
                          c=sample_df['prediction'], label=sample_df['prediction'])
    plt.legend(*scatter.legend_elements(), title='Clusters', loc='upper right')
    plt.title('TCP Disturbances Clustering Scatter')
    plt.xlabel('TCP Output Disturbances')
    plt.ylabel('TCP Input Disturbances')
    plt.savefig('/home/hduser/py_scripts/graph.png')
    plt.clf()

    # Displaying results
    df.groupBy('prediction').count().show()
    print("Time taken = ", time.time() - start_time)

def spark2():
    # Reading the file differently when starting the spark session
    spark = SparkSession.builder.appName('Clustering2').getOrCreate()
    df = spark.read.format("csv").option("header", True).option("separator", ",").load(
        "hdfs://192.168.56.102:9000/data.csv")

    # Resetting the timer
    start_time = time.time()

    # Redoing the calculations for dimensionality reduction
    df = df.withColumn('TCPOutputDisturbance', col('TCPOutputDelay') / col('TCPOutputPacket'))
    df = df.withColumn('TCPInputDisturbance', col('TCPInputDelay') / col('TCPInputPacket'))

    # Select relevant columns
    assembler = VectorAssembler(inputCols=['TCPOutputDisturbance', 'TCPInputDisturbance'], outputCol='features')
    df = assembler.transform(df)

    # Scaling the data
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scalerModel = scaler.fit(df)
    scaledDf = scalerModel.transform(df)

    # Using the data to determine how many clusters are needed
    inertias = []
    for i in range(2, 11):
        kmeans = KMeans(k=i, seed=1)
        model = kmeans.fit(scaledDf)

        # Make predictions
        predictions = model.transform(scaledDf)

        # Evaluate clustering by using the ClusteringEvaluator
        evaluator = ClusteringEvaluator()
        inertia = evaluator.evaluate(predictions)
        inertias.append(inertia)

    # Plotting the data to decide on the cluster number
    plt.plot(range(2, 11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig('/home/hduser/py_scripts/inertia2.png')
    plt.clf()

    # Clustering the data
    kmeans = KMeans(k=6, seed=1)
    model = kmeans.fit(df)
    df = model.transform(df)

    # Collect a sample of data to the driver for plotting
    sample_df = df.select('TCPOutputDisturbance', 'TCPInputDisturbance', 'prediction').sample(False, 0.1).toPandas()

    # Plotting the clustering results
    scatter = plt.scatter(sample_df['TCPOutputDisturbance'], sample_df['TCPInputDisturbance'],
                          c=sample_df['prediction'], label=sample_df['prediction'])
    plt.legend(*scatter.legend_elements(), title='Clusters', loc='upper right')
    plt.title('TCP Disturbances Clustering Scatter')
    plt.xlabel('TCP Output Disturbances')
    plt.ylabel('TCP Input Disturbances')
    plt.savefig('/home/hduser/py_scripts/graph2.png')
    plt.clf()

    # Displaying results
    df.groupBy('prediction').count().show()
    print("Time taken = ", time.time() - start_time)

#py()
spark1()
spark2()