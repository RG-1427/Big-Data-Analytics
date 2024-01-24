#Packages needed
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Imports spark section
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean

def py():
    #Data stoage variables
    tcpOutDisturbances = []
    tcpInpDisturbances = []

    #Creating the data for the batches
    for num in range(31):
        #Reading the files
        file = "data" + str(num + 1) + ".csv"
        df = pd.read_csv(file)

        #Calculating the data
        out = ((df['TCPOutputDelay'].mean() + df['TCPOutputJitter'].mean() + df['TCPOutputPloss'].mean())
                  / df['TCPOutputPacket'].mean())
        tcpOutDisturbances.append(out)
        inp = ((df['TCPInputDelay'].mean() + df['TCPInputJitter'].mean() + df['TCPInputPloss'].mean())
                  / df['TCPInputPacket'].mean())
        tcpInpDisturbances.append(inp)

    #Plotting a line to show when TCP input and TCP output are the same and scattering the data
    plt.plot([min(tcpOutDisturbances), max(tcpOutDisturbances)], [min(tcpInpDisturbances), max(tcpInpDisturbances)], color="red")
    plt.scatter(tcpOutDisturbances, tcpInpDisturbances)

    #Labeling the points with their batch numbers
    counter = 0
    for (i, j) in zip(tcpOutDisturbances, tcpInpDisturbances):
        counter = counter + 1
        plt.text(i, j, counter)

    #labeling the graph
    plt.title('TCP Input vs TCP Output Disturbances per Packet Comparison')
    plt.xlabel('TCP Output Disturbances per Packet')
    plt.ylabel('TCP Input Disturbances per Packet')
    plt.show()

def spark():

    spark = SparkSession.builder.appName('BatchAnalysis').getOrCreate()

    # Data storage variables
    tcpOutDisturbances = []
    tcpInpDisturbances = []

    # Creating the data for the batches
    for num in range(31):
        # Reading the files using Spark
        file = "data" + str(num + 1) + ".csv"
        df = spark.read.format("csv").option("header", True).option("delimiter", ",").load("hdfs://192.168.56.102:9000/" + file)

        # Calculating the means
        means_df = df.agg(
            mean('TCPOutputDelay').alias('mean_TCPOutputDelay'),
            mean('TCPOutputJitter').alias('mean_TCPOutputJitter'),
            mean('TCPOutputPloss').alias('mean_TCPOutputPloss'),
            mean('TCPOutputPacket').alias('mean_TCPOutputPacket'),
            mean('TCPInputDelay').alias('mean_TCPInputDelay'),
            mean('TCPInputJitter').alias('mean_TCPInputJitter'),
            mean('TCPInputPloss').alias('mean_TCPInputPloss'),
            mean('TCPInputPacket').alias('mean_TCPInputPacket')
        )

        out = (means_df.select(
            ((means_df['mean_TCPOutputDelay'] + means_df['mean_TCPOutputJitter'] + means_df['mean_TCPOutputPloss']) /
             means_df['mean_TCPOutputPacket']).alias('tcpOutDisturbance')
        ).collect())[0]['tcpOutDisturbance']

        inp = (means_df.select(
            ((means_df['mean_TCPInputDelay'] + means_df['mean_TCPInputJitter'] + means_df['mean_TCPInputPloss']) /
             means_df['mean_TCPInputPacket']).alias('tcpInpDisturbance')
        ).collect())[0]['tcpInpDisturbance']

        tcpOutDisturbances.append(out)
        tcpInpDisturbances.append(inp)

    # Plotting a line to show when TCP input and TCP output are the same and scattering the data
    plt.plot([min(tcpOutDisturbances), max(tcpOutDisturbances)], [min(tcpInpDisturbances), max(tcpInpDisturbances)],
             color="red")
    plt.scatter(tcpOutDisturbances, tcpInpDisturbances)

    # Labeling the points with their batch numbers
    for counter, (i, j) in enumerate(zip(tcpOutDisturbances, tcpInpDisturbances), start=1):
        plt.text(i, j, str(counter))

    # Labeling the graph
    plt.title('TCP Input vs TCP Output Disturbances per Packet Comparison')
    plt.xlabel('TCP Output Disturbances per Packet')
    plt.ylabel('TCP Input Disturbances per Packet')
    plt.savefig('/home/hduser/py_scripts/batch.png')

py()
#spark()