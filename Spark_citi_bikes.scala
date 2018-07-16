// Databricks notebook source
/*
Citi Bike is the nation's largest bike share program, with 10,000 bikes and 600 stations across Manhattan, Brooklyn, Queens and Jersey City.
We want to analyze this data to find interesting patterns in the traffic.

Dataset description
==========================================
The data includes:

Trip Duration (seconds)
Start Time and Date
Stop Time and Date
Start Station Name
End Station Name
Station ID
Station Lat/Long
Bike ID
User Type (Customer = 24-hour pass or 3-day pass user; Subscriber = Annual Member)
Gender (Zero=unknown; 1=male; 2=female)
Year of Birth

This data has been processed to remove trips that are taken by staff as they service and inspect the system, trips that are taken to/from any of our “test” stations (which we were using more in June and July 2013), and any trips that were below 60 seconds in length (potentially false starts or users trying to re-dock a bike to ensure it's secure).

*/

//Loading the datasets into a DataFrame using the databricks csv library that has been provided by databricks.

val bikes_data_DF = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").option("delimiter", ",").load("/FileStore/tables/2013*.csv")

// COMMAND ----------

//Exploring the schema of the dataframe and caching the data frame for fast query retrieval.

bikes_data_DF.printSchema()
bikes_data_DF.cache()

// COMMAND ----------

bikes_data_DF.count() //counting the no. of rows in the dataframe

// COMMAND ----------

//creating a temporaray table of the dataframe so that we can run SQL queries on it
bikes_data_DF.createOrReplaceTempView("bikesTable")

// COMMAND ----------

/*
We want to find the routes that are most ridden by bikers
We need to remember one thing that a route is denoted by the 'start station id' and 'end station id'.
So we group the routes on 'start station id' and 'end station id' and count the number of routes that are having a particular 'start station id' and 'end station id'.
Finally order them on decreasing order of count and print the first 30 routes having highest counts.
********************************
*/
//most ridden route by bikers
val most_ridden_route = spark.sql("select count(*) as route_count, `start station id`, `end station id` from bikesTable group by `start station id`, `end station id` order by route_count desc")
most_ridden_route.limit(30).show()
//most_ridden_route.count()

// COMMAND ----------

/*
We needfind the longest trip in terms of duration.

We select the trip which has the highest 'tripduration' convert it to days and also fetch the various details for that longest trip like 'starttime', 'stoptime', 'start station id', 'start station name', 'end station id', 'end station name'
*/
//longest trip in terms of duration
val longest_trip = spark.sql("select tripduration / (3600 * 24) as tripduration_in_days, `starttime`, `stoptime`, `start station id`, `start station name`, `end station id`, `end station name` from bikesTable where tripduration = (select max(tripduration) from bikesTable)")
longest_trip.show()

// COMMAND ----------

//We want to find the most popular stations in terms of the no. of times they are visited by our bikers. In order to do this we will create 2 dataframes, one having the start stations ordered in the no. of times they are visited, and the other having the end stations ordered in the no. of times they are visited. After that we are just going to join the two dataframes and find the total count.

//most popular start stations 
val popular_start_stations = spark.sql("select count(`start station id`) as start_station_count, `start station id`, `start station name` from bikesTable group by `start station id`, `start station name` order by start_station_count desc")
println("Displaying the top 10 most popular start stations")
popular_start_stations.limit(10).show()

//most popular end stations
val popular_end_stations = spark.sql("select count(`end station id`) as end_station_count, `end station id`, `end station name` from bikesTable group by `end station id`, `end station name` order by end_station_count desc")
println("Displaying the top 10 most popular end stations")
popular_end_stations.limit(10).show()

// COMMAND ----------

//joining the start stations and their counts and the end stations and their counts based on the the 'start station id' and 'end station id' into a combined DF

val combined_stations = popular_start_stations.as("a").join(popular_end_stations.as("b")).where($"a.`start station id`" === $"b.`end station id`")
combined_stations.show()

// COMMAND ----------

/*
Fetching the station id, station name and summing the start_station_count and end_station_count into the total count of no. of times the station was visited.
Then arranging them on the total count in descending order.
*/

combined_stations.createOrReplaceTempView("combinedStationsTable")
val most_popular_stations = spark.sql("select `start station id` as Station_Id, `start station name` as Station_Name, `start_station_count` + `end_station_count` as total_count from combinedStationsTable order by total_count desc")
most_popular_stations.limit(20).show()

// COMMAND ----------

//counting the total no. of stations in the DataFrame

most_popular_stations.count()

// COMMAND ----------

bikes_data_DF.printSchema()

// COMMAND ----------

//Finding the days of the week that are most popular among bikers
/*first we extract the date from the timestamp data using to_date method
Then from the date we extract the day for that date using dayofmonth method
For using these two methods we need to import two packages*/

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
val start_times = bikes_data_DF.select($"starttime")
val start_times_date = start_times.withColumn("date", to_date($"starttime")).withColumn("day_of_month", dayofmonth(col("date")))
start_times_date.show()

// COMMAND ----------

/*
We need to get the day of week from day of month 

*/

start_times_date.createOrReplaceTempView("start_times_date")
val days_of_week = spark.sql("select case when (day_of_month % 7 = 0) then ((day_of_month % 7) + 7) else (day_of_month % 7) end as day_of_week from start_times_date")
days_of_week.createOrReplaceTempView("days_of_week")
days_of_week.printSchema()
val results = spark.sql("select count(*) as count_by_day_of_week, day_of_week from days_of_week group by day_of_week order by count_by_day_of_week desc")
results.show()


// COMMAND ----------

//Using ML to find the gender of the bikers based on the patterns of the trip
/*
We will be using logistic regression for this purpose
Gender of bikers:-
0 --> Unknown
1 --> Male
2 --> Female

In the gender classification model at first we would want to extract the features that most contribute to the classification
In order for the features to be used by a ML algorithm the features are transformed and put into Feature Vectors, which are vectors of numbers representing the value for each feature.

Label --> Gender(0 or 1 or 2)
Features --> {tripduration, start_day_of_week, stop_day_of_week, start station id, end station id}

*/





val feature_DF_initial = bikes_data_DF.select($"gender", $"tripduration", $"start station id", $"end station id", $"starttime", $"stoptime", $"start station latitude", $"start station longitude", $"end station latitude", $"end station longitude").withColumn("start_day_of_month", dayofmonth(to_date($"starttime"))).withColumn("stop_day_of_month", dayofmonth(to_date($"stoptime")))

//feature_DF_initial.printSchema()
//feature_DF_initial.show()

feature_DF_initial.createOrReplaceTempView("new_features")


//Extarcting and filtering out the features.

val feature_DF = spark.sql("select gender, tripduration, `start station id`, `end station id`, `start station latitude`, `start station longitude`, `end station latitude`, `end station longitude`, case when (start_day_of_month % 7 = 0) then ((start_day_of_month % 7) + 7) else (start_day_of_month % 7) end as start_day_of_week, case when (stop_day_of_month % 7 = 0) then ((stop_day_of_month % 7) + 7) else (stop_day_of_month % 7) end as stop_day_of_week from new_features")
feature_DF.printSchema()

feature_DF.show()



// COMMAND ----------

/*
We start building our model over here.
*/

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.linalg.DenseVector

val featureCols = Array("tripduration", "start station id", "end station id", "start station latitude", "start station longitude", "end station latitude", "end station longitude", "start_day_of_week", "stop_day_of_week")
//set the input and output column names**
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
//return a dataframe with all of the  feature columns in  a vector column**
val feature_DF_2 = assembler.transform(feature_DF)
feature_DF_2.printSchema()

// COMMAND ----------

//  Create a label column with the StringIndexer**
val labelIndexer = new StringIndexer().setInputCol("gender").setOutputCol("label")
val training_data = labelIndexer.fit(feature_DF_2).transform(feature_DF_2)
training_data.printSchema()

// COMMAND ----------

// create the classifier,  set parameters for training**
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
//  use logistic regression to train (fit) the model with the training data**
val model = lr.fit(training_data)    

// Print the coefficients and intercept for logistic regression**
println(s"Coefficients: ${model.coefficientMatrix} Intercept: ${model.interceptVector}")

// COMMAND ----------

//loading test dataset and doing some probing on the test dataset**

val test_data_DF = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").option("delimiter", ",").load("/FileStore/tables/2014*.csv")

test_data_DF.printSchema()
test_data_DF.cache()
test_data_DF.count()

// COMMAND ----------

/*Feature engineering on the test dataset in the same way performed on training dataset**
Extracting the features which we think will be the most suitable for our classification**
extracting the features in suitable formats**
*/

val test_DF_initial = test_data_DF.select($"gender", $"tripduration", $"start station id", $"end station id", $"starttime", $"stoptime", $"start station latitude", $"start station longitude", $"end station latitude", $"end station longitude").withColumn("start_day_of_month", dayofmonth(to_date($"starttime"))).withColumn("stop_day_of_month", dayofmonth(to_date($"stoptime")))


test_DF_initial.createOrReplaceTempView("test_table")

val test_DF_2 = spark.sql("select gender, tripduration, `start station id`, `end station id`, `start station latitude`, `start station longitude`, `end station latitude`, `end station longitude`, case when (start_day_of_month % 7 = 0) then ((start_day_of_month % 7) + 7) else (start_day_of_month % 7) end as start_day_of_week, case when (stop_day_of_month % 7 = 0) then ((stop_day_of_month % 7) + 7) else (stop_day_of_month % 7) end as stop_day_of_week from test_table")

test_DF_2.printSchema()
test_DF_2.show()

// COMMAND ----------

//return a dataframe with all of the  feature columns in  a vector column**
val test_DF_3 = assembler.transform(test_DF_2)
test_DF_3.printSchema()

// COMMAND ----------


val test_data = labelIndexer.fit(test_DF_3).transform(test_DF_3)
test_data.printSchema()

// COMMAND ----------

// run the  model on test features to get predictions**
val predictions = model.transform(test_data)
//As you can see, the previous model transform produced a new columns: rawPrediction, probablity and prediction.**
predictions.show
predictions.printSchema()

// COMMAND ----------

/*
Evaluating the predictions by calculating various metrics
*/
val lp = predictions.select( "label", "prediction")
val counttotal = predictions.count()
val correct = lp.filter($"label" === $"prediction").count()
val wrong = lp.filter(not($"label" === $"prediction")).count()

// COMMAND ----------


