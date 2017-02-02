package spark.ML;

import static org.apache.spark.sql.functions.*;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;


//spark-submit --class spark.ML.ensembleLearning --master local[4] extraGrades-0.0.1-SNAPSHOT.jar input.txt 

public class ensembleLearning {
  public static void main(String[] args) {
	  
    SparkSession spark = SparkSession
      .builder()
      .appName("JavaDecisionTreeClassificationExample")
      .config("spark.sql.crossJoin.enabled", true)
      .getOrCreate();

    // reading and splitting the data-------------------------------------------------------------------->
    // Load the data stored in LIBSVM format as a DataFrame.
    Dataset<Row> data = spark
      .read()
      .format("libsvm")
      .load(args[0]);

    // Index labels, adding metadata to the label column. Fit on whole dataset to include all labels in index.
    StringIndexerModel labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data);

    // Automatically identify categorical features, and index them.
    VectorIndexerModel featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
      .fit(data);

    // Split the data into training and test sets (30% held out for testing).
    Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
    Dataset<Row> trainingData = splits[0];
    Dataset<Row> testData = splits[1];
    // reading and splitting the data DONE----------------------------------------------------------------->
    
    // DecisionTree model --------------------------------------------------------------------------------->
    DecisionTreeClassifier DecisionTreemodel = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures");

    // Convert indexed labels back to original labels.
    IndexToString labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels());

    // Chain indexers and tree in a Pipeline.
    Pipeline DecisionTreepipeline = new Pipeline()
      .setStages(new PipelineStage[]{labelIndexer, featureIndexer, DecisionTreemodel, labelConverter});
    
    // Train a DecisionTree model.
    PipelineModel dtModel = DecisionTreepipeline.fit(trainingData);
    Dataset<Row> dtPredictions = dtModel.transform(testData);
    
    // Logistic Regression model ------------------------------------------------------------------------------>
    LogisticRegression lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8);
    // Fit the model
    LogisticRegressionModel lrModel = lr.fit(trainingData);
    Dataset<Row> lrPredictions = lrModel.transform(testData);

    // NaiveBayes model --------------------------------------------------------------------------------------->
    NaiveBayes nb = new NaiveBayes();
    NaiveBayesModel nbModel = nb.fit(trainingData);
    Dataset<Row> nbPredictions = nbModel.transform(testData);
    
    // storing the results of individual classifiers in to a data frame --------------------------------------->
    Dataset<Row> finalPredictions = testData.select("label");
    finalPredictions = finalPredictions.withColumnRenamed("label", "realLabel");
    finalPredictions = finalPredictions.join(dtPredictions.select("predictedLabel"));
    finalPredictions = finalPredictions.withColumnRenamed("predictedLabel", "dtPrediction");
    finalPredictions = finalPredictions.join(lrPredictions.select("prediction"));
    finalPredictions = finalPredictions.withColumnRenamed("prediction", "lrPrediction");
    finalPredictions = finalPredictions.join(nbPredictions.select("prediction"));
    finalPredictions = finalPredictions.withColumnRenamed("prediction", "nbPrediction");
    
    // Majority Voting ----------------------------------------------------------------------------------------->
    // for majority voting I'm adding all the three labels and setting the final answer based on the sum (0 or 1 means 0, else 1)
    finalPredictions = finalPredictions.join(finalPredictions.select(finalPredictions.col("dtPrediction").plus(finalPredictions.col("lrPrediction").plus(finalPredictions.col("nbPrediction")))));
    finalPredictions = finalPredictions.withColumnRenamed("(dtPrediction + (lrPrediction + nbPrediction))", "Voting");
    // final result after majority voting
    finalPredictions = finalPredictions.withColumn("finalPrediction", when(finalPredictions.col("Voting").equalTo(0.0), 0.0)
    	    .when(finalPredictions.col("Voting").equalTo(1.0), 0.0)
    	    .otherwise(1.0));
    finalPredictions.show(20);
    
    // Select (prediction, true label) and compute test error.
    // This takes time, thus commented out
    /*MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("realLabel")
      .setPredictionCol("finalPrediction")
      .setMetricName("accuracy");
    double accuracy = evaluator.evaluate(finalPredictions);
    System.out.println("Test Error = " + (1.0 - accuracy));*/
    
    spark.stop();
  }
}
