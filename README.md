# ensembleLearning

Ensemble learning is the process by which multiple models, such as classifiers or experts, are strategically generated and combined to solve a particular computational intelligence problem.

In this example I use three different classifiers to make a new model for data classification and learning based on majority voting. The coding is done in JAVA and uses the SPARK to make the algorithm scalable and more efficient. I also use the ML library for SPARK to make the three models.
The three classifiers used are:
1. Decision Tree
2. Logistic Regression
3. Naive Bayes

The code outputs a snippet of the final output which shows the indivisual answer of the three classifiers, the majority voting and the output from the ensemble model compared to the real output values(y).
The code can also calculate the total error rate for the ensemble learning. However, this takes a long time and thus commented out.

Commmand to run the jar file is 
spark-submit --class spark.ML.ensembleLearning --master local[4] extraGrades-0.0.1-SNAPSHOT.jar input.txt 

