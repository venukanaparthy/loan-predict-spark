# loan-predict-spark
Using Spark MLlib Logistic regression to evaluate loan data

Build:
mvn clean package

Run:
spark-submit --class com.esri.spark.LoanPredict target\spark-loan-predict-0.1.jar data\prosperLoanData.csv > out.txt
