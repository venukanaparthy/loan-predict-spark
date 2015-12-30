package com.esri.spark

import org.apache.spark._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import au.com.bytecode.opencsv.CSVParser
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{L1Updater, SquaredL2Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.SparseVector
import scala.math.{abs, max, min, log}
import scalax.chart.module.Charting.XYLineChart
import scala.collection.immutable.IndexedSeq

object LoanPredict {

  var loanStatusIndex = 0
  var isBorrowerHomeownerIndex = 0
  var isEmployedIndex = 0
  var employmentStatusDurationIndex = 0
  var loanOriginalAmountIndex = 0
  var amountDelinquentIndex = 0
  var delinquenciesLast7YearsIndex = 0
  var creditScoreRangeLowerIndex = 0
  var statedMonthlyIncomeIndex = 0
  var debtToIncomeRatioIndex = 0
  
  val logloss = (p:Double, y:Double) => - ( y * log(p) + (1-y) * log( 1 - p) )

  object RegType extends Enumeration {
    type RegType = Value
    val L1, L2 = Value
  }

  def isHeader(line : String): Boolean = line.contains("ListingKey")
 
  def mLine(line: String) = {
    val parser = new CSVParser(',')
    parser.parseLine(line)
  }
  
  def parse(data: Array[String]): LabeledPoint = {
    val loanStatus = data(loanStatusIndex)
    val loanGood = List("Current", "Completed", "FinalPaymentInProgress", "Cancelled").contains(loanStatus)

    val isBorrowerHomeowner = if (data(isBorrowerHomeownerIndex) == "True") 1.0 else 0.0
    val isEmployed = if (data(isEmployedIndex) == "Employed") 1.0 else 0.0
    var employmentStatusDuration = data(employmentStatusDurationIndex).toDouble
    val originalAmount = data(loanOriginalAmountIndex).toDouble
    val amountDelinquent = data(amountDelinquentIndex).toDouble
    val delinquenciesLast7Years = data(delinquenciesLast7YearsIndex).toDouble
    val creditScoreRangeLower = data(creditScoreRangeLowerIndex).toDouble
    val statedMonthlyIncome = data(statedMonthlyIncomeIndex).toDouble
    val incomeToAmountRatio = statedMonthlyIncome / originalAmount
    val debtToIncomeRatio = data(debtToIncomeRatioIndex).toDouble

    val label = if (loanGood) 1.0 else 0.0
    val numFeatures = 10
    val indices = Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    val values = Array(isBorrowerHomeowner, isEmployed, employmentStatusDuration,
      originalAmount, amountDelinquent,
      delinquenciesLast7Years, creditScoreRangeLower,
      statedMonthlyIncome, incomeToAmountRatio, debtToIncomeRatio)

    LabeledPoint(label, Vectors.sparse(numFeatures, indices, values))
  }
  
  def evaluateModel(model:LogisticRegressionModel , data:RDD[LabeledPoint]):Double = {
       
  
      data.map { case LabeledPoint (label, features) => 
                  logloss(model.predict(features), label)
                }.sum/data.count.toDouble      
  }

  def safe[S, T](f: S => T): S => Either[T, Exception] = {
    new Function[S, Either[T, Exception]] with Serializable {
      def apply(s: S): Either[T, Exception] = {
        try {
          Left(f(s))
        } catch {
          case e: Exception => Right(e)
        }
      }
    }
  }
  
  def main(args: Array[String]) {
    if (args.length < 1) {
      System.err.println("Usage LoanPredict <loan csv file>")
      return
    }
    
    var conf = new SparkConf;
    conf.setMaster("local")
         .setAppName("Loan Predict Spark")         
                 
    var sc = new SparkContext(conf)
               
    val loanFile = args(0)
    var loansRawRDD = sc.textFile(loanFile)
    
    //print sample data
    var loansRDD = loansRawRDD.filter { line => !isHeader(line) }    
    
    println("sample data: ")
    loansRDD take(3) foreach println

    //print headers  
    val headerAndRows = loansRawRDD.map { line => mLine(line) }
    val header = headerAndRows.first
    //println(header)
    //header foreach println

    loanStatusIndex = header.indexOf("LoanStatus")
    isEmployedIndex = header.indexOf("EmploymentStatus")
    employmentStatusDurationIndex = header.indexOf("EmploymentStatusDuration")
    isBorrowerHomeownerIndex = header.indexOf("IsBorrowerHomeowner")
    loanOriginalAmountIndex = header.indexOf("LoanOriginalAmount")
    amountDelinquentIndex = header.indexOf("AmountDelinquent")
    delinquenciesLast7YearsIndex = header.indexOf("DelinquenciesLast7Years")
    creditScoreRangeLowerIndex = header.indexOf("CreditScoreRangeLower")
    statedMonthlyIncomeIndex = header.indexOf("StatedMonthlyIncome")
    debtToIncomeRatioIndex = header.indexOf("DebtToIncomeRatio")
    
    val data = headerAndRows.filter(_(0) != header(0))

    val safeParse = safe(parse)
    val labeledPoints = data.map(safeParse)

    //cache data
    labeledPoints.cache()

    //parse and filter invalid data
    val labeledPointsBad = labeledPoints.collect({
            case t if t.isRight => t.right.get
    })
    
    val labeledPointsGood = labeledPoints.collect({
      case t if t.isLeft => t.left.get
    })
    
    //cache valid data points
    labeledPointsGood.cache()

    println("Number of valid input records " + labeledPointsGood.count)
    println("Typical input record:")
    println("  Good/Bad status " + (if (labeledPointsGood.first.label == 1.0) "Good" else "Bad"))
    val firstFeatureVector:Array[Double] = labeledPointsGood.first.features.toArray
    
    println("  IsBorrowerHomeowner " + firstFeatureVector(0))
    println("  IsEmployed " + firstFeatureVector(1))
    println("  employmentStatusDuration " + firstFeatureVector(2))
    println("  LoanOriginalAmount " + firstFeatureVector(3))
    println("  AmountDelinquent " + firstFeatureVector(4))
    println("  DelinquenciesLast7YearsIndex " + firstFeatureVector(5))
    println("  CreditScoreRangeLower " + firstFeatureVector(6))
    println("  StatedMonthlyIncome " + firstFeatureVector(7))
    println("  IncomeToAmountRatio " + firstFeatureVector(8))
    println("  DebtToIncomeRatio " + firstFeatureVector(9))

    
    //print labeled data
    labeledPointsGood.take(3).foreach(println)
    
    val splits = labeledPointsGood.randomSplit(Array(0.8, 0.1,0.1))
    val training = splits(0).cache()
    val validation = splits(1).cache()
    val test = splits(2).cache()
    
    val numTraining = training.count()
    val numVal = validation.count()
    val numTest = test.count()
    println(s"Training record count: $numTraining, validation record count:  $numVal , test record count: $numTest.")

    //unpersit - not needed anymore
    labeledPoints.unpersist(blocking = false)    
    labeledPointsGood.unpersist(blocking = false)

    //model params
    val numIterations = 100
    val regParam = 0.1
    val regType = RegType.L2
    
    val updater = regType match {
      case RegType.L1 => new L1Updater()
      case RegType.L2 => new SquaredL2Updater()
    }
    
    //base logistic regression model
    val model0 =  new LogisticRegressionWithLBFGS
    model0.optimizer
       .setNumIterations(50)       
       .setUpdater(updater)
       .setRegParam(1e-6)
    val currentModel0 = model0.run(training).clearThreshold()
    
    val loglossTraining = evaluateModel(currentModel0, training)
    println(f"Training logloss : $loglossTraining%.6f")
   
    val loglossValidation = evaluateModel(currentModel0, validation)
    println(f"Validation logloss : $loglossValidation%.6f")
    
    
    //perform grid search to find best model and hyper-parameters
    var bestLogLoss = 1e10
    var bestModel:LogisticRegressionModel = null;
    var currentModel:LogisticRegressionModel = null;    
    var loglossVal = 0.0
    val steps:List[Int] = List(1, 10)
    val regParams:List[Double] = List(1e-6, 1e-3)
    for (step <- steps) {
      for (regParam <- regParams) {
       println(f"Step : $step")
       val algorithm = new LogisticRegressionWithLBFGS
       algorithm.optimizer
       .setNumIterations(numIterations)       
       .setUpdater(updater)
       .setRegParam(regParam)
      
       currentModel = algorithm.run(training).clearThreshold()
       loglossVal = evaluateModel(currentModel, validation)
       println(f"Step size = $step, Regularization param = $regParam, Log Loss: $loglossVal%.6f ")   
       if (loglossVal < bestLogLoss){
          bestLogLoss = loglossVal      
          bestModel = currentModel
       }
      }
    }
    println(f"Best validation logloss $bestLogLoss%.6f")
    
    val testLogloss = evaluateModel(bestModel, test)
    println(f"Best test logloss $testLogloss%.6f")
    
    /*val model = algorithm.run(training).clearThreshold()
    // training logloss
    val loss = sc.accumulator(0.0, "LogLoss")
       
    //baseline prediction, always predict 1.0 irrespective of datapoint 
    val TrainOneFrac = training.filter {
              case LabeledPoint(label, features) =>  label == 1.0 
            }.count/training.count.toFloat
        
    println("TrainOneFrac :" +  TrainOneFrac)    
    val trainBaseLogLoss = training.map {
                case LabeledPoint(label, features) =>             
                  logloss(TrainOneFrac, label)              
                }.sum/training.count.toFloat
     
    println("Baseline training logloss: " + trainBaseLogLoss)
                
     val trainLogLoss = training.map {
                case LabeledPoint(label, features) =>             
                  logloss(model.predict(features), label)              
                }.sum/training.count.toFloat
     
    println("Training logloss: " + trainLogLoss)
   
    val TestOneFrac = test.filter {
                  case LabeledPoint(label, features) => label == 1.0
                }.count/test.count.toFloat
    println("TestOneFrac :" +  TrainOneFrac)          
                
    val testBaseLogLoss = test.map {  
                  case LabeledPoint(label, features) => 
                    logloss(TestOneFrac, label)
                }.sum/test.count.toFloat
                
    //testPredictionAndLabel.take(100).foreach(println) 
    println("Test Base logloss: " + testBaseLogLoss)
                
    val testLogLoss = test.map {  
                  case LabeledPoint(label, features) => 
                    logloss(model.predict(features), label)
                }.sum/test.count.toFloat
                
    //testPredictionAndLabel.take(100).foreach(println) 
    println("Test logloss: " + testLogLoss)
   */                             
    val prediction = bestModel.predict(test.map(_.features))     
    val predictionAndLabel = prediction.zip(test.map(_.label))
    
    println("Model Weights:")
    bestModel.weights.toArray.foreach { weight => println("  " + weight) }
      
    val metrics = new BinaryClassificationMetrics(predictionAndLabel)
    println(s"Test areaUnderPR = ${metrics.areaUnderPR()}.")
    
    //get the false-positives and true positives
    val rocRDDD = metrics.roc()  
    val rocArr = rocRDDD.collect.toVector

    //plot the basic ROC curve.
    val chart = XYLineChart(rocArr)
    chart.show("Roc Curve", (1920, 1080), false)

    sc.stop()
  }
}