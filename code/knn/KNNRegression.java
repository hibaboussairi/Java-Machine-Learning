package ml.knn;


import ml.core.MLModel;
import ml.metrics.Metrics;

import java.util.Arrays;
import java.util.Comparator;

public class KNNRegression extends MLModel {
	private int k ;
	private double[][]trainingData;

public KNNRegression (int k) {
	 super("KNN Regression (k=" + k + ")");
     if (k <= 0) throw new IllegalArgumentException("k doit être > 0");
     this.k = k;
     this.trainingData = null;
	
}
public void train(double[][] dataset) {
	 this.trainingData = dataset;
}
public double predict(double[] input) {
	 if (trainingData == null) throw new IllegalStateException("Model not trained");
     double[][] distLabel = new double[trainingData.length][2];
     for (int i = 0; i < trainingData.length; i++) {
         double[] features = Arrays.copyOf(trainingData[i], trainingData[i].length - 1);
         double label = trainingData[i][trainingData[i].length - 1];
         double d = euclideanDistance(features, input);
         distLabel[i][0] = d;
         distLabel[i][1] = label;
     }
     //Tri par distance
     Arrays.sort(distLabel, Comparator.comparingDouble(a -> a[0]));
     int kk = Math.min(k, distLabel.length);
     double sum = 0.0;
     for (int i = 0; i < kk; i++) sum += distLabel[i][1];
     return sum / kk;
 }
@Override
public double score(double[][] testSet) {
    int n = testSet.length;
    double[] yTrue = new double[n];
    double[] yPred = new double[n];
    for (int i = 0; i < n; i++) {
        yTrue[i] = testSet[i][testSet[i].length - 1];
        double[] features = Arrays.copyOf(testSet[i], testSet[i].length - 1);
        yPred[i] = predict(features);
    }
    return Metrics.r2Score(yTrue, yPred);
}
private double euclideanDistance(double[] a, double[] b) {
    double sum = 0.0;
    for (int j = 0; j < a.length; j++) {
        double diff = a[j] - b[j];
        sum += diff * diff;
    }
    return Math.sqrt(sum);
}
}
