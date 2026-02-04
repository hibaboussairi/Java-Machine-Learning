package ml.knn;

import ml.core.MLModel;
import ml.metrics.Metrics;
import java.util.Arrays;

public class KNNClassification extends MLModel {

    private int k;
    private double[][] trainingData;

    // CONSTRUCTEUR
    public KNNClassification(int k) {
        super("KNN Classification (k=" + k + ")");
        if (k <= 0) {
            throw new IllegalArgumentException("k doit être > 0");
        }
        this.k = k;
    }

    // TRAIN
    @Override
    public void train(double[][] dataset) {
        if (!isDatasetValid(dataset)) {
            System.out.println("Dataset invalide pour KNNClassification");
            return;
        }
        trainingData = dataset;
    }

    // PREDICT
    @Override
    public double predict(double[] input) {
        double[][] distances = new double[trainingData.length][2];
        for (int i = 0; i < trainingData.length; i++) {

            double[] features = Arrays.copyOf(
                    trainingData[i],
                    trainingData[i].length - 1
            );

            double label = trainingData[i][trainingData[i].length - 1];

            distances[i][0] = euclideanDistance(input, features);
            distances[i][1] = label;
        }

        // tri par distance croissante
        Arrays.sort(distances, (a, b) -> Double.compare(a[0], b[0]));

        // vote majoritaire(0 / 1)
        int count0 = 0;
        int count1 = 0;

        int neighbors = Math.min(k, trainingData.length);

        for (int i = 0; i < neighbors; i++) {
            int index = (int) distances[i][1];     // index du voisin
            double label = trainingData[index][trainingData[index].length - 1];

            if (label == 0.0) {
                count0++;
            } else {
                count1++;
            }
        }

        return (count1 > count0) ? 1.0 : 0.0;
    }
    //  SCORE
    @Override
    public double score(double[][] testSet) {

        int n = testSet.length;
        double[] yTrue = new double[n];
        double[] yPred = new double[n];

        for (int i = 0; i < n; i++) {

            yTrue[i] = testSet[i][testSet[i].length - 1];

            double[] input = Arrays.copyOf(
                    testSet[i],
                    testSet[i].length - 1
            );

            yPred[i] = predict(input);
        }

        return Metrics.accuracy(yTrue, yPred);
    }

    // DISTANCE
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    // VALIDATION
    private boolean isDatasetValid(double[][] dataset) {
        return dataset != null
                && dataset.length > 0
                && dataset[0].length >= 2;
    }
}
