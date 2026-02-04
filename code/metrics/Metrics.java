package ml.metrics;

public class Metrics {

    public static double r2Score(double[] yTrue, double[] yPred) {
        double mean = 0.0;
        for (double y : yTrue) mean += y;
        mean /= yTrue.length;

        double ssTot = 0.0, ssRes = 0.0;


        for (int i = 0; i < yTrue.length; i++) {
            ssRes += Math.pow(yTrue[i] - yPred[i], 2);
            ssTot += Math.pow(yTrue[i] - mean, 2);
        }
        return 1 - ssRes / ssTot;
    }

    public static double accuracy(double[] yTrue, double[] yPred) {
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (Math.round(yPred[i]) == Math.round(yTrue[i])) correct++;
        }
        return (double) correct / yTrue.length;
    }
}
