package ml.preprocessing;

public class StandardScaler implements Preprocessor {

    private double[] mean;
    private double[] std;
    private boolean fitted = false;

    @Override
    public void fit(double[][] dataset) {

        int cols = dataset[0].length - 1;
        mean = new double[cols];
        std = new double[cols];

        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < dataset.length; i++) {
                sum += dataset[i][j];
            }
            mean[j] = sum / dataset.length;
        }

        for (int j = 0; j < cols; j++) {
            double variance = 0.0;
            for (int i = 0; i < dataset.length; i++) {
                variance += Math.pow(dataset[i][j] - mean[j], 2);
            }
            std[j] = Math.sqrt(variance / dataset.length);
        }

        fitted = true;
    }

    @Override
    public double[][] transform(double[][] dataset) {

        if (!fitted) {
            System.out.println("StandardScaler non fitted !");
            return dataset;
        }

        double[][] result = new double[dataset.length][dataset[0].length];

        for (int i = 0; i < dataset.length; i++) {
            for (int j = 0; j < dataset[i].length - 1; j++) {

                if (std[j] == 0) {
                    result[i][j] = 0;
                } else {
                    result[i][j] =
                            (dataset[i][j] - mean[j]) / std[j];
                }
            }
            result[i][dataset[i].length - 1] =
                    dataset[i][dataset[i].length - 1];
        }
        return result;
    }

    @Override
    public double[][] fitTransform(double[][] dataset) {
        fit(dataset);
        return transform(dataset);
    }
}

