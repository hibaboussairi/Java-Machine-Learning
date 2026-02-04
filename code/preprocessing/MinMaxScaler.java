package ml.preprocessing;

public class MinMaxScaler implements Preprocessor {

    private double[] min;
    private double[] max;
    private boolean fitted = false;

    @Override
    public void fit(double[][] dataset) {

        int cols = dataset[0].length - 1; 
        min = new double[cols];
        max = new double[cols];

        for (int j = 0; j < cols; j++) { 
            min[j] = Double.POSITIVE_INFINITY;
            max[j] = Double.NEGATIVE_INFINITY;

            for (int i = 0; i < dataset.length; i++) {
                min[j] = Math.min(min[j], dataset[i][j]);
                max[j] = Math.max(max[j], dataset[i][j]);
            }
        }
        fitted = true;
    }

    @Override
    public double[][] transform(double[][] dataset) {

        if (!fitted) {
            System.out.println("MinMaxScaler non fitted !");
            return dataset;
        }

        double[][] result = new double[dataset.length][dataset[0].length];

        for (int i = 0; i < dataset.length; i++) {
            for (int j = 0; j < dataset[i].length - 1; j++) {

                if (max[j] == min[j]) {
                    result[i][j] = 0;
                } else {
                    result[i][j] =
                            (dataset[i][j] - min[j]) / (max[j] - min[j]);
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
