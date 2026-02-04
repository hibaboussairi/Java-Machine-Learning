package ml.model_selection;

import java.util.Random;

public class DataUtils {

    public static class SplitResult {
        public double[][] trainSet;
        public double[][] testSet;
    }

    public static SplitResult trainTestSplit(
            double[][] dataset,
            double testRatio,//proportion de données à mettre dans l’ensemble test (ex: 0.2 = 20%)
            long seed) {

        int n = dataset.length;
        int testSize = (int) (n * testRatio);
        int trainSize = n - testSize;

        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        Random rand = new Random(seed);//Cree un tableau d’indices et le mélange aleatoirement
        for (int i = n - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        double[][] train = new double[trainSize][];
        double[][] test = new double[testSize][];

        for (int i = 0; i < trainSize; i++)
            train[i] = dataset[indices[i]];

        for (int i = 0; i < testSize; i++)
            test[i] = dataset[indices[i + trainSize]];

        SplitResult res = new SplitResult();
        res.trainSet = train;
        res.testSet = test;
        return res;
    }
}
