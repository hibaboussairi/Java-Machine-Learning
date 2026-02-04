package ml.app;

import ml.linear.LinearRegression;
import ml.knn.KNNRegression;
import ml.knn.KNNClassification;
import ml.model_selection.DataUtils;
import ml.preprocessing.Preprocessor;
import ml.preprocessing.MinMaxScaler;

public class Main {

    public static void main(String[] args) {

        System.out.println("==============================================");
        System.out.println(" MINI FRAMEWORK MACHINE LEARNING - JAVA ");
        System.out.println("==============================================\n");

        /* DATASET REGRESSION */
        /* y = 2x + bruit */

        double[][] regressionData = {
                {1,  2.1}, {2,  4.2}, {3,  6.0}, {4,  8.3}, {5, 10.1},
                {6, 12.2}, {7, 14.0}, {8, 16.4}, {9, 18.1}, {10, 20.3},
                {11, 22.0}, {12, 24.2}, {13, 26.1}, {14, 28.4}, {15, 30.0},
                {16, 32.3}, {17, 34.1}, {18, 36.2}, {19, 38.4}, {20, 40.1},
                {21, 42.0}, {22, 44.2}, {23, 46.1}, {24, 48.3}, {25, 50.0},
                {26, 52.2}, {27, 54.1}, {28, 56.4}, {29, 58.0}, {30, 60.3}
        };

        /* DATASET CLASSIFICATION */

        double[][] classificationData = {
                {1.0, 1.1, 0}, {1.2, 1.0, 0}, {1.3, 1.4, 0}, {1.5, 1.3, 0},
                {1.7, 1.6, 0}, {1.9, 1.8, 0}, {2.1, 2.0, 0}, {2.3, 2.1, 0},
                {5.0, 5.1, 1}, {5.2, 5.3, 1}, {5.4, 5.5, 1}, {5.6, 5.7, 1},
                {5.8, 5.9, 1}, {6.0, 6.1, 1}, {6.2, 6.3, 1}, {6.4, 6.5, 1}
        };

        double[] testRatios = {0.2, 0.3};
        double[] learningRates = {0.1, 0.01, 0.001};
        int[] ks = {1, 3, 5, 7};

        /* boucle testRatio */

        for (double ratio : testRatios) {

            System.out.println("\n========= testRatio = " + ratio + " =========\n");

            DataUtils.SplitResult splitReg =
                    DataUtils.trainTestSplit(regressionData, ratio, 42);

            /* LINEAR REGRESSION  */

            System.out.println("---- Linear Regression (sans preprocessing) ----");

            for (double lrRate : learningRates) {
                LinearRegression lr = new LinearRegression(lrRate, 1000);
                lr.train(splitReg.trainSet);
                double r2 = lr.score(splitReg.testSet);

                System.out.println("LR (lr=" + lrRate + ") R2 = " + r2);
            }

            System.out.println("\n---- Linear Regression (avec preprocessing) ----");

            Preprocessor scalerLR = new MinMaxScaler();
            double[][] trainLRScaled = scalerLR.fitTransform(splitReg.trainSet);
            double[][] testLRScaled = scalerLR.transform(splitReg.testSet);

            for (double lrRate : learningRates) {
                LinearRegression lr = new LinearRegression(lrRate, 1000);
                lr.train(trainLRScaled);
                double r2 = lr.score(testLRScaled);

                System.out.println("LR (lr=" + lrRate + ") R2 = " + r2);
            }

            /* KNN REGRESSION */

            System.out.println("\n---- KNN Regression (sans preprocessing) ----");

            for (int k : ks) {
                KNNRegression knn = new KNNRegression(k);
                knn.train(splitReg.trainSet);
                double r2 = knn.score(splitReg.testSet);

                System.out.println("KNN Reg (k=" + k + ") R2 = " + r2);
            }

            System.out.println("\n---- KNN Regression (avec preprocessing) ----");

            Preprocessor scalerKNN = new MinMaxScaler();
            double[][] trainKNN = scalerKNN.fitTransform(splitReg.trainSet);
            double[][] testKNN = scalerKNN.transform(splitReg.testSet);

            for (int k : ks) {
                KNNRegression knn = new KNNRegression(k);
                knn.train(trainKNN);
                double r2 = knn.score(testKNN);

                System.out.println("KNN Reg (k=" + k + ") R2 = " + r2);
            }

            /* KNN CLASSIFICATION */

            DataUtils.SplitResult splitCls =
                    DataUtils.trainTestSplit(classificationData, ratio, 7);

            System.out.println("\n---- KNN Classification (sans preprocessing) ----");

            for (int k : ks) {
                KNNClassification knn = new KNNClassification(k);
                knn.train(splitCls.trainSet);
                double acc = knn.score(splitCls.testSet);

                System.out.println("KNN Cls (k=" + k + ") Acc = " + acc);
            }

            System.out.println("\n---- KNN Classification (avec preprocessing) ----");

            Preprocessor scalerCls = new MinMaxScaler();
            double[][] trainCls = scalerCls.fitTransform(splitCls.trainSet);
            double[][] testCls = scalerCls.transform(splitCls.testSet);

            for (int k : ks) {
                KNNClassification knn = new KNNClassification(k);
                knn.train(trainCls);
                double acc = knn.score(testCls);

                System.out.println("KNN Cls (k=" + k + ") Acc = " + acc);
            }
        }

        System.out.println("\n==============================================");
        System.out.println(" FIN DU PROGRAMME ");
        System.out.println("==============================================");
    }
}
