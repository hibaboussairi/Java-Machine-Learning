package ml.preprocessing;

public interface Preprocessor {

    void fit(double[][] dataset);

    double[][] transform(double[][] dataset);

    double[][] fitTransform(double[][] dataset);
}
