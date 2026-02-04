package ml.linear;

import ml.core.MLModel;
import ml.metrics.Metrics;

public class LinearRegression extends  MLModel {
private double slope;
private double intercept;
private double learningRate;
private int numEpochs;
//Constructeur par defaut 
public LinearRegression() {
	super("Linear Regression");
    this.slope=0.0;//pente
    this.intercept=0.0;//l'ordonnee a l'origine
    this.learningRate=0.01;//vitesse d'apprentissage
    this.numEpochs=1000;// nombre d’iterations pour la descente de gradient
    }
    
//Constructeur complet
public LinearRegression( double learningRate, int numEpochs) {//Permet de fixer les hyperparamètres
	super("Linear Regression");
	this.slope = 0.0;
    this.intercept = 0.0;
	this.learningRate = learningRate;
    this.numEpochs = numEpochs;}
@Override
public void train(double[][] dataset) { 
    if (!isDatasetValid(dataset)) {
        System.out.println("Dataset invalide !"); return;
    }
    initializeParameters();
    gradientDescentLoop(dataset);
}
@Override
//methode predict
public double predict(double[] input) {
    double x = input[0]; // 1 feature attendu
    return slope * x + intercept;
}
@Override
//methode score
public double score(double[][] testSet) {
	int n = testSet.length;
    double[] yTrue = new double[n];
    double[] yPred = new double[n];
    for (int i = 0; i < n; i++) {
        yTrue[i] = testSet[i][testSet[i].length - 1]; // cible = derniere colonne
        double x = testSet[i][0]; // 
        yPred[i] = predict(new double[]{x});
    }
    return Metrics.r2Score(yTrue, yPred);
}
private void initializeParameters() {
	slope = 0.0;
    intercept = 0.0;
}
private boolean isDatasetValid(double[][] dataset) {
	return dataset != null && dataset.length > 0 && dataset[0].length >= 2;
}
private void gradientDescentLoop(double[][] dataset) {
	 for(int epoch = 0; epoch < numEpochs; epoch++) {
		 double[] grads = computeGradients(dataset);
         updateParameters(grads[0], grads[1]);
         if (epoch % 100 == 0) {
             System.out.println("Epoch " + epoch + " | Cost = " + computeCost(dataset));
         }
	 }
}
private double[] computeGradients(double[][] dataset) {
		 int n = dataset.length;
	        double gradSlope = 0.0;
	        double gradIntercept = 0.0;
	        for (int i = 0; i < n; i++) {
	            double x = dataset[i][0];
	            double y = dataset[i][dataset[i].length - 1]; // cible
	            double yPred = slope * x + intercept;
	            gradSlope += (yPred - y) * x;
	            gradIntercept += (yPred - y);
	        }
	        gradSlope = (2.0 / n) * gradSlope;
	        gradIntercept = (2.0 / n) * gradIntercept;
	        return new double[]{gradSlope, gradIntercept};
	    }
//Mise a jour des paramètres
private void updateParameters(double gradSlope, double gradIntercept) {
		 slope=slope-learningRate*gradSlope;
		 intercept=intercept-learningRate*gradIntercept;
}
private double computeCost(double[][] dataset) {
	int n = dataset.length;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double x = dataset[i][0];
        double y = dataset[i][dataset[i].length - 1];
        double yPred = slope * x + intercept;
        sum += (yPred - y) * (yPred - y);
    }
    return sum / n;
}
//getters 
public double getSlope() { return slope; }
public double getIntercept() { return intercept; }
	
}
