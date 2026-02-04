 package ml.core;

public  abstract class MLModel {
	protected String name; //nom du modele 
	
public MLModel(String name) {
	this.name=name;
}
public void printStatus() { //méthode utilitaire qui affiche que le modèle est prêt
	 System.out.println("Modèle : " + name + " (prêt)");
}
public abstract void train(double[][] dataset);
public abstract double predict(double[] input);//input contient uniquement les features 
public double[] predict(double[][] inputs){
	if (inputs == null) return null;
	double[] preds = new double[inputs.length];
    for (int i = 0; i < inputs.length; i++) {
        preds[i] = predict(inputs[i]);
    }
    return preds;
}
public abstract double score(double[][] testSet);
public String getName() {
	return this.name;
}
}
