package firtree;

public class FeatureSplit {

	public Feature feature;
	public double[] splits;
	
	public FeatureSplit(Feature feature) {
		this.feature = feature;
		splits = new double[feature.centers.length - 1];
	}
	
}
