package mltk.predictor.evaluation;

import mltk.core.Instances;

/**
 * Class for evaluating root mean squared error (RMSE).
 * 
 * @author Yin Lou, modified by Xiaojie Wang
 *
 */
public class RMSE extends Metric {

	/**
	 * Constructor.
	 */
	public RMSE() {
		super(false);
	}

	@Override
	public double eval(double[] preds, double[] targets) {
		System.out.println("ERROR: do not support weights and use other eval instead");
		System.exit(1);
		double rmse = 0;
		for (int i = 0; i < preds.length; i++) {
			double d = targets[i] - preds[i];
			rmse += d * d;
		}
		rmse = Math.sqrt(rmse / preds.length);
		return rmse;
	}

	@Override
	public double eval(double[] preds, double[] targets, double[] weights) {
		double rmse = 0;
		double length = 0;
		for (int i = 0; i < preds.length; i++) {
			double d = targets[i] - preds[i];
			double w = weights[i];
			rmse += d * d * w;
			length += w;
		}
		rmse = Math.sqrt(rmse / length);
		return rmse;
	}

	@Override
	public double eval(double[] preds, Instances instances) {
		double rmse = 0;
		double length = 0;
		for (int i = 0; i < preds.length; i++) {
			double d = instances.get(i).getTarget() - preds[i];
			double w = instances.get(i).getWeight();
			rmse += d * d * w;
			length += w;
		}
		rmse = Math.sqrt(rmse / length);
		return rmse;
	}

	public String toString() {
		return "RMSE";
	}

	public static void main(String[] args) throws Exception {
		// Consistent with sklearn.metrics.mean_squared_error(y_true, y_score, sample_weight)
		double[] preds = new double[5];
		preds[0] = 0.3;
		preds[1] = 0.1;
		preds[2] = 0.9;
		preds[3] = 0.5;
		preds[4] = 0.7;
		java.util.List<mltk.core.Attribute> attributes = new java.util.ArrayList<mltk.core.Attribute>();
		Instances instances = new Instances(attributes);
		instances.add(new mltk.core.Instance(new double[0], 0, 0.9));
		instances.add(new mltk.core.Instance(new double[0], 1, 0.3));
		instances.add(new mltk.core.Instance(new double[0], 1, 0.1));
		instances.add(new mltk.core.Instance(new double[0], 0, 0.5));
		instances.add(new mltk.core.Instance(new double[0], 1, 0.7));
		RMSE metric = new RMSE();
		double rmse = metric.eval(preds, instances);
		System.out.printf("RMSE=%.6f\n", rmse * rmse);
	}
}
