package mltk.predictor.evaluation;

import mltk.core.Instances;
import mltk.core.Pointers;

/**
 * Class for evaluating error rate.
 * 
 * @author Yin Lou, modified by Xiaojie Wang
 *
 */
public class Error extends Metric {

	/**
	 * Constructor.
	 */
	public Error() {
		super(false);
	}

	@Override
	public double eval(double[] preds, double[] targets) {
		double error = 0;
		for (int i = 0; i < preds.length; i++) {
			// Handles both probability and predicted label
			double cls = preds[i] <= 0 ? 0 : 1;
			if (cls != targets[i]) {
				error++;
			}
		}
		return error / preds.length;
	}

	@Override
	public double eval(double[] preds, double[] targets, double[] weights) {
		System.out.println("ERROR: have not implemented yet and use other eval instead");
		System.exit(1);
		return 0;
	}

	@Override
	public double eval(double[] preds, Instances instances, Pointers pointers) {
		System.err.println("Cannot call Error.eval");
		System.exit(1);
		return 0;
	}
	
	@Override
	public double eval(double[] preds, Instances instances) {
		double error = 0;
		for (int i = 0; i < preds.length; i++) {
			// Handles both probability and predicted label
			double cls = preds[i] <= 0 ? 0 : 1;
			if (cls != instances.get(i).getTarget()) {
				error++;
			}
		}
		return error / preds.length;
	}

}
