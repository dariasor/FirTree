package mltk.predictor.evaluation;

import mltk.core.Instances;
import mltk.core.Pointers;

/**
 * Class for evaluation metrics.
 * 
 * @author Yin Lou, modified by Xiaojie Wang
 *
 */
public abstract class PointwiseMetric extends Metric {
	public PointwiseMetric(boolean isLargerBetter) {
		super(isLargerBetter);
	}
	
	/**
	 * Evaluates predictions on a dataset.
	 * 
	 * @param preds the predictions.
	 * @param instances the dataset.
	 * @return the evaluation measure.
	 */
	public abstract double eval(double[] preds, Instances instances, Pointers pointers);
	
	/**
	 * Evaluates predictions on a dataset.
	 * 
	 * @param preds the predictions.
	 * @param instances the dataset.
	 * @return the evaluation measure.
	 */
	public abstract double eval(double[] preds, Instances instances);

	/**
	 * Evaluates predictions given targets.
	 *
	 * @param preds the predictions.
	 * @param targets the targets.
	 * @return the evaluation measure.
	 */
	public abstract double eval(double[] preds, double[] targets);

	/**
	 * Evaluates predictions given targets with weights.
	 * 
	 * @param preds the predictions.
	 * @param targets the targets.
	 * @param weights the weights.
	 * @return the evaluation measure.
	 */
	public abstract double eval(double[] preds, double[] targets, double[] weights);
}
