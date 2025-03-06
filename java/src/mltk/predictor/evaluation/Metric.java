package mltk.predictor.evaluation;

/**
 * 
 * @author Xiaojie Wang
 *
 */
public abstract class Metric {
	protected boolean isLargerBetter;
		
	/**
	 * Constructor.
	 * 
	 * @param isLargerBetter <code>true</code> if larger value is better.
	 */
	public Metric(boolean isLargerBetter) {
		this.isLargerBetter = isLargerBetter;
	}

	/**
	 * Returns <code>true</code> if the first value is better.
	 * 
	 * @param a the 1st value.
	 * @param b the 2nd value.
	 * @return <code>true</code> if the first value is better.
	 */
	public boolean isFirstBetter(double a, double b) {
		if (isLargerBetter) {
			return a > b;
		} else {
			return a < b;
		}
	}
	
	public double worstValue() {
		if (isLargerBetter) {
			return Double.NEGATIVE_INFINITY;
		} else {
			return Double.POSITIVE_INFINITY;
		}
	}
}
