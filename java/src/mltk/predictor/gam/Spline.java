package mltk.predictor.gam;

import java.util.Arrays;

import mltk.core.Instance;
import mltk.predictor.Regressor;

public abstract class Spline implements Regressor {

	protected int attIndex;
	protected double[] w;
	protected double intercept;
	protected double leftEdge;
	protected double rightEdge;
	
	/**
	 * Returns the value at a given position of this spline.
	 * 
	 * @param x the input.
	 * @return the value at a given position of this spline.
	 */
	public abstract double regress(double x);
	
	@Override
	public double regress(Instance instance) {
		return regress(instance.getValue(attIndex));
	}
	
	public void reset() {
		Arrays.fill(w, 0.0);
	}
	
	public int getAttributeIndex() {
		return attIndex;
	}
	
	public void setAttributeIndex(int attIndex) {
		this.attIndex = attIndex;
	}
	
	public double getLeftEdge() {
		return leftEdge;
	}
	
	public void setLeftEdge(double leftEdge) {
		this.leftEdge = leftEdge;
	}
	
	public double getRightEdge() {
		return rightEdge;
	}
	
	public void setRightEdge(double rightEdge) {
		this.rightEdge = rightEdge;
	}

	public double getIntercept() {
		return intercept;
	}
	
	public void setIntercept(double intercept) {
		this.intercept = intercept;
	}
	
	public double[] getW() {
		return w;
	}
	
	public void setW(double[] w) {
		this.w = w;
	}
	
}
