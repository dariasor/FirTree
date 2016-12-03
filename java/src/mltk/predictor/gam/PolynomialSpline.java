package mltk.predictor.gam;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.Arrays;

import mltk.util.ArrayUtils;

public class PolynomialSpline extends Spline {
	
	public PolynomialSpline() {
		
	}
	
	public PolynomialSpline(int deg, int attIndex) {
		w = new double[deg];
		intercept = 0;
		this.attIndex = attIndex;
		this.leftEdge = 0;
		this.rightEdge = 0;
	}
	
	public PolynomialSpline(int deg) {
		this(deg, -1);
	}
	
	public PolynomialSpline(double[] w, int attIndex) {
		this.w = w;
		intercept = 0;
		this.attIndex = attIndex;
		this.leftEdge = 0;
		this.rightEdge = 0;
	}
	
	public PolynomialSpline(double[] w) {
		this(w, -1);
	}
	
	public int degree() {
		return w.length;
	}

	@Override
	public void read(BufferedReader in) throws Exception {
		String line = in.readLine();
		String[] data = line.split(":");
		attIndex = Integer.parseInt(data[1]);
		
		line = in.readLine();
		data = line.split(":");
		leftEdge = Double.parseDouble(data[1]);

		line = in.readLine();
		data = line.split(":");
		rightEdge = Double.parseDouble(data[1]);

		line = in.readLine();
		data = line.split(":");
		intercept = Double.parseDouble(data[1]);
		
		in.readLine();
		line = in.readLine();
		w = ArrayUtils.doubleArrayFromString(line);
	}

	@Override
	public void write(PrintWriter out) throws Exception {
		out.printf("[Predictor:%s]\n", this.getClass().getCanonicalName());
		out.println("AttIndex:" + attIndex);
		out.println("LeftEdge:" + leftEdge);
		out.println("RightEdge:" + rightEdge);
		out.println("Intercept:" + intercept);
		out.println("Weights:" + w.length);
		out.println(Arrays.toString(w));
	}
	
	@Override
	public double regress(double x) {
		if(leftEdge < rightEdge) {
			if (x < leftEdge) {
				x = leftEdge;
			} else if (x > rightEdge) {
				x = rightEdge;
			}
		}
		double pred = 0;
		double t = 1;
		for (int i = 0; i < w.length; i++) {
			t *= x;
			pred += t * w[i];
		}
		return pred;
	}
	
}
