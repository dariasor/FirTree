package mltk.predictor.evaluation;

import java.util.Arrays;
import java.util.Comparator;

import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.Pointers;
import mltk.util.tuple.DoublePair;
import mltk.util.tuple.DoubleTriple;

/**
 * Class for evaluating area under ROC curve.
 * 
 * @author Yin Lou, modified by Xiaojie Wang
 *
 */
public class AUC extends Metric {

	private class DoublePairComparator implements Comparator<DoublePair> {

		@Override
		public int compare(DoublePair o1, DoublePair o2) {
			if (o1.v1 < o2.v1) {
				return -1;
			} else if (o1.v1 > o2.v1) {
				return 1;
			} else {
				if (o1.v2 < o2.v2) {
					return -1;
				} else if (o1.v2 > o2.v2) {
					return 1;
				} else {
					return 0;
				}
			}
		}

	}

	private class DoubleTripleComparator implements Comparator<DoubleTriple> {

		@Override
		public int compare(DoubleTriple o1, DoubleTriple o2) {
			if (o1.v1 < o2.v1) {
				return -1;
			} else if (o1.v1 > o2.v1) {
				return 1;
			} else {
				return 0;
			}
		}

	}

	/**
	 * Constructor.
	 */
	public AUC() {
		super(true);
	}

	@Override
	public double eval(double[] preds, double[] targets) {
		System.out.println("ERROR: do not support weights and use other eval instead");
		System.exit(1);
		DoublePair[] a = new DoublePair[preds.length];
		for (int i = 0; i < preds.length; i++) {
			a[i] = new DoublePair(preds[i], targets[i]);
		}
		return eval(a);
	}

	@Override
	public double eval(double[] preds, double[] targets, double[] weights) {
		DoubleTriple[] a = new DoubleTriple[preds.length];
		for (int i = 0; i < preds.length; i++) {
			a[i] = new DoubleTriple(preds[i], targets[i], weights[i]);
		}
		return eval(a);
	}

	@Override
	public double eval(double[] preds, Instances instances, Pointers pointers) {
		DoubleTriple[] a = new DoubleTriple[preds.length];
		for (int i = 0; i < preds.length; i++) {
			Instance instance = instances.get(pointers.get(i).getIndex());
			a[i] = new DoubleTriple(preds[i], instance.getTarget(), instance.getWeight());
		}
		return eval(a);
	}
	
	@Override
	public double eval(double[] preds, Instances instances) {
//		DoublePair[] a = new DoublePair[preds.length];
		DoubleTriple[] a = new DoubleTriple[preds.length];
		for (int i = 0; i < preds.length; i++) {
//			a[i] = new DoublePair(preds[i], instances.get(i).getTarget());
			a[i] = new DoubleTriple(preds[i], instances.get(i).getTarget(), instances.get(i).getWeight());
		}
		return eval(a);
	}
	
	protected double eval(DoublePair[] a) {
		System.out.println("ERROR: do not support weights and use other eval instead");
		System.exit(1);
		Arrays.sort(a, new DoublePairComparator());
		double[] fraction = new double[a.length];
		for (int idx = 0; idx < fraction.length;) {
			int begin = idx;
			double pos = 0;
			for (; idx < fraction.length && a[idx].v1 == a[begin].v1; idx++) {
				pos += a[idx].v2;
			}
			double frac = pos / (idx - begin);
			for (int i = begin; i < idx; i++) {
				fraction[i] = frac;
			}
		}

		double tt = 0;
		double tf = 0;
		double ft = 0;
		double ff = 0;

		for (int i = 0; i < a.length; i++) {
			tf += a[i].v2;
			ff += 1 - a[i].v2;
		}

		double area = 0;
		double tpfPrev = 0;
		double fpfPrev = 0;

		for (int i = a.length - 1; i >= 0; i--) {
			tt += fraction[i];
			tf -= fraction[i];
			ft += 1 - fraction[i];
			ff -= 1 - fraction[i];
			double tpf = tt / (tt + tf);
			double fpf = 1.0 - ff / (ft + ff);
			area += 0.5 * (tpf + tpfPrev) * (fpf - fpfPrev);
			tpfPrev = tpf;
			fpfPrev = fpf;
		}

		return area;
	}

	protected double eval(DoubleTriple[] a) {
		Arrays.sort(a, new DoubleTripleComparator());

		double tp = 0; // Variable tp is true positive, equivalent to tt
		double fp = 0; // Variable fp is false positive, equivalent to ft
		double tp_fn = 0; // Variable fn is false negative, equivalent to tf
		double fp_tn = 0; // Variable tn is true negative, equivalent to ff

		for (int i = 0; i < a.length; i++) {
			tp_fn += a[i].v2 * a[i].v3;
			fp_tn += (1 - a[i].v2) * a[i].v3;
		}

		double area = 0;
		double tprPrev = 0;
		double fprPrev = 0;
		int i = a.length - 1;
		while (i >= 0) {
			double threshold = a[i].v1;
			do {
				tp += a[i].v2 * a[i].v3;
				fp += (1 - a[i].v2) * a[i].v3;
				i --;
			} while (i >= 0 && a[i].v1 == threshold);

			// TODO: handle divide by zero here, also in eval(DoublePair[])
			double tpr = tp / tp_fn;
			double fpr = fp / fp_tn;

			area += 0.5 * (tpr + tprPrev) * (fpr - fprPrev);
			tprPrev = tpr;
			fprPrev = fpr;
//			System.out.printf("tp=%f fn=%f fp=%f tn=%f\n", tp, fn, fp, tn);
		}

		return area;
	}

	public String toString() {
		return "AUC";
	}

	public static void main(String[] args) throws Exception {
		// Consistent with sklearn.metrics.roc_auc_score(y_true, y_score, sample_weight)
		DoubleTriple[] a = new DoubleTriple[5];
		a[0] = new DoubleTriple(0.3, 0, 0.9);
		a[1] = new DoubleTriple(0.7, 1, 0.3);
		a[2] = new DoubleTriple(0.5, 1, 0.1);
		a[3] = new DoubleTriple(0.7, 0, 0.5);
		a[4] = new DoubleTriple(0.5, 1, 0.7);
		AUC metric = new AUC();
		double auc = metric.eval(a);
		System.out.printf("AUC=%.6f\n", auc);
	}
	
}
