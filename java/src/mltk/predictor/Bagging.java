package mltk.predictor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.Pointer;
import mltk.core.Pointers;
import mltk.predictor.evaluation.Metric;
import mltk.util.Random;

/**
 * Class for creating bootstrap samples.
 * 
 * @author Yin Lou, modified by Xiaojie Wang
 * 
 */
public class Bagging {

	/**
	 * Returns a bootstrap sample.
	 * 
	 * @param pointers the pointers to original data set.
	 * @return a bootstrap sample.
	 */
	public static Pointers createBootstrapSample(Pointers pointers) {
		Random rand = Random.getInstance();
		Map<Integer, Integer> map = new HashMap<>();
		for (int i = 0; i < pointers.size(); i++) {
			int idx = rand.nextInt(pointers.size());
			if (!map.containsKey(idx)) {
				map.put(idx, 0);
			}
			map.put(idx, map.get(idx) + 1);
		}
		Pointers bag = new Pointers();
		for (Integer idx : map.keySet()) {
			int index = pointers.get(idx).getIndex();
			int weight = map.get(idx);
			Pointer pointer = new Pointer(index, weight);
			bag.add(pointer);
		}
		return bag;
	}
	
	/**
	 * Returns a bootstrap sample.
	 * 
	 * @param instances the data set.
	 * @return a bootstrap sample.
	 */
	public static Instances createBootstrapSample(Instances instances) {
		Random rand = Random.getInstance();
		Map<Integer, Integer> map = new HashMap<>();
		for (int i = 0; i < instances.size(); i++) {
			int idx = rand.nextInt(instances.size());
			if (!map.containsKey(idx)) {
				map.put(idx, 0);
			}
			map.put(idx, map.get(idx) + 1);
		}
		Instances bag = new Instances(instances.getAttributes(), instances.getTargetAttribute(), map.size());
		for (Integer idx : map.keySet()) {
			int weight = map.get(idx);
			Instance instance = instances.get(idx).clone();
			instance.setWeight(weight * instance.getWeight());
			bag.add(instance);
		}
		return bag;
	}

	/**
	 * Returns a bootstrap sample with out-of-bag samples.
	 * 
	 * @param instances the data set.
	 * @param bagIndices
	 * @param oobIndices
	 */
	public static void createBootstrapSample(Instances instances, Map<Integer, Integer> bagIndices,
			List<Integer> oobIndices) {
		Random rand = Random.getInstance();
		for (;;) {
			bagIndices.clear();
			oobIndices.clear();
			for (int i = 0; i < instances.size(); i++) {
				int idx = rand.nextInt(instances.size());
				if (!bagIndices.containsKey(idx)) {
					bagIndices.put(idx, 0);
				}
				bagIndices.put(idx, bagIndices.get(idx) + 1);
			}
			for (int i = 0; i < instances.size(); i++) {
				if (!bagIndices.containsKey(i)) {
					oobIndices.add(i);
				}
			}
			if (oobIndices.size() > 0) {
				break;
			}
		}
	}

	/**
	 * Returns a set of bags.
	 * 
	 * @param pointers the pointers to original dataset.
	 * @param baggingIter the number of bagging iterations.
	 * @return a set of bags.
	 */
	public static Pointers[] createBags(Pointers pointers, int baggingIter) {
		Pointers[] bags = null;
		if (baggingIter <= 0) {
			// No bagging
			bags = new Pointers[] { pointers };
		} else {
			bags = new Pointers[baggingIter];
			for (int i = 0; i < baggingIter; i++) {
				bags[i] = Bagging.createBootstrapSample(pointers);
			}
		}
		return bags;
	}
	
	/**
	 * Returns a set of bags.
	 * 
	 * @param instances the dataset.
	 * @param baggingIter the number of bagging iterations.
	 * @return a set of bags.
	 */
	public static Instances[] createBags(Instances instances, int baggingIter) {
		Instances[] bags = null;
		if (baggingIter <= 0) {
			// No bagging
			bags = new Instances[] { instances };
		} else {
			bags = new Instances[baggingIter];
			for (int i = 0; i < baggingIter; i++) {
				bags[i] = Bagging.createBootstrapSample(instances);
			}
		}
		return bags;
	}
	
	/**
	 * Returns <code>true</code> if the bagging converges.
	 *
	 * @param p the performance vector for each iteration of bagging.
	 * @return <code>true</code> if the bagging converges.
	 */
	public static boolean analyzeBagging(double[] p, Metric metric) {
		final int bn = p.length;
		if (p.length <= 20) {
			return false;
		}

		double bestPerf = p[bn - 1];
		double worstPerf = p[bn - 20];
		for (int i = bn - 20; i < bn; i++) {
			if (metric.isFirstBetter(p[i], bestPerf)) {
				bestPerf = p[i];
			}
			if (!metric.isFirstBetter(p[i], worstPerf)) {
				worstPerf = p[i];
			}
		}
		double relMaxMin = Math.abs(worstPerf - bestPerf) / worstPerf;
		double relImprov;
		if (metric.isFirstBetter(p[bn - 1], p[bn - 21])) {
			relImprov = Math.abs(p[bn - 21] - p[bn - 1]) / p[bn - 21];
		} else {
			// Overfitting
			relImprov = Double.NaN;
		}
		return relMaxMin < 0.02 && (Double.isNaN(relImprov) || relImprov < 0.005);
	}

	/**
	 * Returns <code>true</code> if the bagging converges.
	 *
	 * @param p the performance list for each iteration of bagging.
	 * @return <code>true</code> if the bagging converges.
	 */
	public static boolean analyzeBagging(List<Double> p, Metric metric) {
		if (p.size() <= 20) {
			return false;
		}

		final int bn = p.size();
		double bestPerf = p.get(bn - 1);
		double worstPerf = p.get(bn - 20);
		for (int i = bn - 20; i < bn; i++) {
			if (metric.isFirstBetter(p.get(i), bestPerf)) {
				bestPerf = p.get(i);
			}
			if (!metric.isFirstBetter(p.get(i), worstPerf)) {
				worstPerf = p.get(i);
			}
		}
		double relMaxMin = Math.abs(worstPerf - bestPerf) / worstPerf;
		double relImprov;
		if (metric.isFirstBetter(p.get(bn - 1), p.get(bn - 21))) {
			relImprov = Math.abs(p.get(bn - 21) - p.get(bn - 1)) / p.get(bn - 21);
		} else {
			// Overfitting
			relImprov = Double.NaN;
		}
		return relMaxMin < 0.02 && (Double.isNaN(relImprov) || relImprov < 0.005);
	}

}
