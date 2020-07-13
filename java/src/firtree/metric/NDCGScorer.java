package firtree.metric;

import java.util.HashMap;
import java.util.Map;

import firtree.utilities.RankList;
import firtree.utilities.Sorter;

/**
 * @author Xiaojie Wang
 */
public class NDCGScorer extends DCGScorer {
	
	// Cache the ideal gain corresponding to group id
	protected Map<String, Double> idealGains;
	
	public NDCGScorer() {
		super();
		idealGains = new HashMap<>();
	}
	
	@Override
	public double score(RankList rl) {
		if (rl.size() == 0) {
			return 0;
		}

		int size = k;
		if (k > rl.size() || k <= 0) {
			size = rl.size();
		}

		double[] targets = getTargets(rl);
		double[] predictions = getPredictions(rl);
		
		double ideal = 0;
		Double d = idealGains.get(rl.getGroupId());
		if (d != null) {
			ideal = d;
		} else {
			ideal = getIdealDCG(targets, size);
			idealGains.put(rl.getGroupId(), ideal);
		}
		
		if(ideal <= 0.0) {
			return 0.0;
		}
		return getDCG(targets, predictions, size) / ideal;
	}
	
	@Override
	public MetricScorer copy() {
		return new NDCGScorer();
	}
	
	@Override
	public String name() {
		return "NDCG@" + k;
	}
	
	private double getIdealDCG(double[] targets, int topK) {
		int[] idx = Sorter.sort(targets, false);
		double dcg = 0;
		for (int i = 0; i < topK; i ++) {
			dcg += targets[idx[i]] * discount(i);
		}
		return dcg;
	}
	
}
