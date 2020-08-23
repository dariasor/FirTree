package firtree.metric;

import java.util.HashMap;
import java.util.Map;

import firtree.utilities.Instance;
import firtree.utilities.RankLibError;
import firtree.utilities.RankList;
import firtree.utilities.Sorter;

/**
 * @author Xiaojie Wang
 */
public class NDCGScorer extends DCGScorer {
	
	// Cache the ideal gain corresponding to group id
	protected Map<String, Double> group_id_to_max_dcg;
	
	public NDCGScorer() {
		super();
		group_id_to_max_dcg = new HashMap<>();
	}
	
	@Override
	public double score(RankList rankList) {
		if (rankList.size() == 0)
			return 0.;

		double[] targets = getTargets(rankList);
		double[] predictions = getPredictions(rankList);
		
		double max_dcg = 0.;
		if (group_id_to_max_dcg.containsKey(rankList.getGroupId())) {
			max_dcg = group_id_to_max_dcg.get(rankList.getGroupId());
		} else {
			max_dcg = get_max_dcg(targets);
			group_id_to_max_dcg.put(rankList.getGroupId(), max_dcg);
		}
		
		if (max_dcg < Math.pow(10, -10))
			return 0.;

		double dcg = getDCG(targets, predictions);
		/*// TODO: Debug
		System.out.printf("\tDCG=%f IDCG=%f\n", dcg, max_dcg);
		*/
		
		return dcg / max_dcg;
	}
	
	@Override
	public String name() {
		return "NDCG@" + k;
	}
	
	private double get_max_dcg(double[] targets) {
		int[] idx_to_pos = Sorter.sort(targets, false);
		
		double max_dcg = 0.0;
		for (int idx = 0; idx < targets.length; idx ++) {
			// rank starts with 1, idx starts with 0
			int rank = idx + 1;
			
			// break if position is larger than ndcg_depth
			if (rank > k)
				break;
			
			double target = targets[idx_to_pos[idx]];
			
        	double gain;
        	if (gain_type.equals("linear"))
        		gain = target;
        	else if (gain_type.equals("exponential"))
        		gain = Math.pow(2.0, target) - 1;
        	else
        		throw RankLibError.create(String.format(
        				"Invalid gain_type '%s'. Use 'linear' for linear gain or 'exponential' for exponential gain", 
        				gain_type
        				));
        	
        	max_dcg += gain / logrank(idx);
		}
		
		return max_dcg;
	}
	
	static void test_calculate_ndcg_whenUnweightedData_thenAccurateNDCG(
			NDCGScorer scorer,
			String groupId,
			double[] predictions,
			double[] targets,
			double true_ndcg
			) {
		
		RankList rankList = new RankList(groupId);
		for (int i = 0; i < predictions.length; i ++) {
			Instance instance = new Instance(targets[i]);
			instance.setPrediction(predictions[i]);
			rankList.add(instance);
		}
		double pred_ndcg = scorer.score(rankList);
		if (Math.abs(pred_ndcg - true_ndcg) > Math.pow(10, -10)) {
			System.err.printf("%s fails due to %.10f (pred) != %.10f (true)\n", 
					groupId, pred_ndcg, true_ndcg);
			//System.exit(1);
		} else 
			System.out.printf("%s succeeds\n", groupId);
	}
	
	static void test(String gain_type, int k, double[] true_ndcg) {
		NDCGScorer.setGainType(gain_type);
		NDCGScorer scorer = new NDCGScorer();
		scorer.setK(k);
		int times = 1; // Used to test the cache of ideal gains
		for (int i = 0; i < times; i ++) {
			test_calculate_ndcg_whenUnweightedData_thenAccurateNDCG(
					scorer,
					"Group id 0",
					new double[] {5., 3., 4., 2., 4., 1., 4.},
					new double[] {0., 0., 0., 1.5, 0.5, 0., 1.},
					true_ndcg[0]
					);
			test_calculate_ndcg_whenUnweightedData_thenAccurateNDCG(
					scorer,
					"Group id 1",
					new double[] {5., 3., 4., 2., 5., 1., 2.},
					new double[] {0., 0., 1.2, 1.1, 0., 0., 0.},
					true_ndcg[1]
					);
			test_calculate_ndcg_whenUnweightedData_thenAccurateNDCG(
					scorer,
					"Group id 2",
					new double[] {4., 4., 4., 2., 4., 1., 4.},
					new double[] {0., 0., 0., 1., 1., 0., 1.},
					true_ndcg[2]
					);
			test_calculate_ndcg_whenUnweightedData_thenAccurateNDCG(
					scorer,
					"Group id 3",
					new double[] {4., 4., 4., 4., 4.},
					new double[] {0., 0., 0., 1., 1.},
					true_ndcg[3]
					);
			test_calculate_ndcg_whenUnweightedData_thenAccurateNDCG(
					scorer,
					"Group id 4",
					new double[] {10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 1., 1., 1.},
					new double[] {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.},
					true_ndcg[4]
					);
			test_calculate_ndcg_whenUnweightedData_thenAccurateNDCG(
					scorer,
					"Group id 5",
					new double[] {5., 3., 4., 2., 4., 1, 4},
					new double[] {0, 0, 0, 1.5, 0.5, 0, 1},
					true_ndcg[5]
					);
			// Testing target greater than 10
			test_calculate_ndcg_whenUnweightedData_thenAccurateNDCG(
					scorer,
					"Group id 6",
					new double[] {4., 5., 8., 9., 8., 8.},
					new double[] {3., 6., 10., 0., 0., 2.},
					true_ndcg[6]
					);
			// Testing scientific notation
			test_calculate_ndcg_whenUnweightedData_thenAccurateNDCG(
					scorer,
					"Group id 7",
					new double[] {5., 0.5, 3.},
					new double[] {6e-1, 8e-2, 1},
					true_ndcg[7]
					);
		} // for (int i = 0; i < 3; i ++)
	}
	
	public static void main(String[] args) {
		test("exponential", 10, new double[] {
				0.5203322959222417,
				0.5317565362380837,
				0.7206201105813624,
				0.7231357726898465,
				0.08861964339202387,
				0.5203322959222417,
				0.5254456210544878,
				0.8679843907541499
				});
		test("linear", 100, new double[] {
				0.5523531026111766,
				0.5325611891991362,
				0.7206201105813624,
				0.7231357726898465,
				0.3375054808531018,
				0.5523531026111766,
				0.5967798630475649,
				0.895930857984812
				});
	}
	
}
