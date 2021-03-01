package firtree.metric;

import firtree.utilities.Instance;
import firtree.utilities.RankList;
import mltk.predictor.evaluation.AUC;

/**
 * @author Xiaojie Wang
 */
public class GAUCScorer extends MetricScorer {

	AUC scorer = new AUC();
	
	@Override
	public double score(RankList rankList) {
		if (rankList.size() == 0)
			return 0.;

		double[] targets = getTargets(rankList);
		double[] predictions = getPredictions(rankList);
		double[] weights = getWeights(rankList);
		double auc = scorer.eval(predictions, targets, weights);
		return auc;
	}

	@Override
	public String name() {
		return "GAUC";
	}
	
	static void test_consistency_with_ag_scripts_group_roc(
			GAUCScorer scorer,
			String groupId,
			double[] predictions,
			double[] targets,
			double true_score
			) {
		RankList rankList = new RankList(groupId);
		for (int i = 0; i < predictions.length; i ++) {
			Instance instance = new Instance(
					new mltk.core.Instance(new double[0], targets[i]),
					null
					);
			instance.setPrediction(predictions[i]);
			rankList.add(instance);
		}
		
		double pred_score = scorer.score(rankList);
		if (Math.abs(pred_score - true_score) > Math.pow(10, -6)) {
			System.err.printf("%s fails due to %.10f (pred) != %.10f (true)\n", 
					groupId, pred_score, true_score);
		} else {
			System.out.printf("%s succeeds\n", groupId);
		}
		
	}
	
	public static void main(String[] args) throws Exception {
		GAUCScorer scorer = new GAUCScorer();
		double[] true_scores = new double[] {
				0.4,
				0.45,
				0.5,
				0.5,
				0.0909091,
				0.4,
				0.125,
				0.5
				};
		test_consistency_with_ag_scripts_group_roc(
				scorer,
				"Group id 0",
				new double[] {5, 3, 4, 2, 4, 1, 4},
				new double[] {0, 0, 0, 1, 0, 0, 1},
				true_scores[0]
				);
		test_consistency_with_ag_scripts_group_roc(
				scorer,
				"Group id 1",
				new double[] {5, 3, 4, 2, 5, 1, 2},
				new double[] {0, 0, 1, 1, 0, 0, 0},
				true_scores[1]
				);
		test_consistency_with_ag_scripts_group_roc(
				scorer,
				"Group id 2",
				new double[] {4, 4, 4, 2, 4, 1, 4},
				new double[] {0, 0, 0, 1, 1, 0, 1},
				true_scores[2]
				);
		test_consistency_with_ag_scripts_group_roc(
				scorer,
				"Group id 3",
				new double[] {4, 4, 4, 4, 4},
				new double[] {0, 0, 0, 1, 1},
				true_scores[3]
				);
		test_consistency_with_ag_scripts_group_roc(
				scorer,
				"Group id 4",
				new double[] {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1},
				new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1},
				true_scores[4]
				);
		test_consistency_with_ag_scripts_group_roc(
				scorer,
				"Group id 5",
				new double[] {5, 3, 4, 2, 4, 1, 4},
				new double[] {0, 0, 0, 1, 0, 0, 1},
				true_scores[5]
				);
		// Testing target greater than 10
		test_consistency_with_ag_scripts_group_roc(
				scorer,
				"Group id 6",
				new double[] {4, 5, 8, 9, 8, 8},
				new double[] {1, 1, 1, 0, 0, 1},
				true_scores[6]
				);
		// Testing scientific notation
		test_consistency_with_ag_scripts_group_roc(
				scorer,
				"Group id 7",
				new double[] {5, 0.5, 3},
				new double[] {0, 0, 1},
				true_scores[7]
				);
	}

}
