package firtree.metric;

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

}
