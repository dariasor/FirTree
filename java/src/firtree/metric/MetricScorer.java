/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package firtree.metric;

import java.util.List;
import java.util.Map;

import firtree.utilities.RankList;

/**
 * A generic retrieval measure computation interface.
 * 
 * @author Van Dang
 * @author Xiaojie Wang
 */
public abstract class MetricScorer {
    /** The depth parameter, or how deep of a ranked list to use to score the measure. */
    protected int k = 10;

    public MetricScorer() {

    }

    /**
     * The depth parameter, or how deep of a ranked list to use to score the measure.
     * @param k the new depth for this measure.
     */
    public void setK(final int k) {
        this.k = k;
    }

    /** The depth parameter, or how deep of a ranked list to use to score the measure. */
    public int getK() {
        return k;
    }

    public double score(Map<String, RankList> rankLists) {
		double total = 0.;
		double weight = 0.;
		for (RankList rankList : rankLists.values()) {
			double score = score(rankList);
			if (! Double.isNaN(score)) {
				// When tp_fn and fp_tn are never 0 in computing AUC
				total += score;
				weight += rankList.getWeight();
			}
		}
		return total / weight;
    }

    public abstract double score(RankList rl);

    public abstract String name();

    // XW
	protected double[] getTargets(RankList rl) {
		double[] targets = new double[rl.size()];
		for (int i = 0; i < rl.size(); i ++)
			targets[i] = rl.get(i).getTarget();
		return targets;
	}
	
	// XW
	protected double[] getPredictions(RankList rl) {
		double[] predictions = new double[rl.size()];
		for (int i = 0; i < rl.size(); i ++)
			predictions[i] = rl.get(i).getPrediction();
		return predictions;
	}
	
	protected double[] getWeights(RankList rl) {
		double[] weights = new double[rl.size()];
		for (int i = 0; i < rl.size(); i ++)
			weights[i] = rl.get(i).getWeight();
		return weights;
	}
}
