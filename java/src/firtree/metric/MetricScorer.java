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

    public void loadExternalRelevanceJudgment(final String qrelFile) {

    }

    public double score(final List<RankList> rl) {
        double score = 0.0;
        for (int i = 0; i < rl.size(); i++) {
            score += score(rl.get(i));
        }
        return score / rl.size();
    }

    protected int[] getRelevanceLabels(final RankList rl) {
        final int[] rel = new int[rl.size()];
        for (int i = 0; i < rl.size(); i++) {
            rel[i] = (int) rl.get(i).getTarget(); // XW
        }
        return rel;
    }

    public abstract double score(RankList rl);

    public abstract MetricScorer copy();

    public abstract String name();

    public abstract double[][] swapChange(RankList rl);

    // XW
	public double[] getTargets(RankList rl) {
		double[] targets = new double[rl.size()];
		for (int i = 0; i < rl.size(); i ++) {
			targets[i] = rl.get(i).getTarget();
		}
		return targets;
	}
	
	// XW
	public double[] getPredictions(RankList rl) {
		double[] predictions = new double[rl.size()];
		for (int i = 0; i < rl.size(); i ++) {
			predictions[i] = rl.get(i).getPrediction();
		}
		return predictions;
	}
}
