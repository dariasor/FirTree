/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package firtree.metric;

import java.util.ArrayList;
import java.util.List;

import firtree.utilities.RankLibError;
import firtree.utilities.RankList;
import firtree.utilities.Sorter;

/**
 * The trick is to handle tied items, i.e., the items with same predictions across cutoff k
 * 
 * @author Xiaojie Wang
 */
public class DCGScorer extends MetricScorer {
	protected static double[] logrank; // Lazy cache
	protected static String gain_type = "exponential";
	
	public DCGScorer() {
		this.k = 10;
		// Initialize lazy cache if we have not already done so
		if (logrank == null) {
			logrank = new double[k]; // Test lazy cache by setting k to 2
			for(int idx = 0; idx < logrank.length; idx ++) {
				// rank starts with 1, idx starts with 0
				int rank = idx + 1;
				logrank[idx] = Math.log(rank + 1) / Math.log(2);
			}
		}
	}
	
	@Override
	public double score(RankList rl) {
		if (rl.size() == 0)
			return 0;
		
		double[] targets = getTargets(rl);
		double[] predictions = getPredictions(rl);
		return getDCG(targets, predictions);
	}
	
	@Override
	public String name() {
		return "DCG@" + k;
	}
	
	public double getDCG(double[] targets, double[] predictions) {
		if (targets.length != predictions.length) {
			System.err.println("The number of targets is not equal to that of predictions");
			System.exit(1);
		}
		if (targets.length == 0) {
			System.err.println("There is no instance in the rank list");
			System.exit(1);
		}
		
		int ndcg_at = k;
		int[] idx_to_pos = Sorter.sort(predictions, false);
		
		double dcg_prev_score = min(predictions) - 1.;
		// Gain values of items with the score of the current item
		List<Double> dcg_tied_gains = new ArrayList<>();
		// Sum of discount factors for items with the current item's score
		double dcg_tied_mult = 0.;

        double dcg = 0.;

        for (int idx = 0; idx < targets.length; idx ++) {
        	// rank starts with 1, idx starts with 0
        	int rank = idx + 1;
        	
        	double target = targets[idx_to_pos[idx]];
        	
        	double gain;
        	if (gain_type.equals("linear"))
        		gain = target;
        	else if (gain_type.equals("exponential"))
        		gain = Math.pow(2., target) - 1;
        	else
        		throw RankLibError.create(String.format(
        				"Invalid gain_type '%s'. Use 'linear' for linear gain or 'exponential' for exponential gain", 
        				gain_type
        				));
        	/*// TODO: Debug
        	System.out.printf("\t\t\t%d %f\n", idx, gain);
        	*/
        	
        	double score = predictions[idx_to_pos[idx]];
        	
        	if (Math.abs(score - dcg_prev_score) > Math.pow(10, -10)) {
        		if (dcg_tied_gains.size() != 0) {
        			// Expected DCG for a set of tied items can be computed by
        			// (sum of targets) x (sum of discount factors) / (number of tied items)
        			dcg += 1. * sum(dcg_tied_gains) * dcg_tied_mult / dcg_tied_gains.size();
        			dcg_tied_gains.clear();
        			
        			/*// TODO: Debug
        			System.out.printf("\t\t%d %f\n", idx, dcg);
        			*/
        		}
        		dcg_tied_mult = 0.;
        	}
        	
        	// Record all targets for tied ASINs, but only record discount
        	// factors for top max_rank positions
        	dcg_tied_gains.add(gain);
        	if (rank <= ndcg_at)
        		dcg_tied_mult += 1. / logrank(idx);
        	
        	dcg_prev_score = score;
        }

        // Account for possible tie in last query group
        if (dcg_tied_gains.size() != 0) {
            dcg += 1. * sum(dcg_tied_gains) * dcg_tied_mult / dcg_tied_gains.size();

			/*// TODO: Debug
			System.out.printf("\t\t%d %f %f\n", 
					targets.length, sum(dcg_tied_gains), dcg_tied_mult);
			*/
        }

        return dcg;
	}
	
	private double min(double[] predictions) {
		double min = predictions[0];
		for (int i = 1; i < predictions.length; i ++)
			min = Math.min(min, predictions[i]);
		return min;
	}
	
	private double sum(List<Double> dcg_tied_gains) {
		double sum = 0;
		for (Double dcg_tied_gain : dcg_tied_gains)
			sum += dcg_tied_gain;
		return sum;
	}
	
	// Lazy cache: 0 -> log(2), 1 -> log(3), ...
	protected double logrank(int index) {
		if (index < logrank.length)
			return logrank[index];
		
		// We need to expand our cache
		int cacheSize = logrank.length * 2;
		while (cacheSize <= index)
			cacheSize *= 2;
		
		double[] tmp = new double[cacheSize];
		System.arraycopy(logrank, 0, tmp, 0, logrank.length);
		for(int idx = logrank.length; idx < tmp.length; idx ++) {
        	// rank starts with 1, idx starts with 0
        	int rank = idx + 1;
			tmp[idx] = Math.log(rank + 1) / Math.log(2);
		}
		logrank = tmp;
		return logrank[index];
	}

	public static String getGainType() {
		return gain_type;
	}

	public static void setGainType(String gain_type) {
		DCGScorer.gain_type = gain_type;
	}
	
}
