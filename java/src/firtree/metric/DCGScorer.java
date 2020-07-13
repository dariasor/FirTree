/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package firtree.metric;

import firtree.utilities.RankList;

/**
 * The trick is to handle tied items, i.e., the items with same predictions across cutoff k
 * 
 * @author Xiaojie Wang
 */
public class DCGScorer extends MetricScorer {
	protected static double[] discount; // Lazy cache
	
	public DCGScorer() {
		this.k = 10;
		// Initialize lazy cache if we have not already done so
		if (discount == null) {
			discount = new double[5000];
			for(int i = 0; i < discount.length; i ++) {
				discount[i] = 1.0 / (Math.log(i + 2) / Math.log(2));
			}
		}
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
		return getDCG(targets, predictions, size);
	}
	
	@Override
	public MetricScorer copy() {
		return new DCGScorer();
	}
	
	@Override
	public String name() {
		return "DCG@" + k;
	}
	
	// Note that the predictions are sorted in descending order
	public double getDCG(double[] targets, double[] predictions, int topK) {
		// topK <= targets.length and topK <= predictions.length
		// Tied items are at boundary topK-1, ..., tieK-1
		int tieK = topK;
		while (tieK < predictions.length && predictions[tieK - 1] == predictions[tieK]) {
			tieK += 1;
		}
		// TODO. Need to shrink topK because tied items may not start from topK
		
		double dcg = 0;
		int stop = 0;
		
		// The start needs to be consistent with an iteration
		int start = stop;
		double tiedGain = 0;
		double tiedDiscount = 0;
		
		while (true) {
			// tieK is at most predictions.length
			while (stop < tieK && predictions[stop] == predictions[start]) {
				tiedGain += targets[stop];
				// This condition is for `tied` items at the `boundary`
				if (stop < topK) {
					tiedDiscount += discount[stop];
				}
				stop += 1;
			}
			// stop is at most tieK and hence is at most predictions.length
			// scores at the range [start, stop) are the same
			double inc = tiedGain * tiedDiscount / (stop - start);
			dcg += inc;
			
			// An iteration needs to be consistent with the start
			start = stop;
			tiedGain = 0;
			tiedDiscount = 0;
			
			if (start >= tieK) {
				break;
			}
		}
		return dcg;
	}
	
	// Lazy cache
	protected double discount(int index) {
		if(index < discount.length) {
			return discount[index];
		}
		
		// We need to expand our cache
		int cacheSize = discount.length + 1000;
		while(cacheSize <= index) {
			cacheSize += 1000;
		}
		double[] tmp = new double[cacheSize];
		System.arraycopy(discount, 0, tmp, 0, discount.length);
		for(int i = discount.length; i < tmp.length; i ++) {
			tmp[i] = 1.0 / (Math.log(i + 2) / Math.log(2));
		}
		discount = tmp;
		return discount[index];
	}

	@Override
	public double[][] swapChange(RankList rl) {
		// TODO Auto-generated method stub
		return null;
	}
}
