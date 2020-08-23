package firtree;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;

import firtree.metric.MetricScorer;
import firtree.utilities.RankList;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.Pointer;
import mltk.core.Pointers;
import mltk.predictor.evaluation.Metric;
import mltk.predictor.gam.GAM;
import mltk.predictor.gam.GAMLearner;

public class GAMLearningTask implements Callable<GAMLearningResult> {
	
	boolean isParent;
	InteractionTreeLearnerGAMMC app;
	String tmpDir;
	Instances trainSet;
	Instances validSet;
	GAMLearner learner;
	Metric metric;
	MetricScorer scorer;
	int attIndex;
	String featureName;
	FeatureSplit split;
	double splitPoint;
	
	// Returned results
	double parentScore;
	double splitScore;
	
	public GAMLearningTask(
			InteractionTreeLearnerGAMMC app, 
			String tmpDir, 
			Instances trainSet, 
			Instances validSet, 
			GAMLearner learner, 
			Metric metric,
			MetricScorer scorer
			) {
		this.isParent = true;
		this.app = app;
		this.tmpDir = tmpDir;
		this.trainSet = trainSet;
		this.validSet = validSet;
		this.learner = learner;
		this.metric = metric;
		this.scorer = scorer;
	}
	
	public GAMLearningTask(
			InteractionTreeLearnerGAMMC app, 
			String tmpDir, 
			Instances trainSet, 
			Instances validSet, 
			GAMLearner learner, 
			Metric metric, 
			MetricScorer scorer,
			int attIndex, 
			String featureName, 
			FeatureSplit split, 
			double splitPoint
			) {
		this.isParent = false;
		this.app = app;
		this.tmpDir = tmpDir;
		this.trainSet = trainSet;
		this.validSet = validSet;
		this.learner = learner;
		this.metric = metric;
		this.scorer = scorer;
		this.attIndex = attIndex;
		this.featureName = featureName;
		this.split = split;
		this.splitPoint = splitPoint;
	}
	
	@Override
	public GAMLearningResult call() throws Exception {
		long sleep = 16; 
		while (true) {
			try {
				
				if (isParent) {
					// Learn a parent GAM

					long start = System.currentTimeMillis();
					
					Pointers trainPtr = new Pointers();
					for (int i = 0; i < trainSet.size(); i ++)
						trainPtr.add(new Pointer(i));
					Pointers validPtr = new Pointers();
					for (int i = 0; i < validSet.size(); i ++)
						validPtr.add(new Pointer(i));
					
					GAM gam;
					if(app.getRegression()) {
						gam = learner.buildRegressor(
								trainSet, 
								trainPtr,
								validSet,
								validPtr,
								InteractionTreeLearnerGAMMC.getMaxNumItersGAM(),
								InteractionTreeLearnerGAMMC.getMaxNumLeavesGAM()
								);
					} else {
						gam = learner.buildClassifier(
								trainSet, 
								trainPtr,
								validSet,
								validPtr,
								InteractionTreeLearnerGAMMC.getMaxNumItersGAM(),
								InteractionTreeLearnerGAMMC.getMaxNumLeavesGAM()
								);
					}
					
					if (scorer == null) {
						double[] targetsValid = new double[validSet.size()];
						double[] predsValid = new double[validSet.size()];
						double[] weightsValid = new double[validSet.size()];
						int vNo = 0;
						for (Instance instance : validSet) {
							predsValid[vNo] = gam.regress(instance);
							targetsValid[vNo] = instance.getTarget();
							weightsValid[vNo] = instance.getWeight();
							vNo++;
						}
						parentScore = metric.eval(predsValid, targetsValid, weightsValid);
					} else {
						Map<String, RankList> rankLists = new HashMap<String, RankList>();
						for (Instance allIns : validSet) {
							String groupId = allIns.getGroupId();
							if (! rankLists.containsKey(groupId)) {
								rankLists.put(groupId, new RankList(groupId));
							}
							firtree.utilities.Instance subIns = new firtree.utilities.Instance(allIns.getTarget());
							subIns.setPrediction(gam.regress(allIns));
							subIns.setWeight(allIns.getWeight());
							rankLists.get(groupId).add(subIns);
						}
						for (RankList rankList : rankLists.values()) {
							rankList.setWeight();
							
							// TODO: Remove
							if (Math.abs(rankList.getWeight() - 1.) > Math.pow(10, -10)) {
								System.err.println("GAMLearningTask TODO");
								System.exit(1);
							}
						}
						parentScore = scorer.score(rankLists);
						
						// TODO: Remove
						double avgSize = 0.;
						for (RankList rankList : rankLists.values())
							avgSize += rankList.size();
						avgSize /= rankLists.size();
						InteractionTreeLearnerGAMMC.timeStamp(String.format("Parent GAM has %d lists, each having %.2f points on average", rankLists.size(), avgSize));
					}
					
					long stop = System.currentTimeMillis();
					long elapse = (stop - start) / (1000 * 60) + 1;
					InteractionTreeLearnerGAMMC.timeStamp("Completed training of parent in " + elapse + " min");
				} else {
					// Learn two child GAMs
					
					long start = System.currentTimeMillis();
					// Memory hungry split evaluation
//					splitScore = app.evaluateSplit(tmpDir, 
//							trainSet, validSet, learner, metric, attIndex, split, splitPoint);
					// Memory efficient split evaluation
					splitScore = app.evaluateSplitInPlace(
							tmpDir, 
							trainSet, 
							validSet, 
							learner, 
							metric, 
							scorer,
							attIndex, 
							split, 
							splitPoint
							);
					long stop = System.currentTimeMillis();
					long elapse = (stop - start) / (1000 * 60) + 1;
					InteractionTreeLearnerGAMMC.timeStamp("Completed evaluation of feature " + featureName + " split " + splitPoint + " in " + elapse + " min");
				}
				
				break;
			} catch (OutOfMemoryError rerun) {
				if (isParent) {
					InteractionTreeLearnerGAMMC.timeStamp("Training of parent failed and will be rerun in " + sleep + " min");
				} else {
					InteractionTreeLearnerGAMMC.timeStamp("Evaluation of feature " + featureName + " split " + splitPoint + " failed and will be rerun in " + sleep + " min");
				}
				
				try {
					TimeUnit.MINUTES.sleep(sleep);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				java.util.Random g = new java.util.Random();
				int r = g.nextInt((4 - 1) + 1) + 1;
				sleep *= r; // Prevent dead lock
				
			}
		}
		
		GAMLearningResult result = new GAMLearningResult(isParent);
		if (isParent) {
			result.parentScore = parentScore;
		} else {
			result.attIndex = attIndex;
			result.splitPoint = splitPoint;
			result.splitScore = splitScore;
		}
		return result;
	}
}
