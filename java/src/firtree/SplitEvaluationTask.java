package firtree;

import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;

import mltk.core.Instances;
import mltk.predictor.evaluation.Metric;
import mltk.predictor.gam.GAMLearner;
import mltk.util.tuple.Triple;

public class SplitEvaluationTask implements Callable<Triple<Integer, Double, Double>> {

	InteractionTreeLearnerGAMMC app;
	String tmpDir;
	Instances trainSet;
	Instances validSet;
	GAMLearner learner;
	Metric metric;
	int attIndex;
	String featureName;
	FeatureSplit split;
	double splitPoint;
	double splitScore;
	
	public SplitEvaluationTask(InteractionTreeLearnerGAMMC app, String tmpDir, 
			Instances trainSet, Instances validSet, GAMLearner learner, Metric metric, int attIndex, String featureName, FeatureSplit split, double splitPoint) {
		this.app = app;
		this.tmpDir = tmpDir;
		this.trainSet = trainSet;
		this.validSet = validSet;
		this.learner = learner;
		this.metric = metric;
		this.attIndex = attIndex;
		this.featureName = featureName;
		this.split = split;
		this.splitPoint = splitPoint;
	}

	@Override
	public Triple<Integer, Double, Double> call() throws Exception {
		long sleep = 16; 
		while (true) {
			try {
				long start = System.currentTimeMillis();
				
				// Memory hungry split evaluation
//				splitScore = app.evaluateSplit(tmpDir, 
//						trainSet, validSet, learner, metric, attIndex, split, splitPoint);
				
				// Memory efficient split evaluation
				splitScore = app.evaluateSplitInPlace(tmpDir, 
						trainSet, validSet, learner, metric, attIndex, split, splitPoint);
				
				long stop = System.currentTimeMillis();
				long elapse = (stop - start) / (1000 * 60) + 1;
				InteractionTreeLearnerGAMMC.timeStamp("Completed evaluation of feature " + featureName + " split " + splitPoint + " in " + elapse + " min");
				break;
			} catch (OutOfMemoryError rerun) {
				InteractionTreeLearnerGAMMC.timeStamp("Evaluation of feature " + featureName + " split " + splitPoint + " failed and will be rerun in " + sleep + " min");
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
		
		Triple<Integer, Double, Double> future = new Triple<>(attIndex, splitPoint, splitScore);
		return future;
	}

}
