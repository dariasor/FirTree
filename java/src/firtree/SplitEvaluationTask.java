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
	FeatureSplit split;
	double splitPoint;
	double splitScore;
	
	public SplitEvaluationTask(InteractionTreeLearnerGAMMC app, String tmpDir, 
			Instances trainSet, Instances validSet, GAMLearner learner, Metric metric, int attIndex, FeatureSplit split, double splitPoint) {
		this.app = app;
		this.tmpDir = tmpDir;
		this.trainSet = trainSet;
		this.validSet = validSet;
		this.learner = learner;
		this.metric = metric;
		this.attIndex = attIndex;
		this.split = split;
		this.splitPoint = splitPoint;
	}

	@Override
	public Triple<Integer, Double, Double> call() throws Exception {
		long sleep = 16; // TODO: Set to 16
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
				InteractionTreeLearnerGAMMC.timeStamp(String.format("Evaluating feature %d split %f is successful in %d min", attIndex, splitPoint, elapse));
				break;
			} catch (OutOfMemoryError rerun) {
				InteractionTreeLearnerGAMMC.timeStamp(String.format("Evaluating feature %d split %f will be rerun in %d min", attIndex, splitPoint, sleep));
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
