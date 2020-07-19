package firtree;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Paths;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import firtree.metric.METRIC;
import firtree.metric.MetricScorer;
import firtree.metric.MetricScorerFactory;
import firtree.utilities.FileUtils;
import firtree.utilities.Instance;
import firtree.utilities.RankList;
import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.io.AttrInfo;
import mltk.core.io.AttributesReader;
import mltk.core.io.InstancesReader;
import mltk.util.tuple.IntPair;

/**
 * Optimize any metrics by a coordinate ascent algorithm customized for FirTree models
 * 
 * @author Xiaojie Wang
 */
public class CoorAscentOnLeaves {

	static class Options {
		// The following three arguments come from RegressionOnLeaves
		@Argument(name = "-d", description = "model directory", required = true)
		String dir = ""; //path up to FirTree/

		@Argument(name = "-r", description = "attribute file", required = true)
		String attPath = "";

		@Argument(name = "-y", description = "polynomial degree")
		int polyDegree = 2;
		
		// This argument comes from InteractionTreeLearnerGAMMC but is required
		@Argument(name = "-g", description = "name of the attribute with the group id", required = true)
		String group = "";
		
		@Argument(name = "-m", description = "Prefix of name of output parameter files (default: model)")
		String modelPrefix = "model";
	}
	
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(CoorAscentOnLeaves.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		
		long start = System.currentTimeMillis();

		// XW. OLS is better than uniform in initializing parameters of CA
		RegressionOnLeaves.main(args);
		
		// Load attribute file
		AttrInfo ainfo = AttributesReader.read(opts.attPath);
		
		// Load tree structure and initial parameter values
		FirTree model = new FirTree(ainfo, opts.dir, opts.polyDegree, opts.modelPrefix);
		
		// Load training data
		Map<String, RankList> rankLists = loadRankList(opts, ainfo, model);

		// Fine-tune parameter values of leaf nodes of type MODEL
		MetricScorerFactory factory = new MetricScorerFactory();
		MetricScorer scorer = factory.createScorer(METRIC.NDCG);
		fineTune(opts, model, rankLists, scorer);
		
		// Save final parameter values of leaf nodes of type MODEL
		model.save();
		
		long end = System.currentTimeMillis();
		System.out.println("Finished all in " + (end - start) / 1000.0 + " (s).");
	}
	
	// The hyper-parameters of training model parameters by coordinate ascent
	// delta = [ deltaUnit * deltaBase^0, ..., deltaUnit * deltaBase^deltaMaxPower ]
	public static double deltaUnit = 0.001;
	public static double deltaBase = 2.0;
	public static double deltaMaxPower = 10; // A smaller value speeds up training
	public static double minGainTrain = 0.0001; // A larger value speeds up training
	public static double deltaRatio = 0.01;
	
	protected static void fineTune(
			Options opts,
			FirTree model, 
			Map<String, RankList> rankLists,
			MetricScorer scorer
			) {
		// Create log directory and delete all previous log files
		String logPath = opts.dir + "_CA";
		File logDir = new File(logPath);
		if (! logDir.exists())
			logDir.mkdirs();
		else 
			for (File logFile : logDir.listFiles())
				logFile.delete();
		
		int nIter = 0;
		while (true) {
			// Reset cached predictions helps prevent numerical issues 
			double scoreTrain = getScore(model, rankLists, scorer);
			System.out.printf("Training %s is %f at the start of iteration %d\n", 
					scorer.name(), scoreTrain, nIter);

			double startScoreTrain = scoreTrain;

			// Reset cached predictions is disabled
			List<IntPair> idPairs = model.getShuffledIdPairs();
			for (int i = 0; i < idPairs.size(); i ++) {
				IntPair idPair = idPairs.get(i);
				int activeNode = idPair.v1;
				int activeParam = idPair.v2;

				double origParamValue = model.getParamValue(activeNode, activeParam);
				double bestParamValue = origParamValue;
				double bestScoreTrain = scoreTrain;
				
				// Log parameter value and score
				String strToWrite = String.format("Step\tValue\t%s\n", scorer.name());
				String strBest = String.format("%d\t%s\t%s\n", 0, origParamValue, scoreTrain);
				strToWrite += strBest;
				
				double posDelta = initDelta(origParamValue, 1);
				for (int j = 0; j < deltaMaxPower + 1; j ++) {
					double paramDelta = posDelta;
					
					///* Ablative Debug Start
					scoreTrain = getScore(
							model, rankLists, scorer, activeNode, activeParam, paramDelta);
					//*/
					/*// This code snippet is time-consuming
					model.setParamValue(activeNode, activeParam, paramDelta);
					scoreTrain = getScore(model, rankLists, scorer);
					*/// Ablative Debug End
					
					// Log parameter value and score
					String strCur = String.format(
							"%d\t%s\t%s\n", 
							+ j + 1, 
							model.getParamValue(activeNode, activeParam), 
							scoreTrain
							);
					strToWrite += strCur;
					
					if (scoreTrain > bestScoreTrain) {
						bestParamValue = model.getParamValue(activeNode, activeParam);
						bestScoreTrain = scoreTrain;
						strBest = strCur;
					}
					posDelta *= deltaBase;
				}
				
				double negDelta = initDelta(origParamValue, -1);
				for (int j = 0; j < deltaMaxPower + 1; j ++) {
					double paramDelta = negDelta;
					if (j == 0) {
						// Need to first restore active parameter's value to its original value
						paramDelta += origParamValue - model.getParamValue(activeNode, activeParam);
					}
					
					///* Ablative Debug Start
					scoreTrain = getScore(
							model, rankLists, scorer, activeNode, activeParam, paramDelta);
					//*/
					/*// This code snippet is time-consuming
					model.setParamValue(activeNode, activeParam, paramDelta);
					scoreTrain = getScore(model, rankLists, scorer);
					*/// Ablative Debug End
					
					// Log parameter value and score
					String strCur = String.format(
							"%d\t%s\t%s\n", 
							- j - 1, 
							model.getParamValue(activeNode, activeParam), 
							scoreTrain
							);
					strToWrite += strCur;
					
					if (scoreTrain > bestScoreTrain) {
						bestParamValue = model.getParamValue(activeNode, activeParam);
						bestScoreTrain = scoreTrain;
						strBest = strCur;
					}
					negDelta *= deltaBase;
				}
				
				// Remove features easily causes numerical issues
	
				// Set the active parameter to the best value
				double paramDelta = bestParamValue - model.getParamValue(activeNode, activeParam);

				///* Ablative Debug Start
				scoreTrain = getScore(
						model, rankLists, scorer, activeNode, activeParam, paramDelta);
				//*/
				/*// This code snippet is time-consuming
				model.setParamValue(activeNode, activeParam, paramDelta);
				scoreTrain = getScore(model, rankLists, scorer);
				*/// Ablative Debug End
				
				System.out.printf("\t%s:%f orig:%+.12f best:%+.12f (%02d, %02d)\n", 
						scorer.name(), scoreTrain, 
						origParamValue, bestParamValue, 
						activeNode, activeParam
						);
				
				// Write parameter values and scores to file
				String nodeName = model.getNodeName(activeNode);
				String paramName = model.getParamName(activeNode, activeParam);
				String logName = String.format(
						"Iter-%02d_Pair-%03d_%s_%s", nIter, i, nodeName, paramName);
				strToWrite += strBest;
				String logFile = logPath + File.separator + logName + ".tsv";
				FileUtils.write(logFile, "ASCII", strToWrite);
			} // for (int i = 0; i < idPairs.size(); i ++)

			double gainTrain = scoreTrain - startScoreTrain;
			System.out.printf("\tIncrease training %s by %f (from %f to %f)\n", 
					scorer.name(), gainTrain, startScoreTrain, scoreTrain);
			nIter += 1;
			if (gainTrain < minGainTrain) {
				break;
			}
//			break;
		} // while (true)
	}
		
	protected static double getScore(
			FirTree model, 
			Map<String, RankList> rankLists, 
			MetricScorer scorer,
			int activeNode,
			int activeParam,
			double paramDelta
			) {
		double total = 0;
		double size = 0;
		model.setParamValue(activeNode, activeParam, paramDelta);
		for (RankList rankList : rankLists.values()) {
			if (isActive(rankList, activeNode)) {
				// There is one of the rank list's instances that falls in the active node
				for (Instance instance : rankList.getInstances()) {
					///* Ablative Debug Start
					model.predict(instance, activeNode, activeParam, paramDelta);
					//*/
					/*// This code snippet is time-consuming
					model.predict(instance);
					*/// Ablative Debug End
				}
				double score = scorer.score(rankList);
				rankList.setScore(score); // Easily forgot
			}
			total += rankList.getScore();
			size += rankList.getWeight();
		}
		return total / size;
	}

	protected static double getScore(
			FirTree model, 
			Map<String, RankList> rankLists, 
			MetricScorer scorer
			) {
		double total = 0;
		double size = 0;
		for (RankList rankList : rankLists.values()) {
			for (Instance instance : rankList.getInstances())
				model.predict(instance);
			
			double score = scorer.score(rankList);
			rankList.setScore(score); // Easily forgot
			total += rankList.getScore();
			size += rankList.getWeight();
		}
		return total / size;
	}

	protected static boolean isActive(RankList rankList, int activeNode) {
		for (Instance instance : rankList.getInstances()) {
			if (instance.getNodeIndex() == activeNode)
				return true;
		}
		return false;
	}
	
	protected static double initDelta(double origParamValue, int sign) {
		double absDelta = deltaUnit;
		if (origParamValue != 0.0 && absDelta > 0.5 * Math.abs(origParamValue)) {
			absDelta = deltaRatio * Math.abs(origParamValue);
		}
		return absDelta * sign;
	}

	protected static Map<String, RankList> loadRankList(
			Options opts,
			AttrInfo ainfo,
			FirTree model
			) throws Exception {
		Map<String, RankList> rankLists = new HashMap<>();
		List<String> allLeaves = model.getAllLeaves();
		for (String leafName : allLeaves) {
			String dataPath = Paths.get(opts.dir, "Node_" + leafName, "fir.dta").toString();
			String attPath = Paths.get(opts.dir, "Node_" + leafName, "fir.fs.fs.attr").toString();
			AttrInfo ainfoLeaf = AttributesReader.read(attPath);
			BufferedReader br = new BufferedReader(new FileReader(dataPath));
			for (String line = br.readLine(); line != null; line = br.readLine()) {
				String[] data = line.split("\t");
				Instance instance = new Instance(
						InstancesReader.parseDenseInstance(data, ainfoLeaf, false),
						ainfoLeaf
						);
				String groupId = data[ainfo.nameToCol.get(opts.group)];
				instance.setGroupId(groupId);
				
				// Predict index of the node (must be a leaf) the instance falls in
				int nodeIndex = model.indexLeaf(instance, data);
				String nodeName = model.getNodeName(nodeIndex);
				if (! nodeName.equals(leafName)) {
					System.err.printf("Not all instances in directory %s fall in node %s\n", leafName, nodeName);
					System.exit(1);
				}
				instance.setNodeIndex(nodeIndex);
				
				// Add the instance to the rank list with the same groupId
				if (! rankLists.containsKey(groupId)) {
					rankLists.put(groupId, new RankList(groupId));
				}
				rankLists.get(groupId).add(instance);
				
				if (Math.abs(model.predict(line) - model.predict(instance)) > Math.pow(10, -10)) {
					System.err.println("Diffferent versions of FirTree.predict are inconsistent");
					System.exit(1);
				}			
			}
			br.close();
		}
		
		// Set the weight of a rank list
		for (RankList rankList : rankLists.values())
			rankList.setWeight();
		return rankLists;
	}
	
	protected static void timeStamp(String msg) {
		Date tmpDate = new Date();
		System.out.println("TIMESTAMP >>>> ".concat(tmpDate.toString()).concat(": ").concat(msg));
	}	
}
