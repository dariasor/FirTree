package firtree;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import firtree.metric.GAUCScorer;
import firtree.metric.MetricScorer;
import firtree.metric.NDCGScorer;
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
		// The following arguments come from OrdLeastSquaresOnLeaves
//		@Argument(name="-d", description="FirTree directory output by InteractionTreeLearnerGAMMC", required=true)
//		String dir = ""; // Usually path up to "FirTree" inclusive

		@Argument(name="-l", description="(cropped) treelog.txt which specifies a tree structure", required=true)
		String logPath = "";
		
		@Argument(name = "-r", description = "attribute file", required = true)
		String attPath = "";

		@Argument(name = "-y", description = "polynomial degree")
		int polyDegree = 2;
		
		// This argument comes from InteractionTreeLearnerGAMMC but is required
		@Argument(name = "-g", description = "name of the attribute with the group id", required = true)
		String group = "";
		
		@Argument(name = "-m", description = "Prefix of name of output parameter files (default: ca)")
		String modelPrefix = "ca_params_y2";
		
		// This argument comes from InteractionTreeLearnerGAMMC
		@Argument(name = "-c", description = "(gauc|ndcg) - metric to optimize (default: gauc)")
		String metricStr = "gauc";
		
		@Argument(name = "-a", description = "(params|minmax)")
		String algorithm = "params";
		
		@Argument(name = "-i", description = "Initialize model parameters by OLS or an existing model")
		String modelInitial = "";
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

		// XW. OLS is better than uniform in initializing parameters of CA (manually call OLS)
		if (! opts.modelInitial.equals("")) {
			String dir = Paths.get(opts.logPath).getParent().toString();
			for (File nodeFile : new File(dir).listFiles()) {
				if (nodeFile.getName().startsWith("Node_Root_")) {
					String nodeDir = nodeFile.toString();
					Path modelSrc = Paths.get(nodeDir, opts.modelInitial + ".txt");
					Path modelTgt = Paths.get(nodeDir, opts.modelPrefix + ".txt");
					if (modelSrc.toFile().exists()) {
						timeStamp(String.format("Copy model parameters from %s to %s", modelSrc, modelTgt));
						Files.copy(modelSrc, modelTgt, StandardCopyOption.REPLACE_EXISTING);
					}
					Path constSrc = Paths.get(nodeDir, opts.modelInitial + "_const.txt");
					Path constTgt = Paths.get(nodeDir, opts.modelPrefix + "_const.txt");
					if (constSrc.toFile().exists()) {
						timeStamp(String.format("Copy const parameters from %s to %s", constSrc, constTgt));
						Files.copy(constSrc, constTgt, StandardCopyOption.REPLACE_EXISTING);
					}
				}
			}
		} else {
			OrdLeastSquaresOnLeaves.main(args);
		}
		
		// Load attribute file
		AttrInfo ainfo = AttributesReader.read(opts.attPath);
		
		// Load tree structure and initial parameter values
		FirTree model = new FirTree(ainfo, opts.logPath, opts.polyDegree, opts.modelPrefix);
		for (int i = 0; i < model.nodeAttIdList.size(); i ++) {
			for (int j = 0; j < model.nodeAttIdList.get(i).size(); j ++) {
				if (model.nodeAttIdList.get(i).get(j) != model.lr_attr_ids.get(i).get(j)) {
					System.err.printf("Incosistent attribute order between treelog.txt and parameter files %d %d\n",
							model.nodeAttIdList.get(i).get(j), model.lr_attr_ids.get(i).get(j));
					System.err.printf("%d %d %d -> %s\n", i, j, model.nodeAttIdList.get(i).get(j), ainfo.idToName(model.nodeAttIdList.get(i).get(j)));
					System.err.printf("%d %d %d -> %s\n", i, j, model.lr_attr_ids.get(i).get(j), ainfo.idToName(model.lr_attr_ids.get(i).get(j)));
					System.exit(1);
				}
			}
		}
		
		// Load training data
		Map<String, RankList> rankLists = loadRankList(opts, ainfo, model);

		MetricScorer scorer;
		if (opts.metricStr.equals("gauc")) {
			scorer = new GAUCScorer();
		} else {
			scorer = new NDCGScorer();
		}

		if (opts.algorithm.equals("params")) {
			timeStamp("Tune values of model parameters");
			// Tune parameter values of leaf nodes of type MODEL
			tuneParams(opts, model, rankLists, scorer);
		} else if (opts.algorithm.equals("minmax")) {
			timeStamp("Tune min and max values of features");
			// Tune min and max values of features
			tuneMinMax(opts, model, rankLists, scorer);
		}
		
		long end = System.currentTimeMillis();
		System.out.println("Finished all in " + (end - start) / 1000.0 + " (s).");
	}
	
	protected static int indexMax = 5000;
	protected static int indexNum = 50;
	protected static double minGainBound = 0.00001;
	
	protected static void tuneMinMax(
			Options opts,
			FirTree model, 
			Map<String, RankList> rankLists,
			MetricScorer scorer
			) throws Exception {
		timeStamp("Start to tune min and max values");
		boolean verbose = false; // Print debugging information, which is verbose
		boolean correct = false; // Use time-consuming but correct implementation
		double tolerance = 5 * Math.pow(10, -3);
		
		// Save the original min and max values to end of coefficients
		model.bakMinmax();
		
		int nIter = 0;
		while (true) {
			// Reset cached predictions helps prevent numerical issues 
			double scoreTrain = getScore(model, rankLists, scorer);
			timeStamp(String.format("Training %s is %f at the start of iteration %d", 
					scorer.name(), scoreTrain, nIter));
			//System.exit(0);
			
			double startScoreTrain = scoreTrain;

			// Reset cached predictions is disabled
			List<IntPair> idPairs = model.getBoundIdPairs();
			for (int i = 0; i < idPairs.size(); i ++) {
				IntPair idPair = idPairs.get(i);
				int activeNode = idPair.v1;
				int activeAtt = idPair.v2;

				double bestMin = model.getMinValue(activeNode, activeAtt);
				double bestMax = model.getMaxValue(activeNode, activeAtt);
				double bestScoreTrain = scoreTrain;
				
				int minIndex = model.getMinIndex(activeNode, activeAtt);
				int maxIndex = model.getMaxIndex(activeNode, activeAtt);
				
				int attId = model.lr_attr_ids.get(activeNode).get(activeAtt);
				int indexUnit = Math.max(1, model.attIdToValList.get(attId).size() / indexMax);
				
				if (verbose) {
					System.out.printf("  #%d node, #%d att is %s (id:%d which is not col)\n", 
							activeNode, activeAtt, model.ainfo.idToName(attId), attId);
					System.out.printf("  #%d is min %.0f, #%d is max %.0f, index unit:%d\n", 
							minIndex, bestMin, maxIndex, bestMax, indexUnit);
				}
				
				double prevMin = bestMin;
				for (int j = 0; j < indexNum; j ++) {
					int curIndex = minIndex + j * indexUnit;
					double currMin = model.attIdToValList.get(attId).get(curIndex);
					if (currMin >= bestMax) {
						break;
					}
					double minDelta = currMin - prevMin;
					
					///* Ablative Debug Start
					if (correct) {
						// This code snippet is time-consuming
						model.setMinValue(activeNode, activeAtt, minDelta);
						scoreTrain = getScore(model, rankLists, scorer);
					} else {
						scoreTrain = getScore(
								model, rankLists, scorer, activeNode, activeAtt, minDelta, "min");
					}
					//*/ Ablative Debug End
					///*
					if (verbose) {
						System.out.printf("    #%02d %07d %07d=%07d %.16f (min)\n", j, curIndex, 
								(int) currMin, (int) model.getMinValue(activeNode, activeAtt), scoreTrain);
					}
					//*/
					
					if (scoreTrain > bestScoreTrain) {
						bestMin = model.getMinValue(activeNode, activeAtt);
						bestScoreTrain = scoreTrain;
					}
					
					prevMin = currMin;
				}
				
				// Set the active attribute to the best min value
				double minDelta = bestMin - model.getMinValue(activeNode, activeAtt);
				///* Ablative Debug Start
				if (true) {
					// This code snippet is time-consuming
					model.setMinValue(activeNode, activeAtt, minDelta);
					scoreTrain = getScore(model, rankLists, scorer);
				} else {
					scoreTrain = getScore(
							model, rankLists, scorer, activeNode, activeAtt, minDelta, "min");
				}
				if (Math.abs(scoreTrain - bestScoreTrain) > tolerance) {
					System.err.printf("%s: estimated %f v.s. real %f if setting min to %f\n",
							scorer.name(), bestScoreTrain, scoreTrain, bestMin);
					System.exit(1);
				}
				//*/ Ablative Debug End
				///*
				if (verbose) {
					System.out.printf("  min:%07d score:%.16f (best)\n", (int) bestMin, scoreTrain);
				}
				//*/
				
				double prevMax = bestMax;
				for (int j = 0; j < indexNum; j ++) {
					int curIndex = maxIndex - j * indexUnit;
					double currMax = model.attIdToValList.get(attId).get(curIndex);
					if (currMax <= bestMin) {
						break;
					}
					double maxDelta = currMax - prevMax;
					
					///* Ablative Debug Start
					if (correct) {
						// This code snippet is time-consuming
						model.setMaxValue(activeNode, activeAtt, maxDelta);
						scoreTrain = getScore(model, rankLists, scorer);
					} else {
						scoreTrain = getScore(
								model, rankLists, scorer, activeNode, activeAtt, maxDelta, "max");
					}
					//*/ Ablative Debug End
					if (verbose) {
						System.out.printf("      #%02d %07d %07d=%07d %.16f (max)\n", j, curIndex, 
								(int) currMax, (int) model.getMaxValue(activeNode, activeAtt), scoreTrain);
					}
					//*/
					
					if (scoreTrain > bestScoreTrain) {
						bestMax = model.getMaxValue(activeNode, activeAtt);
						bestScoreTrain = scoreTrain;
					}
					
					prevMax = currMax;
				}
				
				// Set the active attribute to the best max value
				double maxDelta = bestMax - model.getMaxValue(activeNode, activeAtt);
				///* Ablative Debug Start
				if (true) {
					// This code snippet is time-consuming
					model.setMaxValue(activeNode, activeAtt, maxDelta);
					scoreTrain = getScore(model, rankLists, scorer);
				} else {
					scoreTrain = getScore(
							model, rankLists, scorer, activeNode, activeAtt, maxDelta, "max");
				}
				if (Math.abs(scoreTrain - bestScoreTrain) > 5 * Math.pow(10, -3)) {
					System.err.printf("%s: estimated %f v.s. real %f if setting min to %f\n",
							scorer.name(), bestScoreTrain, scoreTrain, bestMax);
					System.exit(1);
				}
				//*/ Ablative Debug End
				///*
				if (verbose) {
					System.out.printf("  max:%07d score:%.16f (best)\n", (int) bestMax, scoreTrain);
				}
				//*/
				
				//break;			
			}// for (int i = 0; i < idPairs.size(); i ++)

			// Save model parameters at each iteration because training takes too long
			model.save(nIter);
			
			double gainTrain = scoreTrain - startScoreTrain;
			timeStamp(String.format("  Increase training %s by %f (from %f to %f)", 
					scorer.name(), gainTrain, startScoreTrain, scoreTrain));
			nIter += 1;
			if (gainTrain < minGainBound) {
				break;
			}
			
			//break;
		} // while (true)
		model.save(-1);
	}
	
	protected static double getScore(
			FirTree model, 
			Map<String, RankList> rankLists, 
			MetricScorer scorer,
			int activeNode,
			int activeAtt,
			double delta,
			String type
			) {
		double total = 0;
		double weight = 0;
		if (type.equals("min")) {
			model.setMinValue(activeNode, activeAtt, delta);
		} else if (type.equals("max")) {
			model.setMaxValue(activeNode, activeAtt, delta);
		} else {
			System.err.printf("Unknown type %s in CoorAscentOnLeaves.getScore\n", type);
			System.exit(1);
		}
		for (RankList rankList : rankLists.values()) {
			if (isActive(rankList, activeNode)) {
				// There is one of the rank list's instances that falls in the active node
				for (Instance instance : rankList.getInstances()) {
					///* Ablative Debug Start
					model.predict(instance, activeNode, activeAtt, delta, type);
					//*/
					/*// This code snippet is time-consuming
					model.predict(instance);
					*/// Ablative Debug End
				}
				// Set is easily forgot
				rankList.setScore(scorer.score(rankList));
			}
			double score = rankList.getScore();
			if (! Double.isNaN(score)) {
				// If tp_fn and fp_tn are never 0 when computing AUC
				total += score;
				weight += rankList.getWeight();
			}
		}
		return total / weight;
	}
	
	// The hyper-parameters of training model parameters by coordinate ascent
	// delta = [ deltaUnit * deltaBase^0, ..., deltaUnit * deltaBase^deltaMaxPower ]
	public static double deltaUnit = 0.001;
	public static double deltaRatio = 0.01;
	
	// (1 + 0.01 * (pow(2.0, 10 + 1) - 1)) = 21.5
	/*//
	public static double deltaBase = 2.0;
	public static double deltaMaxPower = 10; // A smaller value speeds up training
	public static double minGainTrain = 0.0001; // A larger value speeds up training
	*///
	// (1 + 0.01 * (pow(1.1, 80 + 1) - 1)) = 23.5
	public static double deltaBase = 1.1;
	public static double deltaMaxPower = 80; // A smaller value speeds up training
	public static double minGainTrain = 0.00001; // A larger value speeds up training
	
	protected static void tuneParams(
			Options opts,
			FirTree model, 
			Map<String, RankList> rankLists,
			MetricScorer scorer
			) throws Exception {
		timeStamp("Start to tune parameters of linear models");
		
		// Create log directory and delete all previous log files
		String dir = Paths.get(opts.logPath).getParent().toString();
		String logPath = dir + "/CA_" + opts.modelPrefix + "_y" + opts.polyDegree + "_PLOTS";
		File logDir = new File(logPath);
		if (! logDir.exists()) {
			logDir.mkdirs();
		} else {
			for (File logFile : logDir.listFiles()) {
				logFile.delete();
			}
		}
		
		int nIter = 0;
		while (true) {
			// Reset cached predictions helps prevent numerical issues 
			double scoreTrain = getScore(model, rankLists, scorer);
			System.out.printf("Training %s is %f at the start of iteration %d\n", 
					scorer.name(), scoreTrain, nIter);
			//System.exit(0);

			double startScoreTrain = scoreTrain;

			// Reset cached predictions is disabled
			List<IntPair> idPairs = model.getParamIdPairs();
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

			// Save model parameters at each iteration because training takes too long
			model.save(nIter);

			double gainTrain = scoreTrain - startScoreTrain;
			System.out.printf("\tIncrease training %s by %f (from %f to %f)\n", 
					scorer.name(), gainTrain, startScoreTrain, scoreTrain);
			nIter += 1;
			if (gainTrain < minGainTrain) {
				break;
			}
			
			//break;			
		} // while (true)
		model.save(-1);
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
		double weight = 0;
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
				// Set is easily forgot
				rankList.setScore(scorer.score(rankList));
			}
			double score = rankList.getScore();
			if (! Double.isNaN(score)) {
				// If tp_fn and fp_tn are never 0 when computing AUC
				total += score;
				weight += rankList.getWeight();
			}
		}
		return total / weight;
	}

	protected static double getScore(
			FirTree model, 
			Map<String, RankList> rankLists, 
			MetricScorer scorer
			) {
		double total = 0;
		double weight = 0;
		for (RankList rankList : rankLists.values()) {
			for (Instance instance : rankList.getInstances()) {
				model.predict(instance);
			}
			// Set is easily forgot
			rankList.setScore(scorer.score(rankList));
			double score = rankList.getScore();
			if (! Double.isNaN(score)) {
				// If tp_fn and fp_tn are never 0 when computing AUC
				total += score;
				weight += rankList.getWeight();
			}
		}
		return total / weight;
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
		timeStamp("Scan data into rank lists");
		
		String dir = Paths.get(opts.logPath).getParent().toString();
		Map<String, RankList> rankLists = new HashMap<>();
		List<String> allLeaves = model.getAllLeaves();
		for (String leafName : allLeaves) {
			String dataPath = Paths.get(dir, "Node_" + leafName, "fir.dta").toString();
			Map<String, Integer> nameToId = new HashMap<>();
			int leafIndex = model.nodeIndexes.get(leafName);
			List<Integer> attIdList = model.nodeAttIdList.get(leafIndex);
			for (int i = 0; i < attIdList.size(); i ++) {
				nameToId.put(ainfo.idToName(attIdList.get(i)), i);
			}
			BufferedReader br = new BufferedReader(new FileReader(dataPath));
			for (String line = br.readLine(); line != null; line = br.readLine()) {
				String[] data = line.split("\t");
				Instance instance = new Instance(
						InstancesReader.parseDenseInstance(data, ainfo, attIdList, false),
						nameToId
						);
				String groupId = data[ainfo.nameToCol.get(opts.group)];
				instance.setGroupId(groupId);
				
				// Predict index of the node (must be a leaf) the instance falls in
				int nodeIndex = model.indexLeaf(data); // model.indexLeaf(instance, data);
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
				
				model.addAttIdToValSet(attIdList, data);
			}
			br.close();
		}
		
		// Set the weight of a rank list
		for (RankList rankList : rankLists.values())
			rankList.setWeight();
		
		model.addAttIdToValList();
		
		return rankLists;
	}
	
	static String getNodeDir(String dir, String node) {
		return Paths.get(dir, "Node_" + node).toString();
	}
	
	static List<Path> getDataPaths(String dir, String node) {
		List<Path> dataPaths = new ArrayList<Path>();
		String left = getNodeDir(dir, node) + "_L";
		String right = getNodeDir(dir, node) + "_R";
		if (new File(left).exists()) {
			if (new File(right).exists()) {
				dataPaths.addAll(getDataPaths(dir, node + "_L"));
				dataPaths.addAll(getDataPaths(dir, node + "_R"));
			} else {
				System.err.printf("%s does not exist\n", right);
				System.exit(1);
			}
		} else {
			if (new File(right).exists()) {
				System.err.printf("%s does not exist\n", left);
				System.exit(1);
			} else {
				Path dataPath = Paths.get(getNodeDir(dir, node), "fir.dta");
				if (! new File(dataPath.toString()).exists()) {
					System.err.printf("%s does not exist\n", dataPath);
					System.exit(1);
				}
				dataPaths.add(dataPath);
			}
		}
		return dataPaths;
	}
	
	static void timeStamp(String msg) {
		Date date = new Date();
		System.out.println("TIMESTAMP >>>> ".concat(date.toString()).concat(": ").concat(msg));
	}
}
