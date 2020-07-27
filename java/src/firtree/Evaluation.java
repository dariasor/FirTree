package firtree;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

import firtree.metric.GAUCScorer;
import firtree.metric.MetricScorer;
import firtree.metric.NDCGScorer;
import firtree.utilities.Instance;
import firtree.utilities.RankList;
import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.io.AttrInfo;
import mltk.core.io.AttributesReader;

public class Evaluation {
	
	static class Options {
		@Argument(name = "-r", description = "attribute file", required = true)
		String attPath = "";

		@Argument(name = "-test", description = "test set file", required = true)
		String testPath = "";
		
		@Argument(name = "-g", description = "name of the attribute with the group id", required = true)
		String group = "";
		
		@Argument(name = "-o", description = "output file", required = true)
		String outputPath = "";
		
		// This argument comes from InteractionTreeLearnerGAMMC
		@Argument(name = "-c", description = "(gauc|ndcg) - metric to optimize (default: gauc)")
		String metricStr = "gauc";
	}

	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(Evaluation.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		
		AttrInfo ainfo = AttributesReader.read(opts.attPath);
		
		// Load targets and predictions into rank lists of instances
		Map<String, RankList> rankLists = new HashMap<>();
		BufferedReader brT = new BufferedReader(new FileReader(opts.testPath));
		BufferedReader brO = new BufferedReader(new FileReader(opts.outputPath));
		int nLine = 0;
		while (true) {
			String lineT = brT.readLine();
			String lineO = brO.readLine();
			if ((lineT == null) && (lineO != null)) {
				System.err.println("There are more predictions than targets");
				System.exit(1);
			}
			if ((lineT != null) && (lineO == null)) {
				System.err.println("There are more targets than predictions");
				System.exit(1);
			}
			if ((lineT == null) && (lineO == null))
				break;

			String[] data = lineT.split("\t");
			double target = Double.parseDouble(data[ainfo.getClsCol()]);
			Instance instance = new Instance(target);
			String groupId = data[ainfo.nameToCol.get(opts.group)];
			instance.setGroupId(groupId);
			instance.setPrediction(Double.parseDouble(lineO));
			
			if (! rankLists.containsKey(groupId))
				rankLists.put(groupId, new RankList(groupId));
			rankLists.get(groupId).add(instance);
			
			nLine += 1;
			if (nLine % 1000000 == 0) {
				//timeStamp(String.format("Have scanned %d lines", nLine));
			}
		}
		brO.close();
		brT.close();
		
		// Set the weight of a rank list
		for (RankList rankList : rankLists.values())
			rankList.setWeight();

		//test_calculate_ndcg_whenUnweightedData_thenAccurateNDCG(rankLists);
		MetricScorer scorer;
		if (opts.metricStr.equals("gauc"))
			scorer = new GAUCScorer();
		else
			scorer = new NDCGScorer();
		double score = scorer.score(rankLists);
		System.out.printf("Test %s is %.4f given predictions in %s\n", 
				scorer.name(), score, opts.outputPath);
	}
	
	static void test_calculate_ndcg_whenUnweightedData_thenAccurateNDCG(Map<String, RankList> rankLists) {
		NDCGScorer.setGainType("linear");
		NDCGScorer scorer = new NDCGScorer();
		scorer.setK(100);
		Map<String, Double> trueNDCG = new HashMap<String, Double>();
		trueNDCG.put("0", 0.5523531026111766);
		trueNDCG.put("1", 0.5325611891991362);
		trueNDCG.put("2", 0.7206201105813624);
		trueNDCG.put("3", 0.7231357726898465);
		trueNDCG.put("4", 0.3375054808531018);
		trueNDCG.put("5", 0.5523531026111766);
		trueNDCG.put("6", 0.5967798630475649);
		trueNDCG.put("7", 0.895930857984812);
		for (RankList rankList : rankLists.values()) {
			String groupId = rankList.getGroupId();
			double score = scorer.score(rankList);
			if (Math.abs(score - trueNDCG.get(groupId)) > Math.pow(10, -10)) {
				System.err.printf("%s fails due to %.10f (pred) != %.10f (true)\n", 
						groupId, score, trueNDCG.get(groupId));
				//System.exit(1);
			} else 
				System.out.printf("%s succeeds\n", groupId);
		}
	}
	
	static void timeStamp(String msg) {
		Date date = new Date();
		System.out.println("TIMESTAMP >>>> ".concat(date.toString()).concat(": ").concat(msg));
	}
}
