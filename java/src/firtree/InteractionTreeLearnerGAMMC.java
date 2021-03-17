package firtree;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import firtree.metric.GAUCScorer;
import firtree.metric.MetricScorer;
import firtree.metric.NDCGScorer;
import firtree.utilities.RankList;
import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.Pointer;
import mltk.core.Pointers;
import mltk.core.io.*;
import mltk.predictor.evaluation.AUC;
import mltk.predictor.evaluation.Metric;
import mltk.predictor.evaluation.RMSE;
import mltk.predictor.gam.GAMLearner;
import mltk.predictor.gam.GAM;
import mltk.util.Queue;
import mltk.util.Random;
import mltk.util.tuple.*;

/**
 * Build feature interaction and regression tree (FirTree).
 *
 * @author Daria Sorokina, modified by Xiaojie Wang
 *
 */
public class InteractionTreeLearnerGAMMC {
	
	static class NodeCreationThread extends Thread {
		
		Options opts;
		InteractionTreeLearnerGAMMC app;
		int data_size_min;
		String prefix;
		int data_size; 
		int zero_size;
		boolean tree_size_limit_reached;
		int nThread;
		InteractionTreeNode node;
		
		NodeCreationThread(
				Options opts, 
				InteractionTreeLearnerGAMMC app, 
				int data_size, 
				int zero_size, 
				String prefix, 
				boolean tree_size_limit_reached,
				int nThread
				) {
			this.opts = opts;
			this.app = app;
			this.data_size = data_size;
			this.zero_size = zero_size;
			this.prefix = prefix;
			this.tree_size_limit_reached = tree_size_limit_reached;
			this.nThread = nThread;
			node = null;
		}
		
		public void run() {
			try {
				node = app.createNode(data_size, zero_size, prefix, tree_size_limit_reached, nThread);
			} catch (Exception e) {
				e.printStackTrace();
				try {
					PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(opts.dir + "/creation.err", true)));
					pw.printf("Variable prefix %s\n", prefix);
					pw.println(e.toString());
					pw.println();
					pw.flush();
					pw.close();
				} catch (IOException uncatched) {}
			}
		}
	}

	/**
	 * Script locations.
	 */
	private static String AG = "";
	private static String BT = "";
	private static String VIS_EFFECT = "";
	private static String VIS_IPLOT = "";
	private static String RND = "";
	private static String tempDir = "";
	private static String VIS_MV = "";
	private static String VIS_SPLIT = "";
	
	static class Options {
		@Argument(name = "-p", description = "env.config directory", required = true)
		String prefix = null;
		
		@Argument(name = "-d", description = "FirTree directory", required = true)
		String dir = "";
		
		@Argument(name = "-r", description = "attribute file", required = true)
		String attPath = "";
		
		@Argument(name = "-t", description = "training set", required = true)
		String trainPath = "";
		
		@Argument(name = "-g", description = "name of the attribute with the group id (default: \"\")")
		String group = "None";
		
		@Argument(name = "-c", description = "(rms|roc) - metric to train GAMs in splits (default: roc)")
		String metricStr = "roc";
		
		@Argument(name = "-e", description = "(gauc|ndcg) - metric to evaluate splits (default: gauc)")
		String metricEval = "gauc";
		
		@Argument(name = "-h", description = "max tree height(default: 6)")
		int maxHeight = 6;

		@Argument(name = "-n", description = "number of parallel split evaluations (default: #cores)")
		int nSplitEvaluation = Runtime.getRuntime().availableProcessors();
		
		@Argument(name = "-l", description = "min leaf size (default: 70)")
		int leafSize = 70;

		@Argument(name = "-m", description = "max number of leaves(default: 11)")
		int maxLeaves = 11;

		@Argument(name = "-s", description = "(0|1) subsampling method wrt groups: 0 - sample groups, 1 - sample within each group (default: 0)")
		int group_method = 0;
	}
	
	private Options opts;
	private AttrInfo ainfo;
	private int group_col;
	private Boolean regression;
	
	static int maxNumItersGAM = 100;
	static int maxNumLeavesGAM = 3;

	public InteractionTreeLearnerGAMMC(Options opts) throws IOException {
		this.opts = opts;
		ainfo = AttributesReader.read(opts.attPath);
		group_col = 0;
		if(!opts.group.equals("None")) {
			if(ainfo.nameToCol.containsKey(opts.group)) {
				group_col = ainfo.nameToCol.get(opts.group) + 1;
				ainfo.groupCol = ainfo.nameToCol.get(opts.group);
			} else {
				System.err.println("Error: the feature with the name " + opts.group + " does not exist.");
				System.exit(1);
			}
		}
		regression = opts.metricStr.equals("rms");
	}
	
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(InteractionTreeLearnerGAMMC.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		Random.getInstance().setSeed(0);

		File binDir = new File(opts.prefix);
		if (!binDir.exists()) {
			System.err.println("Error: the config directory " + opts.prefix + " does not exist.");
			System.exit(1);
		}
		File cfgFile = new File(opts.prefix + "/env.config");
		if (!cfgFile.exists()) {
			System.err.println("Error: the config file " + opts.prefix + "/env.config does not exist. Wrong config directory?");
			System.exit(1);
		}
		opts.prefix = binDir.getAbsolutePath();
			
		File dir = new File(opts.dir);
		if (!dir.exists()) {
			System.err.println("Error: the FirTree directory " + opts.dir + " does not exist.");
			System.exit(1);
		}
		opts.dir = dir.getAbsolutePath();

		File attrFile = new File(opts.attPath);
		if (!attrFile.exists()) {
			System.err.println("Error: the attribute file " + opts.attPath + " does not exist.");
			System.exit(1);
		}
		opts.attPath = attrFile.getAbsolutePath();

		FileWriter logFile = new FileWriter(opts.dir + "/treelog.txt", false);
		logFile.close();
				
		opts.prefix += File.separator;
		AG = opts.prefix + "fast_interactions.sh";
		BT = opts.prefix + "bt.sh";
		VIS_EFFECT = opts.prefix + "vis_effect.sh";
		VIS_IPLOT = opts.prefix + "vis_iplot.sh";
		RND = opts.prefix + "rnd.sh";
		VIS_MV = opts.prefix + "vis_mv.sh";
		VIS_SPLIT = opts.prefix + "vis_split.sh";
		tempDir = opts.dir + File.separator + "Node";
	
		File root = new File(tempDir + "_Root");
		if (!root.exists()) {
			root.mkdir();
		}
 		FileSystem fs = FileSystems.getDefault();
 		String attrDest = tempDir + "_Root" + File.separator + "fir.attr";
 		String dataDest = tempDir + "_Root" + File.separator + "fir.dta";		
 
		Files.copy(fs.getPath(opts.attPath), fs.getPath(attrDest), StandardCopyOption.REPLACE_EXISTING);
		BufferedReader br = new BufferedReader(new FileReader(opts.trainPath), 65535);
		BufferedWriter data_out = new BufferedWriter(new FileWriter(dataDest));

		long start = System.currentTimeMillis();
		InteractionTreeLearnerGAMMC app = new InteractionTreeLearnerGAMMC(opts);
		
		timeStamp("Initial scan of the data.");

		int data_size = 0;
		int zero_size = 0;
		int clsColNo = app.ainfo.getClsCol();

 		for (String line = br.readLine(); line != null; data_size++) {
 			String[] datapoint = line.split("\t+");
 			if(datapoint.length != app.ainfo.getColN()) {
					System.err.println("Error: The number of values in line " + (data_size + 1) + " does not match the number of attributes specified by the attribute file.");
					System.exit(1);  				
 			}
 			try {
 				double clsValue = Double.parseDouble(datapoint[clsColNo]);
 				if(clsValue == 0)
 					zero_size++;
 				if(!app.regression && ((clsValue < 0) || (clsValue > 1))) {
 					System.err.println("Error: The response column contains value \"" + datapoint[clsColNo] + "\" in line " + (data_size + 1) + ". Not compatible with the AUC metric.");
 					System.exit(1); 					
 				}
			} catch(java.lang.NumberFormatException e) {
				System.err.println("Error: The response column contains a text value \"" + datapoint[clsColNo] + "\" in line " + (data_size + 1));
				System.exit(1);
			}	
 			data_out.write(line + "\n");
			line = br.readLine();
		}
		br.close();			
		data_out.flush();
		data_out.close();

		app.build(data_size, zero_size);
		long end = System.currentTimeMillis();
		
		System.out.println("Finished building tree in " + (end - start) / 1000.0 + " (s).");
	}
	
	public void printLog(StringBuilder text) throws Exception{
		System.out.println(text);
		PrintWriter log = new PrintWriter(new BufferedWriter(new FileWriter(opts.dir + "/treelog.txt", true)));
		log.println(text);
		log.flush();
		log.close();
	}

	public void build(int data_size, int zero_size) throws Exception {
		int nProcessor = Runtime.getRuntime().availableProcessors();
		Map<InteractionTreeNode, String> prefix = new HashMap<>();
		int leafN = 1;
		InteractionTreeNode root = createNode(
					data_size, 
					zero_size, 
					"Root", 
					leafN >= opts.maxLeaves,
					nProcessor
					);
		prefix.put(root, "Root");
		Queue<InteractionTreeNode> q = new Queue<>(); //queue of internal nodes
		if(!root.isLeaf()) {			
			q.enqueue(root);
			leafN++;
		}
		
		while (!q.isEmpty()) {
			InteractionTreeNode node = q.dequeue();
			
			String pre = prefix.get(node);
			InteractionTreeInteriorNode interiorNode = 
					(InteractionTreeInteriorNode) node;
			String dirStr_cur = tempDir + "_" + pre;
			String dirStr_L = tempDir + "_" + pre + "_L";
			String dirStr_R = tempDir + "_" + pre + "_R";
			
			File left_dir = new File(dirStr_L);
			if (!left_dir.exists()) {
				left_dir.mkdir();
			}
			File right_dir = new File(dirStr_R);
			if (!right_dir.exists()) {
				right_dir.mkdir();
			}
	 		String attrSrc = tempDir + "_Root" + File.separator + "fir.attr";
	 		String attrDestL = dirStr_L + File.separator + "fir.attr";
	 		String attrDestR = dirStr_R + File.separator + "fir.attr";
	 		FileSystem fs = FileSystems.getDefault();
			Files.copy(fs.getPath(attrSrc), fs.getPath(attrDestL), StandardCopyOption.REPLACE_EXISTING);				
			Files.copy(fs.getPath(attrSrc), fs.getPath(attrDestR), StandardCopyOption.REPLACE_EXISTING);				
			
			String fileName = File.separator + "fir.dta";
			String dataStr = dirStr_cur + fileName;
			String dataStrL = dirStr_L + fileName;
			String dataStrR = dirStr_R + fileName;
			
			DataSizes sizes = split(ainfo.attributes.get(interiorNode.attIndex).getColumn(), 
					interiorNode.splitPoint, dataStr, dataStrL, dataStrR);
	
			NodeCreationThread lThread = new NodeCreationThread(
					opts, 
					this, 
					sizes.left_size, 
					sizes.left_zero_size, 
					pre + "_L", 
					leafN >= opts.maxLeaves,
					nProcessor / 2
					);
			NodeCreationThread rThread = new NodeCreationThread(
					opts, 
					this, 
					sizes.right_size, 
					sizes.right_zero_size, 
					pre + "_R", 
					leafN >= opts.maxLeaves,
					nProcessor / 2
					);
			lThread.start();
			rThread.start();
			lThread.join();
			rThread.join();
			
			prefix.put(lThread.node, pre + "_L");
			prefix.put(rThread.node, pre + "_R");
			
			if (!lThread.node.isLeaf()) {
				q.enqueue(lThread.node);
				leafN++;
			}
			if (!rThread.node.isLeaf()) {
				q.enqueue(rThread.node);
				leafN++;
			}
		}			
	}
	
	protected static void split(
			Instances instances, 
			int attIndex, 
			double splitPoint, 
			Pointers left, 
			Pointers right
			) {
		for (int i = 0; i < instances.size(); i ++) {
			Instance instance = instances.get(i);
			if ((instance.getValue(attIndex) <= splitPoint) || 
					Double.isNaN(splitPoint) && Double.isNaN(instance.getValue(attIndex))) {
				left.add(new Pointer(i));
			} else {
				right.add(new Pointer(i));
			}
		}
	}
	
	protected static void split(Instances instances, int attIndex, double splitPoint, 
			Instances left, Instances right) {
		for (Instance instance : instances) {
			if ((instance.getValue(attIndex) <= splitPoint) || 
					Double.isNaN(splitPoint) && Double.isNaN(instance.getValue(attIndex))) {
				left.add(instance);
			} else {
				right.add(instance);
			}
		}
	}
	
	protected DataSizes split(int splitCol, double splitPoint, 
			String dataFile, String outLeftFile, String outRightFile) throws IOException {
		
		BufferedReader br = new BufferedReader(new FileReader(dataFile), 65535);
		BufferedWriter outLeft = new BufferedWriter(new FileWriter(outLeftFile));
		BufferedWriter outRight = new BufferedWriter(new FileWriter(outRightFile));
		DataSizes sizes = new DataSizes(0,0,0,0);
		int clsCol = ainfo.getClsCol();
			
		String line = br.readLine();
		while (line != null) {
			String[] data = line.split("\t+");
			line += "\n";
			double value = data[splitCol].equals("?") ?
							Double.NaN :
							Double.parseDouble(data[splitCol]);
			double clsValue = Double.parseDouble(data[clsCol]);
			if ((value <= splitPoint) || Double.isNaN(splitPoint) && Double.isNaN(value)) {
				outLeft.write(line);
				sizes.left_size++;
				if(clsValue == 0)
					sizes.left_zero_size++;
			} else {
				outRight.write(line);
				sizes.right_size++;
				if(clsValue == 0)
					sizes.right_zero_size++;
			}
			line = br.readLine();
		}

		br.close();			
		outLeft.flush();
		outLeft.close();
		outRight.flush();
		outRight.close();
		
		//delete the data from the internal node
		File data = new File(dataFile);
		data.delete();
		
		return sizes;
	}
	
	protected static void timeStamp(String msg){
		Date tmpDate = new Date();
		System.out.println("TIMESTAMP >>>> ".concat(tmpDate.toString()).concat(": ").concat(msg));
	}

	private void subsample(int data_size, int zero_size, double train_coef, int train_abs, int valid_abs, File dir, String dtaAG, int tar_col, int seed) throws Exception	{
		
		if (zero_size <= data_size / 2)
		{
			int train_size =  Math.min((int)(data_size * train_coef), train_abs);
			int valid_size =  Math.min(data_size - train_size, valid_abs);
			
			double portion_train = (double) train_size / data_size;
			double portion_valid = (double) valid_size /  data_size;
			runProcess(dir, RND, "--input " + dtaAG + " --stem fir --group-method " + opts.group_method + " --group " + group_col + 
							" --valid " + portion_valid + " --train " + portion_train + " --rand " + seed);
		} else 
		{			
			int nonzero_size = data_size - zero_size;			
			int nonzero_train_size = (int)Math.min(nonzero_size * train_coef, train_abs / 2.0);
			int zero_train_size = (int)Math.min(zero_size * train_coef, train_abs - nonzero_train_size);
			int nonzero_valid_size = (int)Math.min(nonzero_size - nonzero_train_size, valid_abs / 2.0);
			int zero_valid_size = (int)Math.min(zero_size - zero_train_size, Math.min(valid_abs - nonzero_valid_size, (int)(nonzero_valid_size * ((double)zero_train_size / nonzero_train_size))));			
			
			double portion_zero_train = zero_train_size / (double) zero_size;
			double portion_nonzero_train = nonzero_train_size / (double) nonzero_size;
			double portion_zero_valid = zero_valid_size / (double) zero_size;
			double portion_nonzero_valid = nonzero_valid_size / (double) nonzero_size;
			
			runProcess(dir, RND, "--input " + dtaAG + " --stem fir --group-method " + opts.group_method + " --group " + group_col + " --target " + tar_col +
					" --valid " + portion_nonzero_valid + " --train " + portion_nonzero_train + 
					" --valid-zero " + portion_zero_valid + " --train-zero " + portion_zero_train + " --rand " + seed);		
		}	
	}
	
	protected InteractionTreeNode createNode(
				int data_size, 
				int zero_size, 
				String prefix, 
				boolean tree_size_limit_reached,
				int nThread
				)
			throws Exception {
		StringBuilder sb = new StringBuilder();
		sb.append(prefix + "\n");
		if (data_size < opts.leafSize) {
			sb.append("Constant leaf. Not enough data.\n");
			printLog(sb);		
			return new InteractionTreeLeaf();
		}

		String tmpDir = tempDir + "_" + prefix;
		File dir = new File(tmpDir);
		String attr = tmpDir + File.separator + "fir.attr";
		String attrfs12 = tmpDir + File.separator + "fir.fs.attr";
		String attrfsfs = tmpDir + File.separator + "fir.fs.fs.attr";
		String dtaAG = tmpDir + File.separator + "fir.dta";
		String trainAG = tmpDir + File.separator + "fir.train.ag";
		String validAG = tmpDir + File.separator + "fir.valid.ag";
		String train = tmpDir + File.separator + "fir.train.dta";
		String valid = tmpDir + File.separator + "fir.valid.dta";
		String core = tmpDir + File.separator + "core_features.txt";
		int tar_col = ainfo.clsAttr.getColumn() + 1;		

 		FileSystem fs = FileSystems.getDefault();

		// 1. Create datasets
		// 1.1. Create datasets for ag
 		
 		timeStamp("Prepare train and test data for AG.");
		subsample(data_size, zero_size, 1.0/3.0, 30000, 500000, dir, dtaAG, tar_col, 1); 
		
		Files.move(fs.getPath(train), fs.getPath(trainAG), StandardCopyOption.REPLACE_EXISTING); 
		Files.move(fs.getPath(valid), fs.getPath(validAG), StandardCopyOption.REPLACE_EXISTING); 

		Instances agTrain = InstancesReader.read(ainfo, trainAG, "\t+", true);
		boolean allSame = true;
		for (int i = 1; i < agTrain.size(); i++) {
			Instance instance = agTrain.get(i);
			if (instance.getTarget() != agTrain.get(0).getTarget()) {
				allSame = false;
					break;
			}
		}
		if (allSame) {
			sb.append("Constant leaf. All data points have the same label.\n");
			printLog(sb);
			return new InteractionTreeLeaf();
		}

		// 1.2 Create datasets for bt and gam 
		timeStamp("Prepare train and test data for BT and GAM.");
		subsample(data_size, zero_size, 2.0/3.0, 200000, 500000, dir, dtaAG, tar_col, 2); 
		
		// 2. Fast feature selection
		timeStamp("Select 12 features and add up to 4 split features for AG.");
		runProcess(dir, BT, attr, train, valid, String.format("-k 12 -b 300 -a 0.01 -h %d -s 4", nThread)); 
 
		// XW. Remove bulky BTTemp after running bt_train
		File btDir = new File(tmpDir + File.separator + "BTTemp");
		for (File entry : btDir.listFiles()) {
			entry.delete();
		}
		
		Path fsLogSrc = fs.getPath(tmpDir + File.separator + "log.txt");
		Path fsLogDst = fs.getPath(tmpDir + File.separator + "log_fs.txt");
		Path fsModelSrc = fs.getPath(tmpDir + File.separator + "model.bin");
		Path fsModelDst = fs.getPath(tmpDir + File.separator + "model_fs.bin");
		Files.copy(fsLogSrc, fsLogDst, StandardCopyOption.REPLACE_EXISTING); 
		Files.copy(fsModelSrc, fsModelDst, StandardCopyOption.REPLACE_EXISTING); 
		//clean up large unneeded preds.txt file
		File predsFile = new File(tmpDir + File.separator + "preds.txt");
		predsFile.delete();


		// 3. Run ag and get interactions
		timeStamp("Run AG with selected features on the small train set.");
		runProcess(dir, AG, attrfs12, attrfsfs, trainAG, validAG, String.format("-h %d", nThread)); 
		// Backup log and model
		Path agLogSrc = fs.getPath(tmpDir + File.separator + "log.txt");
		Path agLogDst = fs.getPath(tmpDir + File.separator + "log_ag.txt");
		Path agModelSrc = fs.getPath(tmpDir + File.separator + "model.bin");
		Path agModelDst = fs.getPath(tmpDir + File.separator + "model_ag.bin");
		Files.copy(agLogSrc, agLogDst, StandardCopyOption.REPLACE_EXISTING); 
		Files.copy(agModelSrc, agModelDst, StandardCopyOption.REPLACE_EXISTING); 
		
		//add first column from core_features.txt to treelog.txt		
		sb.append("Core features:\n");
		BufferedReader br = new BufferedReader(new FileReader(core), 65535);
		for (;;) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			String[] data = line.split("\t");
			sb.append("\t" + data[0] + "\n");
		}
		br.close();
		
		//done with AG, so remove bulky AGTemp
		File agDir = new File(tmpDir + File.separator + "AGTemp");
		for (File entry : agDir.listFiles()) {
			entry.delete();
		}

		//3a. If number of leaves or height limit reached, stop here.
		if (tree_size_limit_reached)
		{
			visAllEffectPlots(
					tmpDir, 
					dir, 
					attrfs12, 
					train, 
					valid, 
					sb, 
					"Regression leaf. Number of leaves limit reached.",
					nThread
					);
			return new InteractionTreeLeaf();			
		}
		if (prefix.length() > opts.maxHeight * 2)
		{
			visAllEffectPlots(
					tmpDir, 
					dir, 
					attrfs12, 
					train, 
					valid, 
					sb, 
					"Regression leaf. Branch height limit reached.",
					nThread
					);
			return new InteractionTreeLeaf();			
		}
		
		// 4. Choose candidates
		String candidates = tmpDir + File.separator + "candidates.txt";		
		Pair<List<String>, String> gcret = getCandidates(ainfo, candidates);
		List<String> candidateFeatureNames = gcret.v1;
		String candType = gcret.v2; 
		
		// 5. Prepare train set and valid set for GAM
 		Instances trainSet = InstancesReader.read(ainfo, train, "\t+", true);
		Instances validSet = InstancesReader.read(ainfo, valid, "\t+", true);

		GAMLearner learner = new GAMLearner();
		Metric metric;
		if(regression == true)
			metric = new RMSE();
		else
			metric = new AUC();
		
		// We use RMSE or AUC when scorer is not GAUC or NDCG
		MetricScorer scorer = null;
		if (opts.metricEval.equals("gauc"))
			scorer = new GAUCScorer();
		if (opts.metricEval.equals("ndcg")) {
			int k = 4;
			if (opts.metricEval.contains("@")) {
				k = Integer.parseInt(opts.metricEval.split("@")[1]);
			}
			scorer = new NDCGScorer(k);
		}
		
		learner.setMetric(metric);
		learner.setLearningRate(0.01);
		learner.setBaggingIters(0);

		int attrN = ainfo.attributes.size();
		for(Instance instance : trainSet)
			for(int a = 0; a < attrN; a++)
				if(Double.isNaN(instance.getValue(a)))
					instance.setValue(a, 0);
		for(Instance instance : validSet)
			for(int a = 0; a < attrN; a++)
				if(Double.isNaN(instance.getValue(a)))
					instance.setValue(a, 0);	

		//6. Plots
		timeStamp("Visualization.");
		runProcess(dir, VIS_MV, "AG_PLOTS");

		String interactionGraph = tmpDir + File.separator + "list.txt";
		List<Pair<String, String>> pairs = getInteractions(interactionGraph);
		visIPlot(dir, attrfs12, train, valid, tmpDir, pairs, "v1", nThread);
		visIPlot(dir, attrfs12, valid, train, tmpDir, pairs, "v2", nThread);

		// 7. Read features info: get quantiles from the effect plots
		List<Feature> candidateFeatures = new ArrayList<>();
		for (String key : candidateFeatureNames) {
			Feature feature = Feature.read(tmpDir + File.separator + "AG_PLOTS" + File.separator + key + ".effect.txt");
			candidateFeatures.add(feature);
		}
		
		// 8.1 Build a parent GAM and child GAMs in parallel
		timeStamp("Evaluate splits.");
		int bestAtt = -1;
		double bestSplit = -1;
		double bestScore = metric.worstValue();
		
		List<Future<GAMLearningResult>> results = new ArrayList<>();
		ExecutorService executor = Executors.newFixedThreadPool(opts.nSplitEvaluation);
		// Parent GAM
		timeStamp("Training parent is added to thread pool");
		{
			GAMLearningTask task = new GAMLearningTask(
					this, 
					tmpDir, 
					trainSet, 
					validSet, 
					learner, 
					metric,
					scorer
					);
			Future<GAMLearningResult> result = executor.submit(task);
			results.add(result);
		}
		// Child GAMs
		for (int i = 0; i < candidateFeatures.size(); i++) {
			String featureName = candidateFeatureNames.get(i);
			int attIndex = ainfo.nameToId.get(featureName);
			FeatureSplit split = new FeatureSplit(candidateFeatures.get(i));		
			for (int j = 0; j < split.splits.length; j++) {
				double splitPoint = (split.feature.centers[j] + split.feature.centers[j + 1]) / 2;
				timeStamp("Evaluating feature " + featureName + " split " + splitPoint + " is added to thread pool");
				GAMLearningTask task = new GAMLearningTask(
						this, 
						tmpDir, 
						trainSet, 
						validSet, 
						learner, 
						metric, 
						scorer,
						attIndex, 
						featureName, 
						split, 
						splitPoint
						);
				Future<GAMLearningResult> future = executor.submit(task);
				results.add(future);
			}
		}
		executor.shutdown();
		while (! executor.isTerminated());

		// 8.2 Evaluate all splits and select the best one
		double parentScore = Double.NaN;
		for (Future<GAMLearningResult> result : results) {
			if (result.get().isParent) {
				parentScore = result.get().parentScore;
			} else {
				timeStamp(String.format("Evaluating feature %s split %f is terminated with score %f", 
						ainfo.idToName(result.get().attIndex), result.get().splitPoint, result.get().splitScore));
				if (metric.isFirstBetter(result.get().splitScore, bestScore)) {
					bestAtt = result.get().attIndex;
					bestSplit = result.get().splitPoint;
					bestScore = result.get().splitScore;
				}
			}
		}
		if (scorer == null) {
			sb.append("Parent " + metric.toString() + ": " + parentScore + "\n");
		} else {
			sb.append("Parent " + scorer.name() + ": " + parentScore + "\n");
		}
		PrintWriter out = new PrintWriter(tmpDir + File.separator + "parent.txt");
		out.println(parentScore);
		out.flush();
		out.close();
		timeStamp("Finished training parent and evaluating all of the splits");

		//9. Final output: best split and its visualization
		if (bestAtt >= 0) {
			if (scorer == null) {
				sb.append("Best " + metric + ": " + bestScore + "\n");
			} else {
				sb.append("Best " + scorer.name() + ": " + bestScore + "\n");
			}
			
			if (metric.isFirstBetter(bestScore, parentScore)) {
				sb.append("Best feature: " + ainfo.attributes.get(bestAtt).getName() + "\n");
				sb.append("Best split: " + bestSplit + "\n");
				if(candType.compareTo("w") == 0)
					sb.append("Weak interactions only.\n");
				if(candType.compareTo("d") == 0)
					visAllEffectPlots(
							tmpDir, 
							dir, 
							attrfs12, 
							train, 
							valid, 
							sb, 
							"Dominant feature split.",
							nThread
							);
				else {
					runProcess(dir, VIS_MV, "BT_PLOTS");
					runProcess(dir, VIS_SPLIT, "BT_PLOTS", ainfo.attributes.get(bestAtt).getName(), bestSplit+"");
					printLog(sb);
				}
				// Visualize the splits
				return new InteractionTreeInteriorNode(bestAtt, bestSplit);

			} else {
				visAllEffectPlots(
						tmpDir, 
						dir, 
						attrfs12, 
						train, 
						valid, 
						sb, 
						"Regression leaf. No improvement found.",
						nThread
						);
				return new InteractionTreeLeaf();
			}
		} else {
			visAllEffectPlots(
					tmpDir, 
					dir, 
					attrfs12, 
					train, 
					valid, 
					sb, 
					"Regression leaf. No interactions found.",
					nThread
					);
			return new InteractionTreeLeaf();
		}		
	}
	
	private void visAllEffectPlots(
			String tmpDir, 
			File dir, 
			String attrfs12, 
			String train, 
			String valid, 
			StringBuilder sb, 
			String NodeLabel,
			int nThread
			) throws Exception{
		String coreFeaturesFileName = tmpDir + File.separator + "core_features.txt";
		Set<String> coreFeatureNames = getCoreFeatures(coreFeaturesFileName);
		visEffect(dir, attrfs12, train, valid, tmpDir, coreFeatureNames, "v1", nThread);
		visEffect(dir, attrfs12, valid, train, tmpDir, coreFeatureNames, "v2", nThread);
		runProcess(dir, VIS_MV, "BT_PLOTS");
		sb.append(NodeLabel + "\n");
		printLog(sb);		
	}
	
	protected double evaluateSplitInPlace(
			String tmpDir, 
			Instances trainSet, 
			Instances validSet,
			GAMLearner learner, 
			Metric metric, 
			MetricScorer scorer,
			int attIndex, 
			FeatureSplit split, 
			double splitPoint
			) throws OutOfMemoryError {
		PrintWriter out;

		// 8.1 Split the dataset
		Pointers trainLeft = new Pointers();
		Pointers trainRight = new Pointers();
		split(trainSet, attIndex, splitPoint, trainLeft, trainRight);

		Pointers validLeft = new Pointers();
		Pointers validRight = new Pointers();
		split(validSet, attIndex, splitPoint, validLeft, validRight);

		//8.2.1 Build GAM models
		GAM gamL;
		if(regression) {
			gamL = learner.buildRegressor(
					trainSet, 
					trainLeft, 
					validSet, 
					validLeft, 
					maxNumItersGAM, 
					maxNumLeavesGAM
					);
		} else {
			gamL = learner.buildClassifier(
					trainSet, 
					trainLeft, 
					validSet, 
					validLeft, 
					maxNumItersGAM, 
					maxNumLeavesGAM
					);
		}

		GAM gamR;
		if(regression) {
			gamR = learner.buildRegressor(
					trainSet, 
					trainRight, 
					validSet, 
					validRight, 
					maxNumItersGAM, 
					maxNumLeavesGAM
					);
		} else {
			gamR = learner.buildClassifier(
					trainSet, 
					trainRight, 
					validSet, 
					validRight, 
					maxNumItersGAM, 
					maxNumLeavesGAM
					);
		}
		
		double splitScore = Double.NaN;
		if (scorer == null) {
			int vNo = 0;
			int actual_valid_size = validLeft.size() + validRight.size();
			double[] targets = new double[actual_valid_size];
			double[] preds = new double[actual_valid_size];
			double[] weights = new double[actual_valid_size];
		
			for (Pointer pointer : validLeft) {
				Instance instance = validSet.get(pointer.getIndex());
				targets[vNo] = instance.getTarget();
				preds[vNo] = gamL.regress(instance);
				weights[vNo] = instance.getWeight();
				vNo++;
			}
			for (Pointer pointer : validRight) {
				Instance instance = validSet.get(pointer.getIndex());
				targets[vNo] = instance.getTarget();
				preds[vNo] = gamR.regress(instance);
				weights[vNo] = instance.getWeight();
				vNo++;
			}
			
			splitScore = metric.eval(preds, targets, weights);
		} else {
			Map<String, RankList> rankLists = new HashMap<String, RankList>();
			
			for (Pointer pointer : validLeft) {
				Instance allIns = validSet.get(pointer.getIndex());
				String groupId = allIns.getGroupId();
				if (! rankLists.containsKey(groupId)) {
					rankLists.put(groupId, new RankList(groupId));
				}
				firtree.utilities.Instance subIns = new firtree.utilities.Instance(allIns.getTarget());
				subIns.setPrediction(gamL.regress(allIns));
				subIns.setWeight(allIns.getWeight());
				rankLists.get(groupId).add(subIns);
			}
			for (Pointer pointer : validRight) {
				Instance allIns = validSet.get(pointer.getIndex());
				String groupId = allIns.getGroupId();
				if (! rankLists.containsKey(groupId)) {
					rankLists.put(groupId, new RankList(groupId));
				}
				firtree.utilities.Instance subIns = new firtree.utilities.Instance(allIns.getTarget());
				subIns.setPrediction(gamR.regress(allIns));
				subIns.setWeight(allIns.getWeight());
				rankLists.get(groupId).add(subIns);
			}
			
			for (RankList rankList : rankLists.values()) {
				rankList.setWeight();
			}
			splitScore = scorer.score(rankLists);
			
			double avgSize = 0.;
			for (RankList rankList : rankLists.values())
				avgSize += rankList.size();
			avgSize /= rankLists.size();
			timeStamp(String.format("Child GAMs have %d lists, each having %.2f points on average", rankLists.size(), avgSize));
		}
		
		try {
			out = new PrintWriter(tmpDir + File.separator + "split_" + split.feature.name + "_" + splitPoint + ".txt");
			out.println(splitScore);
			out.flush();
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		return splitScore;
	}
	
	protected double evaluateSplit(String tmpDir, Instances trainSet, Instances validSet,
			GAMLearner learner, Metric metric, int attIndex, FeatureSplit split, double splitPoint) throws OutOfMemoryError {
		int attrN = ainfo.attributes.size();
		int vNo;
		PrintWriter out;

		// 8.1 Split the dataset
		Instances trainLeft = new Instances(ainfo);
		Instances trainRight = new Instances(ainfo);
		split(trainSet, attIndex, splitPoint, trainLeft, trainRight);

		Instances validLeft = new Instances(ainfo);
		Instances validRight = new Instances(ainfo);
		split(validSet, attIndex, splitPoint, validLeft, validRight);

		// 8.2 Build GAMMC and evaluate this split
		Instances trainLeftGAM = trainLeft.copy();
		Instances validLeftGAM = validLeft.copy();
		Instances trainRightGAM = trainRight.copy();
		Instances validRightGAM = validRight.copy();
		
		//8.2.0 Replace missing values with zeros
		for(Instance instance : trainLeftGAM)
			for(int a = 0; a < attrN; a++)
				if(Double.isNaN(instance.getValue(a)))
					instance.setValue(a, 0);
		for(Instance instance : trainRightGAM)
			for(int a = 0; a < attrN; a++)
				if(Double.isNaN(instance.getValue(a)))
					instance.setValue(a, 0);
		for(Instance instance : validLeftGAM)
			for(int a = 0; a < attrN; a++)
				if(Double.isNaN(instance.getValue(a)))
					instance.setValue(a, 0);
		for(Instance instance : validRightGAM)
			for(int a = 0; a < attrN; a++)
				if(Double.isNaN(instance.getValue(a)))
					instance.setValue(a, 0);

		//8.2.1 Build GAM models
		int actual_valid_size = validLeftGAM.size() + validRightGAM.size();
		double[] targets = new double[actual_valid_size];
		double[] preds = new double[actual_valid_size];
		double[] weights = new double[actual_valid_size];
		vNo = 0;
		GAM gamL;
		if(regression)
			gamL = learner.buildRegressor(trainLeftGAM, validLeftGAM, maxNumItersGAM, maxNumLeavesGAM);
		else
			gamL = learner.buildClassifier(trainLeftGAM, validLeftGAM, maxNumItersGAM, maxNumLeavesGAM);
		for (Instance instance : validLeftGAM) {
			targets[vNo] = instance.getTarget();
			preds[vNo] = gamL.regress(instance);
			weights[vNo] = instance.getWeight();
			vNo++;
		}

		GAM gamR;
		if(regression)
			gamR = learner.buildRegressor(trainRightGAM, validRightGAM, maxNumItersGAM, maxNumLeavesGAM);
		else
			gamR = learner.buildClassifier(trainRightGAM, validRightGAM, maxNumItersGAM, maxNumLeavesGAM);
		for (Instance instance : validRightGAM) {
			targets[vNo] = instance.getTarget();
			preds[vNo] = gamR.regress(instance);
			weights[vNo] = instance.getWeight();
			vNo++;
		}

		double splitScore = metric.eval(preds, targets, weights);

		try {
			out = new PrintWriter(tmpDir + File.separator + "split_" + split.feature.name + "_" + splitPoint + ".txt");
			out.println(splitScore);
			out.flush();
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		return splitScore;
	}

	protected void visIPlot(
				File dir, 
				String attr, 
				String train, 
				String valid, 
				String tmpDir, 
				List<Pair<String, String>> pairs, 
				String suffix,
				int nThread
				) throws Exception{
		// Here we build a large BT model. 
		runProcess(dir, BT, attr, train, valid, String.format("-b 300 -a 0.01 -k 0 -h %d", nThread));

		// Run visualization
		for (Pair<String, String> pair : pairs) {
			runProcess(dir, VIS_IPLOT, attr, train, pair.v1, pair.v2, suffix);
		}		

		//backup model and log files
		FileSystem fs = FileSystems.getDefault();
		Path btLogSrc = fs.getPath(tmpDir + File.separator + "log.txt");
		Path btLogDst = fs.getPath(tmpDir + File.separator + "log_bt." + suffix + ".txt");
		Path btModelSrc = fs.getPath(tmpDir + File.separator + "model.bin");
		Path btModelDst = fs.getPath(tmpDir + File.separator + "model_bt." + suffix + ".bin");
		Files.move(btLogSrc, btLogDst, StandardCopyOption.REPLACE_EXISTING);
		Files.move(btModelSrc, btModelDst, StandardCopyOption.REPLACE_EXISTING);

		//clean up large unneeded preds.txt file
		File predsFile = new File(tmpDir + File.separator + "preds.txt");
		predsFile.delete();
	}
	
	protected void visEffect(
			File dir, 
			String attr, 
			String train, 
			String valid, 
			String tmpDir, 
			Set<String> features, 
			String suffix,
			int nThread
			) throws Exception{
		// Here we build a large BT model
		runProcess(dir, BT, attr, train, valid, String.format("-b 300 -a 0.01 -k 0 -h %d", nThread));
	
		// Run visualization
		for (String feat : features) {
			runProcess(dir, VIS_EFFECT, attr, train, feat, suffix);
		}
		
		//backup model and log files
		FileSystem fs = FileSystems.getDefault();
		Path btLogSrc = fs.getPath(tmpDir + File.separator + "log.txt");
		Path btLogDst = fs.getPath(tmpDir + File.separator + "log_bt." + suffix + ".txt");
		Path btModelSrc = fs.getPath(tmpDir + File.separator + "model.bin");
		Path btModelDst = fs.getPath(tmpDir + File.separator + "model_bt." + suffix + ".bin");
		Files.move(btLogSrc, btLogDst, StandardCopyOption.REPLACE_EXISTING);
		Files.move(btModelSrc, btModelDst, StandardCopyOption.REPLACE_EXISTING);

		//clean up large unneeded preds.txt file
		File predsFile = new File(tmpDir + File.separator + "preds.txt");
		predsFile.delete();
	
	}
	
	protected static void readPredicts(String path, List<Double> preds) throws Exception {
		BufferedReader br = new BufferedReader(new FileReader(path), 65535);
		for (;;) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			preds.add(Double.valueOf(line));
		}
		br.close();
	}
	
	protected static void runProcess(File dir, String ... cmd) throws Exception {
		for (String str: cmd) {
			System.out.print(str + " ");
		}
		System.out.println();
		ProcessBuilder pb = new ProcessBuilder(cmd);
		pb.directory(dir);
		pb.redirectErrorStream(true);
		Process intProcess = pb.start();
		InputStreamReader isr = new InputStreamReader(intProcess.getInputStream());
		BufferedReader br = new BufferedReader(isr);
		
		int exitValue = intProcess.waitFor();
		
		String str = br.readLine();
		while (str != null) {
			System.out.println(str);
			str = br.readLine();
		}
		br.close();
		
		if (exitValue != 0) {
			System.exit(exitValue);
		}
		
	}
	
	protected List<String> readAttrs(String attr) throws Exception {
		List<String> attrs = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(attr), 65535);
		for (;;) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			String[] data = line.split(": ");
			attrs.add(data[0]);
		}
		br.close();
		return attrs;
	}
	

	protected static List<Pair<String, String>> getInteractions(String interactionGraph) 
			throws Exception {
		List<Pair<String, String>> candidates = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(interactionGraph));
		for (;;) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			String[] data = line.split("\\s+");
			if (data[2].indexOf("inf") >= 0 || data[2].indexOf("nan") >= 0 || data[3].indexOf("inf") >= 0 || data[3].indexOf("nan") >= 0) {
				continue;
			}
			double strength = Double.parseDouble(data[2]);
			double scale = Double.parseDouble(data[3]);
			if ((strength >= 3.0) && (scale >= 0.02)) {
				candidates.add(new Pair<String, String>(data[0], data[1]));
			}
		}
		br.close();
		
		return candidates;
	}
	
	protected static Pair<List<String>, String> getCandidates(AttrInfo ainfo, String candFName) 
			throws Exception {
		List<String> candidates = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(candFName));
		String candType = null;
		
		while (true) {
			String line = br.readLine();
			if (line == null)
				break;
			if (candidates.size() >= 3)
				break;
			String[] data = line.split("\\s+");
			
			// If split features are specified, do not consider features that are not split
			if (ainfo.splitNames.size() > 0 && !ainfo.splitNames.contains(data[0])) {
				timeStamp(String.format("Feature %s is not a split feature and excluded", data[0]));
				continue;
			}
			
			if (candType == null)
				candType = data[1];
			
			if (candType.compareTo(data[1]) == 0) {
				if (ainfo.leafNames.contains(data[0])) {
					timeStamp(String.format("Feature %s is a leaf feature and excluded", data[0]));
				} else {
					timeStamp(String.format("Feature %s (%s) is included", data[0], candType));
					candidates.add(data[0]);
				}
			}
			else
				break;
		}
		/*//
		for (int i = 0; i < 3; i++) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			String[] data = line.split("\\s+");
			if(candType == null)
				candType = data[1];
			if(candType.compareTo(data[1]) == 0)
				candidates.add(data[0]);
			else
				break;
		}
		*///
		br.close();
		
		return new Pair<List<String>, String> (candidates, candType);
	}
	
	protected static Set<String> getCoreFeatures(String coreFeaturesFileName) throws Exception{
		
		Set<String> coreFeatures = new HashSet<>();

		BufferedReader br = new BufferedReader(new FileReader(coreFeaturesFileName));
		for (;;) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			String[] data = line.split("\\s+");
			coreFeatures.add(data[0]);		
		}
		br.close();
		
		return coreFeatures;
	}

	public Boolean getRegression() {
		return regression;
	}

	public static int getMaxNumItersGAM() {
		return maxNumItersGAM;
	}

	public static int getMaxNumLeavesGAM() {
		return maxNumLeavesGAM;
	}
	
}
