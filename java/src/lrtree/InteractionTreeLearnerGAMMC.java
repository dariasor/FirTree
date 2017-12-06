package lrtree;

import java.util.Date;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.io.AttrInfo;
import mltk.core.io.AttributesReader;
import mltk.core.io.InstancesReader;
import mltk.predictor.evaluation.AUC;
import mltk.predictor.gam.GAMLearner;
import mltk.predictor.gam.GAM;
import mltk.util.Queue;
import mltk.util.Random;
import mltk.util.tuple.Pair;
import mltk.util.tuple.IntPair;


public class InteractionTreeLearnerGAMMC{
	
	static class MyThread extends Thread {
		
		int limit;
		String prefix;
		int data_size;
		InteractionTreeNode node;
		InteractionTreeLearnerGAMMC app;
		
		MyThread(InteractionTreeLearnerGAMMC app, int data_size, String prefix, int limit) {
			this.data_size = data_size;
			this.prefix = prefix;
			this.limit = limit;
			this.app = app;
			node = null;
		}
		
		public void run() {
			try {
				node = app.createNode(data_size, prefix, limit);
			} catch (Exception e) {
				e.printStackTrace();
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
		
		@Argument(name = "-d", description = "working directory", required = true)
		String dir = "";
		
		@Argument(name = "-r", description = "attribute file", required = true)
		String attPath = "";
		
		@Argument(name = "-t", description = "training set", required = true)
		String trainPath = "";
		
		@Argument(name = "-o", description = "output tree structure file", required = true)
		String outputPath = "";
		
		@Argument(name = "-a", description = "alpha (default: 0.0, full tree)")
		double alpha = 0.0;
		
		@Argument(name = "-l", description = "leaf size (default: 15K)")
		int leafSize = 15000;
	
		@Argument(name = "-w", description = "strong interaction threshold (default: 7)")
		double wThreshold = 7;
		
		@Argument(name = "-g", description = "column with the group id (default: 0, no grouping)")
		int group = 0;
		
	}
	
	private Options opts;
	private AttrInfo ainfo;
	
	public InteractionTreeLearnerGAMMC(Options opts) throws IOException {
		this.opts = opts;
		ainfo = AttributesReader.read(opts.attPath);
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
			System.out.println("Error: the config directory " + opts.prefix + " does not exist.");
			System.exit(1);
		}
		File cfgFile = new File(opts.prefix + "/env.config");
		if (!cfgFile.exists()) {
			System.out.println("Error: the config file " + opts.prefix + "/env.config does not exist. Wrong config directory?");
			System.exit(1);
		}
			
		File dir = new File(opts.dir);
		if (!dir.exists()) {
			System.out.println("Error: the working directory " + opts.dir + " does not exist.");
			System.exit(1);
		}		
	
		File attrFile = new File(opts.attPath);
		if (!attrFile.exists()) {
			System.out.println("Error: the attribute file " + opts.attPath + " does not exist.");
			System.exit(1);
		}
		
		File outFile = new File(opts.outputPath);
		if (outFile.exists() && !outFile.isFile()) {
			System.out.println("Error: the output file " + opts.outputPath + " is not a regular file.");
			System.exit(1);
		}
			
		opts.prefix += File.separator;
		AG = opts.prefix + "fast_interactions.sh";
		BT = opts.prefix + "bt.sh";
		VIS_EFFECT = opts.prefix + "vis_effect.sh";
		VIS_IPLOT = opts.prefix + "vis_iplot.sh";
		RND = opts.prefix + "rnd.sh";
		VIS_MV = opts.prefix + "vis_mv.sh";
		VIS_SPLIT = opts.prefix + "vis_split.sh";
		tempDir = opts.dir + File.separator + "tmp";
	
		File root = new File(tempDir + "_Root");
		if (!root.exists()) {
			root.mkdir();
		}
 		FileSystem fs = FileSystems.getDefault();
 		String attrDest = tempDir + "_Root" + File.separator + "ltr.attr";
 		String dataDest = tempDir + "_Root" + File.separator + "ltr.dta";		
 
		Files.copy(fs.getPath(opts.attPath), fs.getPath(attrDest), StandardCopyOption.REPLACE_EXISTING);
		
		BufferedReader br = new BufferedReader(new FileReader(opts.trainPath), 65535);
		BufferedWriter data_out = new BufferedWriter(new FileWriter(dataDest));
		int data_size = 0;
		for (String line = br.readLine(); line != null; data_size++) {
				data_out.write(line + "\n");
				line = br.readLine();
		}
		br.close();			
		data_out.flush();
		data_out.close();
		
		long start = System.currentTimeMillis();
		InteractionTreeLearnerGAMMC app = new InteractionTreeLearnerGAMMC(opts);
		InteractionTree tree = app.build(data_size);
		long end = System.currentTimeMillis();
		
		System.out.println("Finished building tree in " + (end - start) / 1000.0 + " (s).");
		PrintWriter out = new PrintWriter(opts.outputPath);
		tree.writeStructure(out);
		out.flush();
		out.close();
	}

	public InteractionTree build(int data_size) throws Exception {
		int limit = Math.max((int)(data_size * opts.alpha), opts.leafSize);
		InteractionTree tree = new InteractionTree();
		Map<InteractionTreeNode, String> prefix = new HashMap<>();
		tree.root = createNode(data_size, "Root", limit);
		prefix.put(tree.root, "Root");
		Queue<InteractionTreeNode> q = new Queue<>();
		q.enqueue(tree.root);
		
		while (!q.isEmpty()) {
			InteractionTreeNode node = q.dequeue();

			
			if (!node.isLeaf()) {
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
		 		String attrSrc = tempDir + "_Root" + File.separator + "ltr.attr";
		 		String attrDestL = dirStr_L + File.separator + "ltr.attr";
		 		String attrDestR = dirStr_R + File.separator + "ltr.attr";
		 		FileSystem fs = FileSystems.getDefault();
				Files.copy(fs.getPath(attrSrc), fs.getPath(attrDestL), StandardCopyOption.REPLACE_EXISTING);				
				Files.copy(fs.getPath(attrSrc), fs.getPath(attrDestR), StandardCopyOption.REPLACE_EXISTING);				
				
				String fileName = File.separator + "ltr.dta";
				String dataStr = dirStr_cur + fileName;
				String dataStrL = dirStr_L + fileName;
				String dataStrR = dirStr_R + fileName;
				
				IntPair sizes = split(ainfo.columns.get(interiorNode.attIndex), 
						interiorNode.splitPoint, dataStr, dataStrL, dataStrR);
				
				MyThread lThread = new MyThread(this, sizes.v1, pre + "_L", limit);
				MyThread rThread = new MyThread(this, sizes.v2, pre + "_R", limit);
				lThread.start();
				rThread.start();
				lThread.join();
				rThread.join();
				
				interiorNode.left = lThread.node;
				interiorNode.right = rThread.node;
				
				prefix.put(interiorNode.left, pre + "_L");
				prefix.put(interiorNode.right, pre + "_R");
				
				q.enqueue(interiorNode.left);
				q.enqueue(interiorNode.right);
			}
			
		}
		
		return tree;
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
	
	protected static IntPair split(int col, double splitPoint, 
			String dataFile, String outLeftFile, String outRightFile) throws IOException {
		
		BufferedReader br = new BufferedReader(new FileReader(dataFile), 65535);
		BufferedWriter outLeft = new BufferedWriter(new FileWriter(outLeftFile));
		BufferedWriter outRight = new BufferedWriter(new FileWriter(outRightFile));
		int ln = 0; int rn = 0;
		
		String line = br.readLine();
		while (line != null) {
			String[] data = line.split("\t+");
			line += "\n";
			double value = data[col].equals("?") ?
							Double.NaN :
							Double.parseDouble(data[col]);
			if ((value <= splitPoint) || Double.isNaN(splitPoint) && Double.isNaN(value)) {
				outLeft.write(line);
				ln++;
			} else {
				outRight.write(line);
				rn++;
			}
			line = br.readLine();
		}

		br.close();			
		outLeft.flush();
		outLeft.close();
		outRight.flush();
		outRight.close();
		
		return new IntPair(ln, rn);
	}
	
	protected static void timeStamp(String msg){
		Date tmpDate = new Date();
		System.out.println("TIMESTAMP >>>> ".concat(tmpDate.toString()).concat(": ").concat(msg));
	}
	
	protected InteractionTreeNode createNode(int data_size, String prefix, int limit) 
			throws Exception {
		StringBuilder sb = new StringBuilder();
		sb.append(prefix + "\n");
		if (data_size < limit) {
			sb.append("Not enough data.\n");
			System.out.println(sb);
			return new InteractionTreeLeaf();
		}

		String tmpDir = tempDir + "_" + prefix;
		File dir = new File(tmpDir);
		String attr = tmpDir + File.separator + "ltr.attr";
		String attrfs = tmpDir + File.separator + "ltr.fs.attr";
		String attrfsfs = tmpDir + File.separator + "ltr.fs.fs.attr";
		String dtaAG = tmpDir + File.separator + "ltr.dta";
		String trainAG = tmpDir + File.separator + "ltr.train.ag";
		String validAG = tmpDir + File.separator + "ltr.valid.ag";
		String train = tmpDir + File.separator + "ltr.train.dta";
		String valid = tmpDir + File.separator + "ltr.valid.dta";

 		FileSystem fs = FileSystems.getDefault();

		// 1. Create datasets
		// 1.1. Create datasets for ag
 		
 		timeStamp("Prepare train and test data for AG.");
		double train_size =  Math.min(data_size / 3, 30000);
		double valid_size =  Math.min(data_size - train_size, 500000);
		double portion_train = train_size /  (double) data_size;
		double portion_valid = valid_size / (double) data_size;

		runProcess(dir, RND, dtaAG, "ltr", portion_valid + "", portion_train + "", opts.group + "");
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
			sb.append("All data points have the same label.\n");
			System.out.println(sb);
			return new InteractionTreeLeaf();
		}

		// 1.2 Create datasets for bt and gam 
		timeStamp("Prepare train and test data for BT and GAM.");
		train_size =  Math.min(data_size * 0.6667, 200000);
		valid_size =  Math.min(data_size - train_size, 500000);
		portion_train = train_size / (double) data_size;
		portion_valid = valid_size / (double) data_size;
		runProcess(dir, RND, dtaAG, "ltr", portion_valid + "", portion_train + "", opts.group + "");

		// 2. Fast feature selection
		timeStamp("Select 12 features for AG.");
		// Here ltr.attr -> ltr.fs.attr
		runProcess(dir, BT, attr, train, valid, "-k 12 -b 300 -a 0.01 -c roc"); 
		
		Path fsLogSrc = fs.getPath(tmpDir + File.separator + "log.txt");
		Path fsLogDst = fs.getPath(tmpDir + File.separator + "log_fs.txt");
		Path fsModelSrc = fs.getPath(tmpDir + File.separator + "model.bin");
		Path fsModelDst = fs.getPath(tmpDir + File.separator + "model_fs.bin");
		Files.copy(fsLogSrc, fsLogDst, StandardCopyOption.REPLACE_EXISTING);
		Files.copy(fsModelSrc, fsModelDst, StandardCopyOption.REPLACE_EXISTING);

		// 3. Run ag and get interactions
		timeStamp("Run AG with selected features on the small train set.");
		runProcess(dir, AG, attrfs, attrfsfs, trainAG, validAG);
		// Backup log and model
		Path agLogSrc = fs.getPath(tmpDir + File.separator + "log.txt");
		Path agLogDst = fs.getPath(tmpDir + File.separator + "log_ag.txt");
		Path agModelSrc = fs.getPath(tmpDir + File.separator + "model.bin");
		Path agModelDst = fs.getPath(tmpDir + File.separator + "model_ag.bin");
		Files.copy(agLogSrc, agLogDst, StandardCopyOption.REPLACE_EXISTING);
		Files.copy(agModelSrc, agModelDst, StandardCopyOption.REPLACE_EXISTING);

		// 4. Choose candidates
		String interactionGraph = tmpDir + File.separator + "list.txt";
		List<String> featureCandidates = getCandidates(interactionGraph, opts.wThreshold);
		
		PrintWriter out = new PrintWriter(tmpDir + File.separator + "candidates.txt");
		for (String name : featureCandidates) {
			out.println(name);
		}
		out.flush();
		out.close();

		// 5. Build a GAM for parent
 		Instances trainSet = InstancesReader.read(ainfo, train, "\t+", true);
		Instances validSet = InstancesReader.read(ainfo, valid, "\t+", true);
		timeStamp("Build a GAM for the parent node.");
		GAMLearner learner = new GAMLearner();
		learner.setMetric(new AUC());
		learner.setLearningRate(0.01);
		learner.setBaggingIters(0);
		
		//Create a copy of data with zeros instead of NaNs		
		Instances trainGAM = trainSet.copy();
		Instances validGAM = validSet.copy();
		int attrN = ainfo.attributes.size();
		for(Instance instance : trainGAM)
			for(int a = 0; a < attrN; a++)
				if(Double.isNaN(instance.getValue(a)))
					instance.setValue(a, 0);
		for(Instance instance : validGAM)
			for(int a = 0; a < attrN; a++)
				if(Double.isNaN(instance.getValue(a)))
					instance.setValue(a, 0);
		
		GAM gam = learner.buildClassifier(trainGAM, validGAM, 100, 3);
		double[] targetsValid = new double[validGAM.size()];
		double[] predsValid = new double[validGAM.size()];
		int vNo = 0;
		for (Instance instance : validGAM) {
			predsValid[vNo] = gam.regress(instance);
			targetsValid[vNo] = instance.getTarget();
			vNo++;
		}
		
		double rocParent = new AUC().eval(predsValid, targetsValid);
		sb.append("Parent ROC: " + rocParent + "\n");
		out = new PrintWriter(tmpDir + File.separator + "parent.txt");
		out.println(rocParent);
		out.flush();
		out.close();
		

		//6. Plots
		timeStamp("Visualization.");

		// Here current model.bin is produced by AG. No need to run additional scripts.
		for (String feat : featureCandidates) {
			runProcess(dir, VIS_EFFECT, attr, train, feat);
		}
		List<Pair<String, String>> pairs = getCandidates(interactionGraph);
		for (Pair<String, String> pair : pairs) {
			runProcess(dir, VIS_IPLOT, attr, train, pair.v1, pair.v2);
		}
		runProcess(dir, VIS_MV, "AG_PLOTS");
		// Backup visualization log
		Path visLogSrc = fs.getPath(tmpDir + File.separator + "log.txt");
		Path visLogDst = fs.getPath(tmpDir + File.separator + "AG_PLOTS/log_vis.txt");
		Files.copy(visLogSrc, visLogDst, StandardCopyOption.REPLACE_EXISTING);

		// Here we build a full model. Backup bagged tree.
		runProcess(dir, BT, attrfs, train, valid, "-b 300 -a 0.01 -c roc");    
		Path btLogSrc = fs.getPath(tmpDir + File.separator + "log.txt");
		Path btLogDst = fs.getPath(tmpDir + File.separator + "log_bt.txt");
		Path btModelSrc = fs.getPath(tmpDir + File.separator + "model.bin");
		Path btModelDst = fs.getPath(tmpDir + File.separator + "model_bt.bin");
		Files.copy(btLogSrc, btLogDst, StandardCopyOption.REPLACE_EXISTING);
		Files.copy(btModelSrc, btModelDst, StandardCopyOption.REPLACE_EXISTING);

		// Run visualization again.
		for (String feat : featureCandidates) {
			runProcess(dir, VIS_EFFECT, attr, train, feat);
		}
		for (Pair<String, String> pair : pairs) {
			runProcess(dir, VIS_IPLOT, attr, train, pair.v1, pair.v2);
		}

		// Backup log and plot files.
		runProcess(dir, VIS_MV, "BT_PLOTS");
		visLogSrc = fs.getPath(tmpDir + File.separator + "log.txt");
		visLogDst = fs.getPath(tmpDir + File.separator + "BT_PLOTS/log_vis.txt");
		Files.copy(visLogSrc, visLogDst, StandardCopyOption.REPLACE_EXISTING);

		// 7. Read features info: get quantiles from the effect plots
		List<Feature> features = new ArrayList<>();
		for (String key : featureCandidates) {
			Feature feature = Feature.read(tmpDir + File.separator + key + ".effect.txt");
			features.add(feature);
		}

		// 8. Evaluate splits with GAMs
		timeStamp("Evaluate splits.");
		int bestAtt = -1;
		double bestSplit = -1;
		double bestROC = -1;
		List<Integer> candidates = ainfo.getIds(featureCandidates);
		
		for (int i = 0; i < features.size(); i++) {
			int attIndex = candidates.get(i);
			FeatureSplit split = new FeatureSplit(features.get(i));		
			for (int j = 0; j < split.splits.length; j++) {
				
				double splitPoint = (split.feature.centers[j] + split.feature.centers[j + 1]) / 2;
				timeStamp("Split dataset for feature #" + i + " split #" + j + ".");
				// 8.1 Split the dataset
				Instances trainLeft = new Instances(ainfo);
				Instances trainRight = new Instances(ainfo);
				split(trainSet, attIndex, splitPoint, trainLeft, trainRight);

				Instances validLeft = new Instances(ainfo);
				Instances validRight = new Instances(ainfo);
				split(validSet, attIndex, splitPoint, validLeft, validRight);
				timeStamp("Build and evaluate the split of feature #"+i+" split #"+j+".");
				// 8.2 Build GAMMC and evaluate this split
				Instances trainLeftGAM = trainLeft.copy();
				Instances validLeftGAM = validLeft.copy();				
				Instances trainRightGAM = trainRight.copy();
				Instances validRightGAM = validRight.copy();				
				//8.2.0 Replace missing values with zeros
				timeStamp("Scan NaNs.");
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
				
				int actual_valid_size = validLeftGAM.size() + validRightGAM.size();
				double[] targets = new double[actual_valid_size];
				double[] preds = new double[actual_valid_size];
				vNo = 0;
				// GAM gamL = learner.buildClassifier(trainLeftGAM, validLeftGAM, 10000 / ainfo.attributes.size(), 3);
				GAM gamL = learner.buildClassifier(trainLeftGAM, validLeftGAM, 100, 3);
				for (Instance instance : validLeftGAM) {
					targets[vNo] = instance.getTarget();
					preds[vNo] = gamL.regress(instance);
					vNo++;
				}

				// GAM gamR = learner.buildClassifier(trainRightGAM, validRightGAM, 10000 / ainfo.attributes.size(), 3);
				GAM gamR = learner.buildClassifier(trainRightGAM, validRightGAM, 100, 3);
				for (Instance instance : validRightGAM) {
					targets[vNo] = instance.getTarget();
					preds[vNo] = gamR.regress(instance);
					vNo++;
				}

				double rocSplit = new AUC().eval(preds, targets);
				
				out = new PrintWriter(tmpDir + File.separator + "split_" + split.feature.name + "_" + splitPoint + ".txt");
				out.println(rocSplit);		
				out.flush();
				out.close();
				
				if (rocSplit > bestROC) {
					bestROC = rocSplit;
					bestAtt = attIndex;
					bestSplit = splitPoint;
				}
			}
			
		}
	
		//9. Final output: best split and its visualization
		if (bestAtt >= 0) {
			sb.append("Best ROC: " + bestROC + "\n");
			if (bestROC > rocParent) {
				sb.append("Best feature: " + ainfo.attributes.get(bestAtt).getName() + "\n");
				sb.append("Best split: " + bestSplit + "\n");
				// Visualize the splits
				//runProcess(dir, VIS_SPLIT, "AG_PLOTS", ainfo.attributes.get(bestAtt).getName(), bestSplit+"");
				runProcess(dir, VIS_SPLIT, "BT_PLOTS", ainfo.attributes.get(bestAtt).getName(), bestSplit+"");
			}
			else {
				sb.append("No improvement found.\n");
				System.out.println(sb);
				return new InteractionTreeLeaf();
			}
		} else {
			sb.append("No interactions found.\n");
			System.out.println(sb);
			return new InteractionTreeLeaf();
		}
		System.out.println(sb);
		return new InteractionTreeInteriorNode(bestAtt, bestSplit);
		
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
	
	
	protected static List<Pair<String, String>> getCandidates(String interactionGraph) 
			throws Exception {
		List<Pair<String, String>> candidates = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(interactionGraph));
		for (;;) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			String[] data = line.split("\\s+");
			if (data[2].indexOf("inf") >= 0 || data[2].indexOf("nan") >= 0) {
				continue;
			}
			double w = Double.parseDouble(data[2]);
			if (w >= 3) {
				candidates.add(new Pair<String, String>(data[0], data[1]));
			}
		}
		br.close();
		
		return candidates;
	}
	
	protected static List<String> getCandidates(String interactionGraph, double wThreshold) 
			throws Exception {
		List<String> candidates = new ArrayList<>();
		Map<String, Map<String, Double>> graph = new HashMap<>();

		BufferedReader br = new BufferedReader(new FileReader(interactionGraph));
		for (;;) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			String[] data = line.split("\\s+");
			String x = data[0];
			String y = data[1];
			if (data[2].indexOf("inf") >= 0 || data[2].indexOf("nan") >= 0) {
				continue;
			}
			double w = Double.parseDouble(data[2]);
			if (w >= 3) {
				if (!graph.containsKey(x)) {
					graph.put(x, new HashMap<String, Double>());
				}
				if (!graph.containsKey(y)) {
					graph.put(y, new HashMap<String, Double>());
				}
				graph.get(x).put(y, w);
				graph.get(y).put(x, w);
			}
			
		}
		br.close();
		
		Set<String> strongCands = new HashSet<>();
		Set<String> weakCands = new HashSet<>();

		for (String node : graph.keySet()) {
			Map<String, Double> adjList = graph.get(node);
			int weakN = 0;
			for (double w : adjList.values()) {
				if(w >= wThreshold) {
					strongCands.add(node);
				}
				if(w >= 3) {
					weakN++;
				}
			}
			if(weakN >= 3) {
				weakCands.add(node);
			}
		}
		
		Set<String> supCandidates;
		if (strongCands.size() > 0)
			supCandidates = strongCands;
		else
			supCandidates = weakCands;
		
		while (graph.size() > 0 && supCandidates.size() > 0 && candidates.size() < 3) {
			double maxWSum = -1;
			String bestNode = "";
			
			for (String node : supCandidates) {
				Map<String, Double> adjList = graph.get(node);
				double wSum = 0;
				for (double w : adjList.values()) {
					wSum += w;
				}
				if (wSum > maxWSum) {
					maxWSum = wSum;
					bestNode = node;
				}
			}
			
			candidates.add(bestNode);
			supCandidates.remove(bestNode);

			Map<String, Double> adjList = graph.get(bestNode);
			for (String neighbor : adjList.keySet()) {
				graph.get(neighbor).remove(bestNode);
			}
			graph.remove(bestNode);
		}
		
		return candidates;
	}
	
}
