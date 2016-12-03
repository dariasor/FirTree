package lrtree;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.core.FastVector;

/* 
import com.linkedin.ltr.Evaluation; --> import mltk.predictor.evaluation.AUC;
import com.linkedin.ltr.core.io.DenseInstancesReader; --> import mltk.core.io.InstancesReader;
import com.linkedin.ltr.util.Element; --> import mltk.util.Element;
*/

import mltk.predictor.Regressor;
import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.Instance;
import mltk.core.Instances;
import mltk.core.Attribute;
import mltk.core.io.AttrInfo;
import mltk.core.io.AttributesReader;
import mltk.core.io.InstancesReader;
import mltk.predictor.evaluation.AUC;
import mltk.predictor.gam.GAMLearner;
import mltk.predictor.gam.GAM;
import mltk.predictor.gam.PolynomialSpline;
import mltk.util.Queue;
import mltk.util.Random;
import mltk.util.Element;
import mltk.util.tuple.Pair;
import mltk.util.tuple.IntPair;
/**
 * Class for building leaf models.
 * 
 * @author ylou
 *
 */
public class InteractionTreeBuilderPolyWekaFSTopK {
	
	@Argument(name = "-r", description = "attribute file path", required = true)
	String attPath = "";
	
	@Argument(name = "-t", description = "training set path", required = true)
	String trainPath = "";
	
	@Argument(name = "-i", description = "input model path", required = true)
	String inputPath = "";
	
	@Argument(name = "-o", description = "output model path", required = true)
	String outputPath = "";
	
	@Argument(name = "-T", description = "test set path")
	String testPath = null;
	
	@Argument(name = "-f", description = "working directory path", required = true)
	String dirPath = "";
	
	@Argument(name = "-d", description = "degree of polynomial fitting (default: 3)")
	int deg = 3;
	
	@Argument(name = "-k", description = "number of polynomial transformations (default: 3)")
	int k = 3;
	
	@Argument(name = "-s", description = "random seed (default: 0)")
	long seed = 3;
	
	public static void main(String[] args) throws Exception {
		InteractionTreeBuilderPolyWekaFSTopK app = new InteractionTreeBuilderPolyWekaFSTopK();
		CmdLineParser parser = new CmdLineParser(app);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		
		Random.getInstance().setSeed(app.seed);
		
		app.dirPath += "/tmp_";
		
		Instances instances = InstancesReader.read(app.attPath, app.trainPath,  "\t+", true);
		
		System.out.println("Reading selected features");
		
		InteractionTree tree = new InteractionTree();
		BufferedReader in = new BufferedReader(new FileReader(app.inputPath));
		tree.readStructure(in);
		in.close();
		
		System.out.println("Start building trees");
		long start = System.currentTimeMillis();
		build(tree, instances, app.deg, app.k, app.dirPath);
		long end = System.currentTimeMillis();
		System.out.println("Finished building trees");
		System.out.println("Time: " + (end - start) / 1000.0 + " (s).");
		
		PrintWriter out = new PrintWriter(app.outputPath);
		tree.writeModel(out);
		out.flush();
		out.close();
		
		if (app.testPath != null) {
			Instances test = InstancesReader.read(app.attPath, app.trainPath, "\t+", true);
			System.out.println("Error: " + evaluateError(tree, test));
			double roc = evaluateROC(tree, test);
			System.out.println("AUC: " + roc);
		}
		
	}

	public static void build(InteractionTree tree, Instances instances, int deg, int k, String dir) throws Exception {
		Map<InteractionTreeLeaf, Instances> datasets = new HashMap<>();
		List<Attribute> attributes = instances.getAttributes();
		for (Instance instance : instances) {
			InteractionTreeLeaf leaf = tree.getLeaf(instance);
			if (!datasets.containsKey(leaf)) {
				Instances dataset = new Instances(attributes, instances.getClassAttribute());
				datasets.put(leaf, dataset);
			}
			datasets.get(leaf).add(instance);
		}
		
		double[] lambda = {0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005};
		Map<InteractionTreeLeaf, Instances> trains = new HashMap<>();
		Instances validfull = new Instances(attributes, instances.getClassAttribute());
		List<InteractionTreeLeaf> leaves = new ArrayList<>();
		tree.getLeaves(leaves);
		Map<InteractionTreeNode, String> paths = new HashMap<>();
		tree.computePaths(paths);
		for (InteractionTreeLeaf leaf : leaves) {
			Instances dataset = datasets.get(leaf);
			Instances train = new Instances(attributes, instances.getClassAttribute());
			Instances valid = new Instances(attributes, instances.getClassAttribute());
			trains.put(leaf, train);
			split(dataset, 0.2, train, valid);
			for (Instance instance : valid) {
				validfull.add(instance);
			}
			System.out.println("Splitting into training (" + train.size() + ") and validation (" + valid.size() + ")");
		}
		
		Map<InteractionTreeLeaf, Set<String>> selectedFeatures = new HashMap<>();
		double bestLambda = 0;
		double bestROC = 0;
		for (double l : lambda) {
			int count = 1;
			System.out.println("Testing lambda " + l);
			for (InteractionTreeLeaf leaf : leaves) {
				Instances train = trains.get(leaf);
				System.out.println("Building leaf node " + count++ + "/" + datasets.size());
				String fsFile = dir + paths.get(leaf) + "/log_ag.txt";
				Set<String> selected = new HashSet<>();
				try {
					BufferedReader br = new BufferedReader(new FileReader(fsFile));

					boolean found = false;
					out:
					for (;;) {
						String line = br.readLine();
						if (line == null) {
							break;
						}
						if (line.startsWith("Resulting set of attributes:")) {
							found = true;
							for (;;) {
								String feat = br.readLine();
								if (feat.equalsIgnoreCase("")) {
									break out;
								}
								selected.add(feat);
							}
						}
					}
					if (!found) {
						System.out.println("Something is wrong with this file: " + fsFile);
						System.exit(1);
					}
				} catch (FileNotFoundException e) {
			
				}
				selectedFeatures.put(leaf, selected);
				leaf.gam = buildModel(train, l * train.size(), deg, k, selected);
			}
			double roc = evaluateROC(tree, validfull);
			if (roc > bestROC) {
				bestROC = roc;
				bestLambda = l;
			}
		}
		System.out.println("Best lambda: " + bestLambda);
		
		int count = 1;
		for (InteractionTreeLeaf leaf : datasets.keySet()) {
			Instances dataset = datasets.get(leaf);
			Set<String> selected = selectedFeatures.get(leaf);
			//DenseInstancesWriter2.write(dataset, "dataset_" + count + ".txt");
			System.out.println("Building leaf node " + count++ + "/" + datasets.size());
			leaf.gam = buildModel(dataset, bestLambda * dataset.size(), deg, k, selected);
		}
	}
	
	protected static GAM buildModel(Instances instances, double lambda, int deg, int topk, Set<String> selected) throws Exception {
		// Convert to Weka's object
		List<Attribute> attributes = instances.getAttributes();
		FastVector attInfo = new FastVector();
		for (int i = 0; i < attributes.size(); i++) {
			Attribute attr = attributes.get(i);
			if (selected.contains(attr.getName())) {
				for (int j = 0; j < deg; j++) {
					weka.core.Attribute att = new weka.core.Attribute(attr.getName() + "_" + j, i);
					attInfo.addElement(att);
				}
			} else {
				weka.core.Attribute att = new weka.core.Attribute(attr.getName(), i);
				attInfo.addElement(att);
			}
		}
		FastVector strs = new FastVector();
		strs.addElement("0");
		strs.addElement("1");
		weka.core.Attribute cls = new weka.core.Attribute(instances.getClassAttribute().getName(), strs, attributes.size());
		attInfo.addElement(cls);
		weka.core.Instances dataset = new weka.core.Instances("", attInfo, instances.size());
		for (Instance instance : instances) {
			double[] values = instance.getValues();
			double[] v = new double[attInfo.size()];
			int k = 0;
			for (int i = 0; i < attributes.size(); i++) {
				Attribute attr = attributes.get(i);
				if (selected.contains(attr.getName())) {
					for (int j = 0; j < deg; j++) {
						v[k++] = Math.pow(values[i], j + 1);
					}
				} else {
					v[k++] = values[i];
				}
			}
			v[k] = instance.getClassValue();
			weka.core.Instance ins = new weka.core.Instance(instance.getWeight(), v);
			dataset.add(ins);
		}
		
		dataset.setClassIndex(dataset.numAttributes() - 1);
		GAM gam = fit(dataset, attributes, lambda);
		
		List<Regressor> regressors = gam.getRegressors();
		List<Element<Integer>> weights = new ArrayList<>(regressors.size());
		for (int i = 0; i < regressors.size(); i++) {
			PolynomialSpline spline = (PolynomialSpline) regressors.get(i);
			double w = getWeight(instances, spline);
			weights.add(new Element<Integer>(spline.getAttributeIndex(), w));
		}
		Collections.sort(weights);
		Collections.reverse(weights);
		
		Set<String> wekaSelected = new HashSet<>();
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < Math.min(Math.min(weights.size(), topk), selected.size()); i++) {
			wekaSelected.add(attributes.get(weights.get(i).element).getName());
			sb.append(attributes.get(weights.get(i).element).getName()).append(" ").append(weights.get(i).weight).append(" ");
		}
		if(sb.length() != 0) {
			System.out.println("Polynomial transformations: " + sb.toString());
		} else {
			System.out.println("No transformations");
		}
		
		// Refit using weka
		List<Integer> toRemove = new ArrayList<>();
		for (int i = 0; i < dataset.numAttributes() - 1; i++) {
			weka.core.Attribute attribute = dataset.attribute(i);
			String[] data = attribute.name().split("_");
			if (data.length == 2) {
				String name = data[0];
				int d = Integer.parseInt(data[1]);
				if (!wekaSelected.contains(name) && d != 0) {
					toRemove.add(i);
				}
			}
		}
		for (int i = toRemove.size() - 1; i >= 0; i--) {
			dataset.deleteAttributeAt(toRemove.get(i));
		}
		dataset.setClassIndex(dataset.numAttributes() - 1);
		
		GAM gamNew = fit(dataset, attributes, lambda);
		
		List<Regressor> newRegressors = gamNew.getRegressors();
		for (int i = 0; i < newRegressors.size(); i++) {
			((PolynomialSpline)(newRegressors.get(i))).setLeftEdge(((PolynomialSpline)(gam.getRegressors().get(i))).getLeftEdge());
			((PolynomialSpline)(newRegressors.get(i))).setRightEdge(((PolynomialSpline)(gam.getRegressors().get(i))).getRightEdge());
		}
		
		return gamNew;
	}

	protected static double getWeight(Instances instances, PolynomialSpline spline) throws Exception {
		
		int attIndex = spline.getAttributeIndex();
		
		List<Double> values = new ArrayList<>();
		for (Instance instance : instances) {
			values.add(instance.getValue(attIndex));
		}
		Collections.sort(values);
		int leftEdgeInx = values.size() / 200;
		int rightEdgeInx = values.size() - leftEdgeInx - 1;
		spline.setLeftEdge(values.get(leftEdgeInx));
		spline.setRightEdge(values.get(rightEdgeInx));
		
		FastVector attInfo = new FastVector();
		weka.core.Attribute att = new weka.core.Attribute("x", 0);
		attInfo.addElement(att);
		weka.core.Attribute cls = new weka.core.Attribute("target", 1);
		attInfo.addElement(cls);
		weka.core.Instances dataset = new weka.core.Instances("", attInfo, instances.size());
		for (int i = leftEdgeInx; i <= rightEdgeInx; i++) {
			double[] v = new double[] {values.get(i), spline.regress(values.get(i))};
			weka.core.Instance ins = new weka.core.Instance(1.0, v);
			dataset.add(ins);
		}
		
		dataset.setClassIndex(1);
		
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(dataset);
		
		double w = 0;
		for (int i = leftEdgeInx; i <= rightEdgeInx; i++) {
			double p1 = spline.regress(values.get(i));
			double p2 = lr.classifyInstance(dataset.instance(i - leftEdgeInx));
			double d = p1 - p2;
			w += d * d;
		}
		return w;
	}
	
	protected static GAM fit(weka.core.Instances dataset, List<Attribute> attributes, double lambda) throws Exception {
		Logistic lr = new Logistic();
		lr.setRidge(lambda);
		lr.buildClassifier(dataset);
		
		String[] lrInfo = lr.toString().split("\n");
		// Parse weka's LR
		GAM gam = new GAM();
		Map<String, List<Double>> map = new HashMap<>();
		for (int i = 5; i < lrInfo.length; i++) {
			String line = lrInfo[i];
			String[] data = line.trim().split("\\s+");
			String name = data[0].split("_")[0];
			Double w = -Double.valueOf(data[1]);
			if (data[0].equalsIgnoreCase("intercept")) {
				gam.setIntercept(w);
				break;
			} else {
				if (!map.containsKey(name)) {
					map.put(name, new ArrayList<Double>());
				}
				map.get(name).add(w);
			}
		}
		List<Element<List<Double>>> list = new ArrayList<>();
		for (Map.Entry<String, List<Double>> entry : map.entrySet()) {
			list.add(new Element<List<Double>>(entry.getValue(), getAttIndex(attributes, entry.getKey())));
		}
		Collections.sort(list);
		
		for (Element<List<Double>> element : list) {
			int attIndex = (int) element.weight;
			List<Double> weights = element.element;
			double[] w = new double[weights.size()];
			for (int i = 0; i < w.length; i++) {
				w[i] = weights.get(i);
			}
			PolynomialSpline spline = new PolynomialSpline(w, attIndex);
			gam.add(new int[] {attIndex}, spline);
		}
		return gam;
	}
	
	protected static int getAttIndex(List<Attribute> attributes, String name) {
		for (Attribute attribute : attributes) {
			if (attribute.getName().equalsIgnoreCase(name)) {
				return attribute.getIndex();
			}
		}
		return -1;
	}
	
	protected static void split(Instances instances, double ratio, Instances train, Instances valid) {
		Map<String, List<Instance>> map = new HashMap<>();
		for (Instance instance : instances) {
			String query = instance.getQuery();
			if (!map.containsKey(query)) {
				map.put(query, new ArrayList<Instance>());
			}
			map.get(query).add(instance);
		}
		
		int nValid = (int) (instances.size() * ratio);
		List<String> queries = new ArrayList<>(map.keySet());
		while (valid.size() < nValid) {
			int idx = Random.getInstance().nextInt(queries.size());
			List<Instance> l = map.get(queries.get(idx));
			for (Instance instance : l) {
				valid.add(instance);
			}
			queries.remove(idx);
		}
		
		for (int i = 0; i < queries.size(); i++) {
			List<Instance> l = map.get(queries.get(i));
			for (Instance instance : l) {
				train.add(instance);
			}
		}
	}
	
	/*
	protected static void split(Instances instances, double ratio, Instances train, Instances valid) {
		Map<String, List<Instance>> map = new HashMap<>();
		for (Instance instance : instances) {
			String query = instance.getQuery();
			if (!map.containsKey(query)) {
				map.put(query, new ArrayList<Instance>());
			}
			map.get(query).add(instance);
		}
		
		List<Element<List<Instance>>> list = new ArrayList<>();
		for (List<Instance> v : map.values()) {
			list.add(new Element<List<Instance>>(v, v.size()));
		}
		Collections.sort(list);
		Collections.reverse(list);
		
		int nValid = (int) (instances.size() * ratio);
		int i = 0;
		while (valid.size() < nValid) {
			List<Instance> l = list.get(i).element;
			for (Instance instance : l) {
				valid.add(instance);
			}
			i++;
		}
		for (; i < list.size(); i++) {
			List<Instance> l = list.get(i).element;
			for (Instance instance : l) {
				train.add(instance);
			}
		}
	}
	*/
	
	protected void split(Instances instances, int attIndex, double splitPoint,
			Instances left, Instances right) {
		for (Instance instance : instances) {
			if (instance.getValue(attIndex) <= splitPoint) {
				left.add(instance);
			} else {
				right.add(instance);
			}
		}
	}
	
	static double evaluateROC(GAM gam, Instances instances) {
		double[] preds = new double[instances.size()];
		double[] targets = new double[instances.size()];
		for (int i = 0; i < instances.size(); i++) {
			Instance instance = instances.get(i);
			double pred = gam.regress(instance);
			preds[i] = 1.0 / (1 + Math.exp(-pred));
			targets[i] = instance.getClassValue();
		}
		double roc = new AUC().eval(preds, targets);
		return roc;
	}

	static double evaluateROC(InteractionTree tree, Instances instances) {
		double[] preds = new double[instances.size()];
		double[] targets = new double[instances.size()];
		for (int i = 0; i < instances.size(); i++) {
			Instance instance = instances.get(i);
			GAM gam = tree.getLeaf(instance).gam;
			double pred = gam.regress(instance);
			preds[i] = 1.0 / (1 + Math.exp(-pred));
			targets[i] = instance.getClassValue();
		}
		double roc = new AUC().eval(preds, targets);
		return roc;
	}
	
	static double evaluateError(InteractionTree tree, Instances instances) {
		double error = 0.0;
		for (int i = 0; i < instances.size(); i++) {
			Instance instance = instances.get(i);
			GAM gam = tree.getLeaf(instance).gam;
			double pred = gam.classify(instance);
			if (pred != instance.getClassValue()) {
				error++;
			}
		}
		return error / instances.size();
	}
	
}
