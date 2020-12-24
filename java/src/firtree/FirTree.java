package firtree;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import firtree.utilities.Instance;
import mltk.core.io.AttrInfo;
import mltk.util.tuple.IntPair;

public class FirTree {

	private enum NodeType { SPLIT, MODEL, CONST }

	public String dir;
	public List<List<Integer>> nodeAttIdList;
	
	public AttrInfo ainfo;
	private int polyDegree; //degree of polynomials in the leaf models. 0 means that the leaf models are not yet trained.
	private String modelPrefix;

	private ArrayList<String> node_name;
	private ArrayList<NodeType> node_type;
	private ArrayList<Integer> split_attr_id;
	private ArrayList<Double> split_val;
	private int nodeN;
	
	// XW. Outer: Number of all nodes of the three NodeType
	// XW. Inner: Number of attributes used by a node
	public ArrayList<ArrayList<Integer>> lr_attr_ids;
	
	// XW. Trainable parameters
	private ArrayList<ArrayList<ArrayList<Double>>> lr_coefs; 
	// XW. Outer: Number of all nodes of the three NodeType
	// XW. Middle: Number of attributes used by a leaf node
	// XW. Inner: Polynomial terms, min value and max value of an attribute

	// XW. Trainable parameters
	private double[] intercept_val;
	// XW. Number of all nodes of the three NodeType

	// XW. Non-trainable parameters
	private double[] const_val;
	
	// XW. Map names to indexes of node_name for fast index look up
	public Map<String, Integer> nodeIndexes;

	// XW
	private int INTERCEPT = -1;
	
	public FirTree(AttrInfo ainfo, String logPath, int polyDegree, String modelPrefix) 
			throws Exception {
		this(ainfo, logPath, polyDegree, modelPrefix, 0);
	}
	
	public FirTree(AttrInfo ainfo, String logPath, int polyDegree, String modelPrefix, int override) 
			throws Exception {
		this.ainfo = ainfo;
		this.polyDegree = polyDegree;
		this.modelPrefix = modelPrefix;
		
		File logFile = new File(logPath);
		if (!logFile.exists() || !logFile.isFile()) { 
			System.err.printf("treelog.txt does not exist in %s\n", logPath);
			System.exit(1);
		}
		
		this.dir = Paths.get(logPath).getParent().toString();
		
		node_name = new ArrayList<String>();
		node_type = new ArrayList<NodeType>();
		split_attr_id = new ArrayList<Integer>();
		split_val = new ArrayList<Double>();
		
		nodeAttIdList = new ArrayList<List<Integer>>();

		// Parse the treelog.txt file in the directory to a FirTree model
		String strLog = new String(Files.readAllBytes(Paths.get(logPath)));
		String[] strNodes = strLog.split("\n\n");
		for (String strNode : strNodes) {
			List<Integer> attIdList = new ArrayList<Integer>();
			
			String[] lines = strNode.strip().split("\n");

			String first = lines[0].trim();
			if (! first.matches("Root(.*)")) {
				System.err.printf("Error: %s is not a valid node name\n", first);
				System.exit(1);
			}
			node_name.add(first);
			
			String last = lines[lines.length - 1].trim();
			if (last.matches("Constant leaf(.*)")) {
				node_type.add(NodeType.CONST);
				split_attr_id.add(-1);
				split_val.add(Double.POSITIVE_INFINITY);
			} else if (last.matches("Regression leaf(.*)")) {
				node_type.add(NodeType.MODEL);
				split_attr_id.add(-1);
				split_val.add(Double.POSITIVE_INFINITY);
				
				if (lines[1].matches("Core features(.*)")) {
					for (int i = 2; i < lines.length; i ++) {
						if (! lines[i].startsWith("\t")) {
							break;
						}
						attIdList.add(ainfo.nameToId.get(lines[i].strip()));
					}
				} else {
					System.err.printf("Error: no core features for %s\n", first);
					System.exit(1);
				}
			} else if (last.matches("Best split(.*)")) {
				node_type.add(NodeType.SPLIT);
				split_attr_id.add(ainfo.nameToId.get(lines[lines.length - 2].split(": ")[1]));
				split_val.add(Double.parseDouble(last.split(": ")[1]));
			} else {
				System.err.printf("Error: can't parse node type %s\n", last);
				System.exit(1);
			}
			
			nodeAttIdList.add(attIdList);
		}
		/*//
		BufferedReader treelog = new BufferedReader(new FileReader(logPath), 65535);
		String line_tree = treelog.readLine();
		while(line_tree != null) {
			if(line_tree.matches("Root(.*)")) {
				node_name.add(line_tree.trim());
			}
			if(line_tree.matches("Constant leaf(.*)")) {
				node_type.add(NodeType.CONST);
				split_attr_id.add(-1);
				split_val.add(Double.POSITIVE_INFINITY);
			}
			if(line_tree.matches("Regression leaf(.*)")) {
				node_type.add(NodeType.MODEL);
				split_attr_id.add(-1);
				split_val.add(Double.POSITIVE_INFINITY);
			}
			if(line_tree.matches("Best feature:(.*)")) {
				node_type.add(NodeType.SPLIT);
				String current_attr = line_tree.split(" ")[2];
				split_attr_id.add(ainfo.nameToId.get(current_attr));
				line_tree = treelog.readLine();
				split_val.add(Double.parseDouble(line_tree.split(" ")[2]));
			}
			line_tree = treelog.readLine();
		}
		treelog.close();
		*///
		nodeN = node_name.size();
		
		// XW. Build map from names to indexes of node_name
		nodeIndexes = new HashMap<>();
		for (int nodeNo = 0; nodeNo < node_name.size(); nodeNo ++) {
			nodeIndexes.put(node_name.get(nodeNo), nodeNo);
		}
		
		// XW. Check whether all parameter files exist
		boolean allExist = true;
		List<String> allLeaves = getAllLeaves();
		for (String leafName : allLeaves) {
			String paramPath = getParamPath(leafName);
			File paramFile = new File(paramPath);
			if (! paramFile.exists()) {
				allExist = false;
				break;
			}
		}
		if (override == 1) {
			allExist = false;
		}
		
		// XW. If all parameter files exist, load all of the parameters
		// XW. Parameters include intercept, coefficients, and constants on leaves
		const_val = new double[nodeN];
		intercept_val = new double[nodeN];
		lr_attr_ids = new ArrayList<ArrayList<Integer>>(nodeN);
		lr_coefs = new ArrayList<ArrayList<ArrayList<Double>>>(nodeN);
		if (allExist) {
			timeStamp("Load existing model parameters");
			
			// Load linear models and constants on FirTree leaves
			for(int nodeNo = 0; nodeNo < node_name.size(); nodeNo++){
	
				ArrayList<Integer> hold_lr_attr_ids = new ArrayList<Integer>();
				ArrayList<ArrayList<Double>> hold_lr_attr_coefs = new ArrayList<ArrayList<Double>>();
	
				if(node_type.get(nodeNo) == NodeType.MODEL) {
					// there is a model on the leaf
					String paramPath = getParamPath(nodeNo);
					BufferedReader lr_text = new BufferedReader(new FileReader(paramPath));
					String line = lr_text.readLine();
					intercept_val[nodeNo] = Double.parseDouble(line.split("\t")[1]);
					line = lr_text.readLine();
					while(line != null) {
						String[] lr_data_string = line.split("\t");
						Integer attr_id = ainfo.nameToId.get(lr_data_string[0]);
						if(attr_id == null)
						{
							System.out.println("Error: not a valid attribute name " + lr_data_string[0] + " in " + paramPath);
							System.exit(1);
						}
						hold_lr_attr_ids.add(attr_id);
	
						int size = polyDegree + 2;
						if (lr_data_string.length >= polyDegree + 5) {
							size = polyDegree + 4;
						}
						ArrayList<Double> tempnode_tempattr_model_coef = new ArrayList<Double>();
						for (int i = 0; i < size; i ++) {
							// the last two items are (min and max) range of the attribute
							// the previous items are coefficients of the corresponding polynomial terms of the attribute
							try {
								tempnode_tempattr_model_coef.add(Double.parseDouble(lr_data_string[i + 1]));
							} catch(Exception e) {
								System.out.println("Error: can't parse " + lr_data_string[i + 1] + " in " + paramPath);
								System.exit(1);
							}
						}
						hold_lr_attr_coefs.add(tempnode_tempattr_model_coef);
						line = lr_text.readLine();
					}
					lr_attr_ids.add(hold_lr_attr_ids);
					lr_coefs.add(hold_lr_attr_coefs);
					lr_text.close();
				} else {
					// there is no model on the node
					lr_attr_ids.add(hold_lr_attr_ids);
					lr_coefs.add(hold_lr_attr_coefs);
					if(node_type.get(nodeNo) == NodeType.CONST) {
						// there is a const on the node
						String paramPath = getParamPath(nodeNo);
						BufferedReader constRead = new BufferedReader(new FileReader(paramPath));
						String line = constRead.readLine();
						const_val[nodeNo] = Double.parseDouble(line.split(": ")[1]);
						constRead.close();
					}
				}
			}
		}
	}
	
	/**
	 * @author Xiaojie Wang
	 * 
	 * This constructor function is deprecated for two reasons:
	 * (1) The use of polyDegree is ambiguous: it is used to indicate loading parameter files or not
	 * (2) It does not allow adding a prefix to the name of parameter files
	 */
	@Deprecated
	public FirTree(AttrInfo ainfo_in, String dir, int poly_degree_in) throws Exception {

		// Load treeforPred
		node_name = new ArrayList<String>();
		node_type = new ArrayList<NodeType>();
		split_attr_id = new ArrayList<Integer>();
		split_val = new ArrayList<Double>();
		polyDegree = poly_degree_in;
		ainfo = ainfo_in;

		BufferedReader treelog = new BufferedReader(new FileReader(dir + "/treelog.txt"), 65535);
		String line_tree = treelog.readLine();

		while(line_tree != null) {
			if(line_tree.matches("Root(.*)")) {
				node_name.add(line_tree.trim());
			}
			if(line_tree.matches("Constant leaf(.*)")) {
				node_type.add(NodeType.CONST);
				split_attr_id.add(-1);
				split_val.add(Double.POSITIVE_INFINITY);
			}
			if(line_tree.matches("Regression leaf(.*)")) {
				node_type.add(NodeType.MODEL);
				split_attr_id.add(-1);
				split_val.add(Double.POSITIVE_INFINITY);
			}
			if(line_tree.matches("Best feature:(.*)")) {
				node_type.add(NodeType.SPLIT);
				String current_attr = line_tree.split(" ")[2];
				split_attr_id.add(ainfo.nameToId.get(current_attr));
				line_tree = treelog.readLine();
				split_val.add(Double.parseDouble(line_tree.split(" ")[2]));
			}
			line_tree = treelog.readLine();
		}
		treelog.close();
		nodeN = node_name.size();
		
		// XW. Build map from names to indexes of node_name
		nodeIndexes = new HashMap<>();
		for (int nodeNo = 0; nodeNo < node_name.size(); nodeNo ++) {
			nodeIndexes.put(node_name.get(nodeNo), nodeNo);
		}
		
		if(polyDegree > 0) {
			// Load models and constants on FirTree leaves
			const_val = new double[nodeN];
			intercept_val = new double[nodeN];
			lr_attr_ids = new ArrayList<ArrayList<Integer>>(nodeN);
			lr_coefs = new ArrayList<ArrayList<ArrayList<Double>>>(nodeN);
	
			for(int nodeNo = 0; nodeNo < node_name.size(); nodeNo++){
	
				ArrayList<Integer> hold_lr_attr_ids = new ArrayList<Integer>();
				ArrayList<ArrayList<Double>> hold_lr_attr_coefs = new ArrayList<ArrayList<Double>>();
	
				if(node_type.get(nodeNo) == NodeType.MODEL) {
					// there is a model on the leaf
					String lr_file_name = dir + "/Node_" + node_name.get(nodeNo) + "/model_polydegree_" + polyDegree + ".txt";
					BufferedReader lr_text = new BufferedReader(new FileReader(lr_file_name));
					String line = lr_text.readLine();
					intercept_val[nodeNo] = Double.parseDouble(line.split("\t")[1]);
					line = lr_text.readLine();
					while(line != null) {
						String[] lr_data_string = line.split("\t");
						Integer attr_id = ainfo.nameToId.get(lr_data_string[0]);
						if(attr_id == null)
						{
							System.out.println("Error: not a valid attribute name " + lr_data_string[0] + " in " + lr_file_name);
							System.exit(1);
						}
						hold_lr_attr_ids.add(attr_id);
	
						ArrayList<Double> tempnode_tempattr_model_coef = new ArrayList<Double>();
						for(int i = 0; i < polyDegree + 2; i++){
							// the last two items are (min and max) range of the attribute
							// the previous items are coefficients of the corresponding polynomial terms of the attribute
							try {
								tempnode_tempattr_model_coef.add(Double.parseDouble(lr_data_string[i + 1]));
							} catch(Exception e) {
								System.out.println("Error: can't parse " + lr_data_string[i + 1] + " in " + lr_file_name);
								System.exit(1);
							}
						}
						hold_lr_attr_coefs.add(tempnode_tempattr_model_coef);
						line = lr_text.readLine();
					}
					lr_attr_ids.add(hold_lr_attr_ids);
					lr_coefs.add(hold_lr_attr_coefs);
					lr_text.close();
				} else {
					// there is no model on the node
					lr_attr_ids.add(hold_lr_attr_ids);
					lr_coefs.add(hold_lr_attr_coefs);
					if(node_type.get(nodeNo) == NodeType.CONST) {
						// there is a const on the node
						BufferedReader constRead = new BufferedReader(new FileReader(dir + "/Node_" + node_name.get(nodeNo) + "/model_const.txt"));
						String line = constRead.readLine();
						const_val[nodeNo] = Double.parseDouble(line.split(": ")[1]);
						constRead.close();
					}
				}
			}
		}
	}

	//return list of names of regression leaves
	public ArrayList<String> getRegressionLeaves() {
		ArrayList<String> leafNames = new ArrayList<String>();
		for(int i = 0; i < nodeN; i++)
			if(node_type.get(i) == NodeType.MODEL) {
				leafNames.add(node_name.get(i));
			}
		return leafNames;
	}
	
	//return list of names of constant leaves
	public ArrayList<String> getConstLeaves() {
		ArrayList<String> leafNames = new ArrayList<String>();
		for(int i = 0; i < nodeN; i++)
			if(node_type.get(i) == NodeType.CONST) {
				leafNames.add(node_name.get(i));
			}
		return leafNames;
	}
	

	public void outcpp(String outputPath) throws Exception
	{
		BufferedWriter cpp_out = new BufferedWriter(new FileWriter(outputPath));

		String current_node_name = "Root";
		Boolean first_time = true;

		cpp_out.write("    double prediction = 0;\n\n");

		while(true)
		{
			int current_node_index = node_name.indexOf(current_node_name);
			NodeType current_type = node_type.get(current_node_index);
			int height = current_node_name.length() - current_node_name.replace("_", "").length(); //number of "_" in the node name
			String tabs = String.join("", Collections.nCopies(height, "    ")); //sequence of tabs, each tab is represented by 4 spaces

			if(first_time)
			{
				if(current_node_name.endsWith("R"))
					cpp_out.write(tabs + "} else {\n");

				cpp_out.write(tabs + "    //" + current_node_name + "\n");

				if(current_type == NodeType.SPLIT) 	{
					double current_split_value = split_val.get(current_node_index);
					String current_split_attr = ainfo.idToName(split_attr_id.get(current_node_index));

					cpp_out.write(tabs + "    if (" + current_split_attr + " <= " + current_split_value + ") {\n");
					current_node_name += "_L";
				} else {
					first_time = false;
					if(current_type == NodeType.CONST) {
						cpp_out.write(tabs + "    prediction = " + const_val[current_node_index] + ";\n");
					} else {
						//linear regression model in the leaf
						ArrayList<ArrayList<Double>> current_lr_coefs = lr_coefs.get(current_node_index);

						for(int lr_attr_index = 0; lr_attr_index < lr_attr_ids.get(current_node_index).size(); lr_attr_index++) {
						    String current_lr_attr = ainfo.idToName(lr_attr_ids.get(current_node_index).get(lr_attr_index));
							double current_min = current_lr_coefs.get(lr_attr_index).get(polyDegree);
							double current_max = current_lr_coefs.get(lr_attr_index).get(polyDegree + 1);
							//double x1_cap = (x1 < min1) ? min1 : (x1 > max1) ? max1 : x1;
							cpp_out.write(
								tabs + "    double " + current_lr_attr + "_cap =\n" + tabs + "        (" + current_lr_attr +	" < " +
								current_min + ") ?\n" + tabs + "        " + current_min + " :\n" + tabs + "        (" + current_lr_attr + " > " +
								current_max + ") ? " + current_max + " : " + current_lr_attr + ";\n"
							);
						}
						//prediction = b0
						cpp_out.write("\n" + tabs + "    prediction = " + intercept_val[current_node_index]);
						for(int lr_attr_index = 0; lr_attr_index < lr_attr_ids.get(current_node_index).size(); lr_attr_index++) {
						    String current_lr_attr_cap = ainfo.idToName(lr_attr_ids.get(current_node_index).get(lr_attr_index)) + "_cap";
							// + x1_cap *
							cpp_out.write("\n" + tabs + "        + " + current_lr_attr_cap + " *\n" + tabs + "        ");
							for(int degree_index = 0; degree_index < polyDegree - 1; degree_index++)
								//(b11 + x1_cap * (b12 + x1_cap * ...
								cpp_out.write("(" + current_lr_coefs.get(lr_attr_index).get(degree_index) + " + " + current_lr_attr_cap + " *\n" + tabs + "        ");
							//b13)))
							cpp_out.write(current_lr_coefs.get(lr_attr_index).get(polyDegree - 1) + String.join("", Collections.nCopies(polyDegree - 1, ")")));
						}
						cpp_out.write(";\n");
					}
				}
			} else {
				if(current_node_name.endsWith("L")) {
					current_node_name = current_node_name.replaceAll("L$","R"); //replace last L with R
					first_time = true;
				} else if(current_node_name.endsWith("R")) {
					cpp_out.write(tabs + "}\n");
					current_node_name = current_node_name.replaceAll("_R$",""); //remove last "_R"
				} else {//Node_Root
					break;
				}
			}
		}

		cpp_out.close();
	}

	public void outjava(String outputPath) throws Exception
	{
		BufferedWriter java_out = new BufferedWriter(new FileWriter(outputPath));
		String current_node_name = "Root";
		Boolean first_time = true;

		java_out.write("        double prediction = 0;\n\n");

		while(true)
		{
			int current_node_index = node_name.indexOf(current_node_name);
			NodeType current_type = node_type.get(current_node_index);
			int height = current_node_name.length() - current_node_name.replace("_", "").length() + 1; //number of "_" in the node name
			String tabs = String.join("", Collections.nCopies(height, "    ")); //sequence of tabs, each tab is represented by 4 spaces

			if(first_time)
			{
				if(current_node_name.endsWith("R"))
					java_out.write(tabs + "} else {\n");

				java_out.write(tabs + "    //" + current_node_name + "\n");

				if(current_type == NodeType.SPLIT) 	{
					double current_split_value = split_val.get(current_node_index);
					String current_split_attr = ainfo.idToName(split_attr_id.get(current_node_index)).replace("_","");

					java_out.write(tabs + "    if (" + current_split_attr + " <= " + current_split_value + ") {\n");
					current_node_name += "_L";
				} else {
					first_time = false;
					if(current_type == NodeType.CONST) {
						java_out.write(tabs + "    prediction = " + const_val[current_node_index] + ";\n");
					} else {
						//linear regression model in the leaf
						ArrayList<ArrayList<Double>> current_lr_coefs = lr_coefs.get(current_node_index);

						for(int lr_attr_index = 0; lr_attr_index < lr_attr_ids.get(current_node_index).size(); lr_attr_index++) {
						    String current_lr_attr = ainfo.idToName(lr_attr_ids.get(current_node_index).get(lr_attr_index)).replace("_","");
							double current_min = current_lr_coefs.get(lr_attr_index).get(polyDegree);
							double current_max = current_lr_coefs.get(lr_attr_index).get(polyDegree + 1);
							//double xcap = Math.max(0.0, Math.min(x, 5760.0));
							java_out.write(tabs + "    double " + current_lr_attr + "cap = Math.max(" + current_min + ", Math.min(" + current_lr_attr + ", " + current_max + "));\n");
						}
						//prediction = b0
						java_out.write("\n" + tabs + "    prediction = " + intercept_val[current_node_index]);
						for(int lr_attr_index = 0; lr_attr_index < lr_attr_ids.get(current_node_index).size(); lr_attr_index++) {
						    String current_lr_attr_cap = ainfo.idToName(lr_attr_ids.get(current_node_index).get(lr_attr_index)).replace("_","") + "cap";
							// + x1_cap *
						    java_out.write("\n" + tabs + "        + " + current_lr_attr_cap + " *\n" + tabs + "        ");
							for(int degree_index = 0; degree_index < polyDegree - 1; degree_index++)
								//(b11 + x1_cap * (b12 + x1_cap * ...
								java_out.write("(" + current_lr_coefs.get(lr_attr_index).get(degree_index) + " + " + current_lr_attr_cap + " *\n" + tabs + "        ");
							//b13)))
							java_out.write(current_lr_coefs.get(lr_attr_index).get(polyDegree - 1) + String.join("", Collections.nCopies(polyDegree - 1, ")")));
						}
						java_out.write(";\n");
					}
				}
			} else {
				if(current_node_name.endsWith("L")) {
					current_node_name = current_node_name.replaceAll("L$","R"); //replace last L with R
					first_time = true;
				} else if(current_node_name.endsWith("R")) {
					java_out.write(tabs + "}\n");
					current_node_name = current_node_name.replaceAll("_R$",""); //remove last "_R"
				} else {//Node_Root
					break;
				}
			}
		}
		java_out.close();
	}

	public double predict(String data_str) {
		String[] data = data_str.split("\t");
		if(data.length != ainfo.getColN())
		{ 
			System.err.println("The number of columns in the data does not match the number of attributes in the file.");
			System.exit(1);
		}
		int current_index = 0;
		String next_node = new String();

		while(true){
			String current_node = node_name.get(current_index);
			NodeType current_type = node_type.get(current_index);

			if(current_type == NodeType.SPLIT) {
				// identify the data point falls in which node (L or R) on this split.
				int current_col = ainfo.idToCol(split_attr_id.get(current_index));
				double current_val = Double.parseDouble(data[current_col]);
				double current_split = split_val.get(current_index);
				if(current_val <= current_split) {
					next_node = current_node + "_L";
				} else {
					next_node = current_node + "_R";
				}
				int next_index = node_name.indexOf(next_node);
				current_index = next_index;
			} else {
				// predict for this data point based on the const or model on the leaf it falls in.
				double val = 0;
				if(current_type == NodeType.CONST){
					// predict by the const on the leaf
					val = const_val[current_index];
				} else {
					// predict by the model on the leaf
					val = intercept_val[current_index]; // intercept
					ArrayList<Integer> current_lr_attr_ids = lr_attr_ids.get(current_index);
					ArrayList<ArrayList<Double>> current_lr_coefs = lr_coefs.get(current_index);
					for(int i_attr = 0; i_attr < current_lr_attr_ids.size(); i_attr++){
						double current_x = Double.parseDouble(data[ainfo.idToCol(current_lr_attr_ids.get(i_attr))]);
						double current_feature_min = current_lr_coefs.get(i_attr).get(polyDegree);
						double current_feature_max = current_lr_coefs.get(i_attr).get(polyDegree + 1);
						if(current_x < current_feature_min){
							current_x = current_feature_min;
						}
						if(current_x > current_feature_max){
							current_x = current_feature_max;
						}
						for(int i = 0; i < polyDegree; i++){
							val += current_lr_coefs.get(i_attr).get(i)*Math.pow(current_x, i + 1); // each attribute
						}
					}
				}

				return val;
			}
		}
	}
 	
	// XW. Largely equivalent to predict(String) but cache predictions in instances
	public double predict(Instance instance) {
		if (! instance.isIndexed()) {
			System.err.println("Please call indexLeaf(Instance, String[]) when loading rank lists");
			System.exit(1);
		}
		double[] data = instance.getValues();
		
		int currentIndex = instance.getNodeIndex();
		NodeType currentType = node_type.get(currentIndex);
		
		// Use the leaf node the instance falls in to compute a prediction
		double prediction = 0;
		if (currentType == NodeType.CONST) {
			prediction = const_val[currentIndex];
		} else {
			prediction = intercept_val[currentIndex];
			ArrayList<Integer> leafAttrIds = lr_attr_ids.get(currentIndex);
			ArrayList<ArrayList<Double>> leafCoefs = lr_coefs.get(currentIndex);
			for (int attIndex = 0; attIndex < leafAttrIds.size(); attIndex ++) {
				String attName = ainfo.idToName(leafAttrIds.get(attIndex));
				double value = data[instance.getAttId(attName)];
				value = truncate(currentIndex, attIndex, value);
				for (int j = 0; j < polyDegree; j ++) {
					// (j + 1)-th power of the attIndex-th attribute
					prediction += leafCoefs.get(attIndex).get(j) * Math.pow(value, j + 1);
				}
			}
		}
		
		// Cache the prediction for future incremental update
		instance.setPrediction(prediction); // Easily forgot
		return prediction;
	}
	
	// XW. This is used to speed up training by incrementally computing predictions
	public double predict(
			Instance instance, 
			int activeNode, 
			int activeParam,
			double paramDelta
			) {
		if (! instance.isIndexed()) {
			System.err.println("Please call indexLeaf(Instance, String[]) when loading rank lists");
			System.exit(1);
		}
		double[] data = instance.getValues();
		
		double prediction = instance.getPrediction();
		int currentIndex = instance.getNodeIndex();
		NodeType currentType = node_type.get(currentIndex);
		// Only need to update the instances that fall into the active leaf node
		if (currentIndex == activeNode) {
			if (currentType == NodeType.CONST) {
				System.err.println("Impossbile to sample CONST leaves from FirTree.getShuffledIdPairs");
				System.exit(1);
			}
			
			// Incremental update of the previous prediction of the instance
			
			// If oldPrediction = w_1*x_1 + ... + old_w_i*x_i and we change old_w_i to new_w_i
			// Then newPrediction = oldPrediction + (new_w_i - old_w_i)*x_i
			// Because newPrediction - oldPrediction = (new_w_i - old_w_i)*x_i
			
			///* Ablative Debug Start
			if (activeParam == INTERCEPT) {
				// Caused by updating intercept, equivalent to set x_i to 1
				prediction += paramDelta; // paramDelta = new_w_i - old_w_i
			} else {
				// Caused by updating coefficients
				int attIndex = getAttIndex(activeNode, activeParam);
				int polyIndex = getPolyIndex(activeNode, activeParam);
				ArrayList<Integer> leafAttrIds = lr_attr_ids.get(currentIndex);
			
				// This code snippet is copied from predict(Instance)
				String attName = ainfo.idToName(leafAttrIds.get(attIndex));
				double value = data[instance.getAttId(attName)];
				value = truncate(currentIndex, attIndex, value); // currentIndex == activeNode
				double paramValue = Math.pow(value, polyIndex + 1);
			
				// paramDelta = new_w_i - old_w_i
				// paramValue = (polyIndex + 1)-th power of the attIndex-th attribute
				prediction += paramDelta * paramValue;
			}
			//*/
			/*// This code snippet comes from predict(Instance)
			prediction = intercept_val[currentIndex];
			ArrayList<Integer> leafAttrIds = lr_attr_ids.get(currentIndex);
			ArrayList<ArrayList<Double>> leafCoefs = lr_coefs.get(currentIndex);
			for (int i = 0; i < leafAttrIds.size(); i ++) {
				String attName = ainfo.idToName(leafAttrIds.get(i));
				double x = data[instance.getAttId(attName)];
				double xMin = leafCoefs.get(i).get(polyDegree);
				double xMax = leafCoefs.get(i).get(polyDegree + 1);
				if (x < xMin)
					x = xMin;
				if (x > xMax)
					x = xMax;
				for (int j = 0; j < polyDegree; j ++) {
					// (j + 1)-th power of the i-th attribute
					prediction += leafCoefs.get(i).get(j) * Math.pow(x, j + 1);
				}
			}
			*/// Ablative Debug End
			
			instance.setPrediction(prediction); // Easily forgot
		}
		return prediction;
	}
	
	public double predict(
			Instance instance, 
			int activeNode, 
			int activeAtt,
			double delta,
			String type
			) {
		if (! instance.isIndexed()) {
			System.err.println("Please call indexLeaf(Instance, String[]) when loading rank lists");
			System.exit(1);
		}
		double[] data = instance.getValues();
		
		double prediction = instance.getPrediction();
		int currentIndex = instance.getNodeIndex();
		NodeType currentType = node_type.get(currentIndex);
		// Only need to update the instances that fall into the active leaf node
		if (currentIndex == activeNode) {
			if (currentType == NodeType.CONST) {
				System.err.println("Impossbile to sample CONST leaves from FirTree.getShuffledIdPairs");
				System.exit(1);
			}
			
			// Incremental update of the previous prediction of the instance
			
			/*//
			// Simplify the way to get feature value
			if (instance.getAttId(ainfo.idToName(lr_attr_ids.get(activeNode).get(activeAtt))) != activeAtt) {
				System.err.println("Oops");
				System.exit(1);
			}
			*///
			
			///* Ablative Debug Start
			double x = data[activeAtt];
			if (type.equals("min")) {
				double newMin = getMinValue(activeNode, activeAtt);
				double oldMin = newMin - delta;
				if (x <= oldMin && x <= newMin) {
					for (int j = 0; j < polyDegree; j ++) {
						double param = lr_coefs.get(activeNode).get(activeAtt).get(j);
						prediction += param * (Math.pow(newMin, j + 1) - Math.pow(oldMin, j + 1));
					}
				} else if (x > oldMin && x <= newMin) {
					for (int j = 0; j < polyDegree; j ++) {
						double param = lr_coefs.get(activeNode).get(activeAtt).get(j);
						prediction += param * (Math.pow(newMin, j + 1) - Math.pow(x, j + 1));
					}
				} else if (x > newMin && x <= oldMin) {
					for (int j = 0; j < polyDegree; j ++) {
						double param = lr_coefs.get(activeNode).get(activeAtt).get(j);
						prediction += param * (Math.pow(x, j + 1) - Math.pow(oldMin, j + 1));
					}
				} else {
					// Prediction does not change
				}
			} else if (type.equals("max")) {
				double newMax = getMaxValue(activeNode, activeAtt);
				double oldMax = newMax - delta;
				if (x > oldMax && x > newMax) {
					for (int j = 0; j < polyDegree; j ++) {
						double param = lr_coefs.get(activeNode).get(activeAtt).get(j);
						prediction += param * (Math.pow(newMax, j + 1) - Math.pow(oldMax, j + 1));
					}
				} else if (x <= oldMax && x > newMax) {
					for (int j = 0; j < polyDegree; j ++) {
						double param = lr_coefs.get(activeNode).get(activeAtt).get(j);
						prediction += param * (Math.pow(newMax, j + 1) - Math.pow(x, j + 1));
					}
				} else if (x <= newMax && x > oldMax) {
					for (int j = 0; j < polyDegree; j ++) {
						double param = lr_coefs.get(activeNode).get(activeAtt).get(j);
						prediction += param * (Math.pow(x, j + 1) - Math.pow(oldMax, j + 1));
					}
				} else {
					// Prediction does not change
				}
			} else {
				System.err.printf("Unknown type %s in FirTree.predict\n", type);
				System.exit(1);
			}
			//*/
			/*// This code snippet comes from predict(Instance)
			prediction = intercept_val[currentIndex];
			ArrayList<Integer> leafAttrIds = lr_attr_ids.get(currentIndex);
			ArrayList<ArrayList<Double>> leafCoefs = lr_coefs.get(currentIndex);
			for (int i = 0; i < leafAttrIds.size(); i ++) {
				String attName = ainfo.idToName(leafAttrIds.get(i));
				double x = data[instance.getAttId(attName)];
				double xMin = leafCoefs.get(i).get(polyDegree);
				double xMax = leafCoefs.get(i).get(polyDegree + 1);
				if (x < xMin)
					x = xMin;
				if (x > xMax)
					x = xMax;
				for (int j = 0; j < polyDegree; j ++) {
					// (j + 1)-th power of the i-th attribute
					prediction += leafCoefs.get(i).get(j) * Math.pow(x, j + 1);
				}
			}
			*/// Ablative Debug End
			
			instance.setPrediction(prediction); // Easily forgot
		}
		return prediction;
	}
	
	// XW. Return a list of names of all leaves, either MODEL or CONST
	public List<String> getAllLeaves() {
		List<String> allLeaves = new ArrayList<>();
		allLeaves.addAll(getRegressionLeaves());
		allLeaves.addAll(getConstLeaves());
		return allLeaves;
	}
	
	// XW. A leaf's coefficients are two dimensional first by attribute then by polynomial degree
	// XW. Map sampled one-dimensional parameter id to index of the attribute
	private int getAttIndex(int activeNode, int activeParam) {
		// Number of attributes used by the node (must be a leaf)
		int n = lr_attr_ids.get(activeNode).size();
		// 0, n, 2n, ... -> 0-th attribute
		// ...
		// n-1, 2n-1, 3n-1, ... -> last attribute
		return activeParam % n;
	}
	
	// XW. Map sampled one-dimensional parameter id to index of the polynomial degree
	private int getPolyIndex(int activeNode, int activeParam) {
		// Number of attributes used by the node (must be a leaf)
		int n = lr_attr_ids.get(activeNode).size();
		// 0, 1, ..., n-1 -> 0
		// n, n+1, ..., 2n-1 -> 1
		// ...
		return activeParam / n;
	}
	
	// XW. Decide which leaf node an instance falls in
	public int indexLeaf(Instance instance, String[] data) {
		if (data.length != ainfo.getColN()) {
			System.err.println("FirTree.indexLeaf: The number of columns in the data does not match the number of attributes in the file");
			System.exit(1);
		}
		
		int currentIndex = 0;
		String nextNode = new String();

		while(true){
			String currentNode = node_name.get(currentIndex);
			NodeType currentType = node_type.get(currentIndex);

			if(currentType == NodeType.SPLIT) {
				// Find which of the current node's children (L or R) the instance falls in
				int currentCol = ainfo.idToCol(split_attr_id.get(currentIndex));
				double currentVal = Double.parseDouble(data[currentCol]);
				double currentSplit = split_val.get(currentIndex);
				if(currentVal <= currentSplit) {
					nextNode = currentNode + "_L";
				} else {
					nextNode = currentNode + "_R";
				}
				int nextIndex = nodeIndexes.get(nextNode);
				currentIndex = nextIndex;
			} else {
				// Fall in a leaf node of type MODEL or CONST
				break;
			}
		}
		
		return currentIndex;
	}
	
	// XW
	public String getNodeName(int nodeIndex) {
		return node_name.get(nodeIndex);
	}
	
	// XW
	public List<IntPair> getParamIdPairs() {
		List<IntPair> idPairs = new ArrayList<>();
		
		List<String> modelLeaves = getRegressionLeaves();
		for (int i = 0; i < modelLeaves.size(); i ++) {
			int activeNode = nodeIndexes.get(modelLeaves.get(i));

			// Intercept is denoted by -1
			idPairs.add(new IntPair(activeNode, INTERCEPT));
			
			// Index of coefficients starts from 0
			int nCoef = lr_attr_ids.get(activeNode).size() * polyDegree;
			for (int activeParam = 0; activeParam < nCoef; activeParam ++) {
				idPairs.add(new IntPair(activeNode, activeParam));
			}
		}
		
		Collections.shuffle(idPairs, new Random(666));
		/*//
		Collections.shuffle(idPairs);
		*///
		
		return idPairs;
	}
	
	public List<IntPair> getBoundIdPairs() {
		List<IntPair> idPairs = new ArrayList<>();
		
		List<String> modelLeaves = getRegressionLeaves();
		for (int i = 0; i < modelLeaves.size(); i ++) {
			int activeNode = nodeIndexes.get(modelLeaves.get(i));
			for (int activeAtt = 0; activeAtt < lr_coefs.get(activeNode).size(); activeAtt ++) {
				idPairs.add(new IntPair(activeNode, activeAtt));
			}
		}
		
		Collections.shuffle(idPairs, new Random(666));
		
		return idPairs;
	}
	
	// XW
	public double getParamValue(int activeNode, int activeParam) {
		if (activeParam == INTERCEPT) {
			return intercept_val[activeNode];
		} else {
			int attIndex = getAttIndex(activeNode, activeParam);
			int polyIndex = getPolyIndex(activeNode, activeParam);
			return lr_coefs.get(activeNode).get(attIndex).get(polyIndex);
		}
	}
	
	// XW
	public void setParamValue(int activeNode, int activeParam, double paramDelta) {
		if (activeParam == INTERCEPT) {
			intercept_val[activeNode] += paramDelta;
		} else {
			int attIndex = getAttIndex(activeNode, activeParam);
			int polyIndex = getPolyIndex(activeNode, activeParam);
			lr_coefs.get(activeNode).get(attIndex).set(
					polyIndex, lr_coefs.get(activeNode).get(attIndex).get(polyIndex) + paramDelta);
		}
	}
	
	// XW
	public String getParamName(int activeNode, int activeParam) {
		if (activeParam == INTERCEPT) {
			return "intercept";
		} else {
			int attIndex = getAttIndex(activeNode, activeParam);
			int polyIndex = getPolyIndex(activeNode, activeParam);
			int attrId = lr_attr_ids.get(activeNode).get(attIndex);
			String paramName = ainfo.idToName(attrId);
			paramName = String.format("%s^%d", paramName, polyIndex + 1).replace("_", "-");
			return paramName;
		}
	}
	
	// XW
	public void save(int nIter) throws Exception {
		List<String> modelLeaves = getRegressionLeaves();
		for (String leafName : modelLeaves) {
			int nodeIndex = nodeIndexes.get(leafName);
			
			String paramPath = getParamPath(nodeIndex);
			if ((nIter >= 0) && (! paramPath.endsWith("_const.txt"))) {
				paramPath = paramPath.replace(".txt", "_n" + nIter + ".txt");
			}
			BufferedWriter bw = new BufferedWriter(new FileWriter(paramPath));

			ArrayList<Integer> leafAttrIds = lr_attr_ids.get(nodeIndex);
			ArrayList<ArrayList<Double>> leafCoefs = lr_coefs.get(nodeIndex);

			bw.write("intercept\t" + intercept_val[nodeIndex] + "\n");
			for (int i = 0; i < leafAttrIds.size(); i ++) {
				int attrId = leafAttrIds.get(i);
				bw.write(ainfo.idToName(attrId) + "\t");
				for (int j = 0; j < polyDegree; j ++) {
					bw.write(leafCoefs.get(i).get(j) + "\t" );
				}
				
				if (leafCoefs.get(i).size() == polyDegree + 2) {
					// These min and max values are varied (used for training and testing)
					bw.write(leafCoefs.get(i).get(polyDegree) + "\t"); // Min
					bw.write(leafCoefs.get(i).get(polyDegree + 1) + "\n"); // Max
				} else if (leafCoefs.get(i).size() == polyDegree + 4) {
					// These min and max values are varied (used for training and testing)
					bw.write(leafCoefs.get(i).get(polyDegree) + "\t"); // Min
					bw.write(leafCoefs.get(i).get(polyDegree + 1) + "\t"); // Max
					// These min and max values are fixed (determined by a training set)
					bw.write(leafCoefs.get(i).get(polyDegree + 2) + "\t"); // Min
					bw.write(leafCoefs.get(i).get(polyDegree + 3) + "\n"); // Max
				} else {
					System.err.printf("%d coefficients (including min/max) with degree %d\n", 
							leafCoefs.get(i).size(), polyDegree);
					System.exit(1);
				}
			}
			bw.flush();
			bw.close();
		}
	}
	
	public String getParamPath(String nodeName) {
		int nodeIndex = nodeIndexes.get(nodeName);
		return getParamPath(nodeIndex);
	}
	
	private String getParamPath(int nodeIndex) {
		String paramPath = dir + "/Node_" + node_name.get(nodeIndex) + "/" + modelPrefix;
		if (node_type.get(nodeIndex) == NodeType.MODEL) {
			paramPath += ".txt";
		} else if (node_type.get(nodeIndex) == NodeType.CONST) {
			paramPath += "_const.txt";
		} else {
			System.err.printf("Cannot save parameters of %s\n", node_name.get(nodeIndex));
			System.exit(1);
		}
		return paramPath;
	}
	
	public double truncate(int nodeIndex, int attIndex, double value) {
		if (lr_coefs.size() > 0) {
			double min = lr_coefs.get(nodeIndex).get(attIndex).get(polyDegree);
			double max = lr_coefs.get(nodeIndex).get(attIndex).get(polyDegree + 1);
			return Math.min(Math.max(value, min), max);
		} else {
			return value;
		}
	}
	
	public void bakMinmax() {
		for (int nodeIndex = 0; nodeIndex < lr_coefs.size(); nodeIndex ++) {
			for (int attIndex = 0; attIndex < lr_coefs.get(nodeIndex).size(); attIndex ++) {
				int size = lr_coefs.get(nodeIndex).get(attIndex).size();
				// Backup min/max only when we haven't done it yet
				if (size == polyDegree + 2) {
					double min = lr_coefs.get(nodeIndex).get(attIndex).get(size - 2);
					double max = lr_coefs.get(nodeIndex).get(attIndex).get(size - 1);
					lr_coefs.get(nodeIndex).get(attIndex).add(min);
					lr_coefs.get(nodeIndex).get(attIndex).add(max);
				}
			}
			/*//
			System.out.printf("%d %d\n", nodeIndex, lr_coefs.get(nodeIndex).size());
			for (int attIndex = 0; attIndex < lr_coefs.get(nodeIndex).size(); attIndex ++) {
				System.out.printf("  %s", ainfo.idToName(lr_attr_ids.get(nodeIndex).get(attIndex)));
				for (int i = 0; i < lr_coefs.get(nodeIndex).get(attIndex).size(); i ++) {
					System.out.printf("  %s", lr_coefs.get(nodeIndex).get(attIndex).get(i));
				}
				System.out.println();
			}
			*///
		}
		
	}
	
	Map<Integer, Set<Integer>> attIdToValSet = new HashMap<>();
	public void addAttIdToValSet(List<Integer> attIdList, String[] data) {
		for (Integer attId : attIdList) {
			if (! attIdToValSet.containsKey(attId)) {
				attIdToValSet.put(attId, new HashSet<>());
			}
			attIdToValSet.get(attId).add(Integer.parseInt(data[ainfo.idToCol(attId)]));
		}
	}
	
	Map<Integer, List<Integer>> attIdToValList = new HashMap<>();
	public void addAttIdToValList() {
		for (Integer attId : attIdToValSet.keySet()) {
			List<Integer> valList = new ArrayList<>(attIdToValSet.get(attId));
			Collections.sort(valList);
			attIdToValList.put(attId, valList);
		}
		
		for (Integer attId : attIdToValList.keySet()) {
			List<Integer> valList = attIdToValList.get(attId);
			int min = Collections.min(valList);
			int max = Collections.max(valList);
			int interval = (max - min) / (valList.size() - 1);
			timeStamp(String.format("%s: %d distinct values in [%d, %d] average interval %d",
					ainfo.idToName(attId), valList.size(), min, max, interval));
		}
	}
	
	public double getMinValue(int nodeIndex, int attIndex) {
		return lr_coefs.get(nodeIndex).get(attIndex).get(polyDegree);
	}
	
	public double getMaxValue(int nodeIndex, int attIndex) {
		return lr_coefs.get(nodeIndex).get(attIndex).get(polyDegree + 1);
	}
	
	public void setMinValue(int activeNode, int activeAtt, double minDelta) {
		int polyIndex = polyDegree;
		lr_coefs.get(activeNode).get(activeAtt).set(
				polyIndex, lr_coefs.get(activeNode).get(activeAtt).get(polyIndex) + minDelta);
	}
	
	public void setMaxValue(int activeNode, int activeAtt, double maxDelta) {
		int polyIndex = polyDegree + 1;
		lr_coefs.get(activeNode).get(activeAtt).set(
				polyIndex, lr_coefs.get(activeNode).get(activeAtt).get(polyIndex) + maxDelta);
	}
	
	public int getMinIndex(int nodeIndex, int attIndex) {
		double min = getMinValue(nodeIndex, attIndex);
		int attId = lr_attr_ids.get(nodeIndex).get(attIndex);
		int origMinIndex = 0;
		for (; origMinIndex < attIdToValList.get(attId).size(); origMinIndex ++) {
			if (attIdToValList.get(attId).get(origMinIndex) >= min) {
				break;
			}
		}
		if (attIdToValList.get(attId).get(origMinIndex) == min) {
			return origMinIndex;
		} else {
			return Math.max(0, origMinIndex - 1);
		}
	}
	
	public int getMaxIndex(int nodeIndex, int attIndex) {
		double max = getMaxValue(nodeIndex, attIndex);
		int attId = lr_attr_ids.get(nodeIndex).get(attIndex);
		int origMaxIndex = attIdToValList.get(attId).size() - 1;
		for (; origMaxIndex > -1; origMaxIndex --) {
			if (attIdToValList.get(attId).get(origMaxIndex) <= max) {
				break;
			}
		}
		if (attIdToValList.get(attId).get(origMaxIndex) == max) {
			return origMaxIndex;
		} else {
			return Math.min(attIdToValList.get(attId).size() - 1, origMaxIndex + 1);
		}
	}
	
	protected static void timeStamp(String msg){
		Date tmpDate = new Date();
		System.out.println("TIMESTAMP >>>> ".concat(tmpDate.toString()).concat(": ").concat(msg));
	}
}

