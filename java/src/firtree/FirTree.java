package firtree;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;

import mltk.core.io.AttrInfo;

public class FirTree {

	private enum NodeType { SPLIT, MODEL, CONST }

	private AttrInfo ainfo;

	private ArrayList<String> node_name;
	private ArrayList<NodeType> node_type;
	private ArrayList<Integer> split_attr_id;
	private ArrayList<Double> split_val;
	private int nodeN;

	private int poly_degree; //degree of polynomials in the leaf models. 0 means that the leaf models are not yet trained.
	private double[] const_val;
	private double[] intercept_val;
	private ArrayList<ArrayList<Integer>> lr_attr_ids;
	private ArrayList<ArrayList<ArrayList<Double>>> lr_coefs;

	public FirTree(AttrInfo ainfo_in, String dir, int poly_degree_in) throws Exception {

		// Load treeforPred
		node_name = new ArrayList<String>();
		node_type = new ArrayList<NodeType>();
		split_attr_id = new ArrayList<Integer>();
		split_val = new ArrayList<Double>();
		poly_degree = poly_degree_in;
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
		
		if(poly_degree > 0) {
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
					String lr_file_name = dir + "/Node_" + node_name.get(nodeNo) + "/model_polydegree_" + poly_degree + ".txt";
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
						for(int i = 0; i < poly_degree + 2; i++){
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
						double current_feature_min = current_lr_coefs.get(i_attr).get(poly_degree);
						double current_feature_max = current_lr_coefs.get(i_attr).get(poly_degree + 1);
						if(current_x < current_feature_min){
							current_x = current_feature_min;
						}
						if(current_x > current_feature_max){
							current_x = current_feature_max;
						}
						for(int i = 0; i < poly_degree; i++){
							val += current_lr_coefs.get(i_attr).get(i)*Math.pow(current_x, i + 1); // each attribute
						}
					}
				}

				return val;
			}
		}
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
							double current_min = current_lr_coefs.get(lr_attr_index).get(poly_degree);
							double current_max = current_lr_coefs.get(lr_attr_index).get(poly_degree + 1);
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
							for(int degree_index = 0; degree_index < poly_degree - 1; degree_index++)
								//(b11 + x1_cap * (b12 + x1_cap * ...
								cpp_out.write("(" + current_lr_coefs.get(lr_attr_index).get(degree_index) + " + " + current_lr_attr_cap + " *\n" + tabs + "        ");
							//b13)))
							cpp_out.write(current_lr_coefs.get(lr_attr_index).get(poly_degree - 1) + String.join("", Collections.nCopies(poly_degree - 1, ")")));
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
							double current_min = current_lr_coefs.get(lr_attr_index).get(poly_degree);
							double current_max = current_lr_coefs.get(lr_attr_index).get(poly_degree + 1);
							//double xcap = Math.max(0.0, Math.min(x, 5760.0));
							java_out.write(tabs + "    double " + current_lr_attr + "cap = Math.max(" + current_min + ", Math.min(" + current_lr_attr + ", " + current_max + "));\n");
						}
						//prediction = b0
						java_out.write("\n" + tabs + "    prediction = " + intercept_val[current_node_index]);
						for(int lr_attr_index = 0; lr_attr_index < lr_attr_ids.get(current_node_index).size(); lr_attr_index++) {
						    String current_lr_attr_cap = ainfo.idToName(lr_attr_ids.get(current_node_index).get(lr_attr_index)).replace("_","") + "cap";
							// + x1_cap *
						    java_out.write("\n" + tabs + "        + " + current_lr_attr_cap + " *\n" + tabs + "        ");
							for(int degree_index = 0; degree_index < poly_degree - 1; degree_index++)
								//(b11 + x1_cap * (b12 + x1_cap * ...
								java_out.write("(" + current_lr_coefs.get(lr_attr_index).get(degree_index) + " + " + current_lr_attr_cap + " *\n" + tabs + "        ");
							//b13)))
							java_out.write(current_lr_coefs.get(lr_attr_index).get(poly_degree - 1) + String.join("", Collections.nCopies(poly_degree - 1, ")")));
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
}

