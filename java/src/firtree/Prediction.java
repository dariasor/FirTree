/*
 * This code loads the result from RegressionFirTree (a regression model with transformed features or a constant on each leaf)
 * and predicts using these models and constants, based on which leaves the data points fall in.
 */

package firtree;

import java.io.*;
import java.util.*;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.io.AttrInfo;
import mltk.core.io.AttributesReader;

public class Prediction {
	
	static class Options {
		@Argument(name = "-d", description = "model directory", required = true)
		String dir = ""; //usually path up to "FirTree" inclusive
		
		@Argument(name = "-r", description = "attribute file", required = true)
		String attPath = "";
		
		@Argument(name = "-o", description = "output file", required = true)
		String outputPath = ""; 
		
		@Argument(name = "-test", description = "test set file", required = true)
		String testPath = "";		
		
		@Argument(name = "-y", description = "polynomial degree")
		int poly_degree = 2;

	}

	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(RegressionOnLeaves.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		
		long start = System.currentTimeMillis();

		timeStamp("Load attrFile");
		AttrInfo ainfo = AttributesReader.read(opts.attPath);
		
		// Load treeforPred
		ArrayList<String> node_name = new ArrayList<String>();
		ArrayList<String> node_type = new ArrayList<String>();
		ArrayList<Integer> attr_col = new ArrayList<Integer>();
		ArrayList<Double> split_val = new ArrayList<Double>();
		
		BufferedReader treelog = new BufferedReader(new FileReader(opts.dir + "/treelog.txt"), 65535);
		String line_tree = treelog.readLine();
		
		while(line_tree != null){
			if(line_tree.matches("Root(.*)")){
				node_name.add(line_tree.trim());
			}
			if(line_tree.matches("Not enough data.") || line_tree.matches("Const")){
				node_type.add("const");
				attr_col.add(-1);
				split_val.add(Double.POSITIVE_INFINITY);				
			}
			if(line_tree.matches("No (.*)d.")){
				node_type.add("model");
				attr_col.add(-1);
				split_val.add(Double.POSITIVE_INFINITY);				
			}
			if(line_tree.matches("Best feature:(.*)")){
				node_type.add("split");
				String current_attr = line_tree.split(" ")[2];
				attr_col.add(ainfo.nameToCol.get(current_attr));
				line_tree = treelog.readLine();
				split_val.add(Double.parseDouble(line_tree.split(" ")[2]));
				
			}
			line_tree = treelog.readLine();
		}
		treelog.close();

		// Load models and constants on FirTree leaves
		timeStamp("Load models");
		int nodeN = node_name.size();
		double[] const_val = new double[nodeN];
		double[] intercept_val = new double[nodeN];
		ArrayList<ArrayList<Integer>> model_col = new ArrayList<ArrayList<Integer>>(nodeN);
		ArrayList<ArrayList<ArrayList<Double>>> model_coef = new ArrayList<ArrayList<ArrayList<Double>>>(nodeN);

		for(int nodeNo = 0; nodeNo < node_name.size(); nodeNo++){

			ArrayList<Integer> tempnode_model_col = new ArrayList<Integer>();
			ArrayList<ArrayList<Double>> tempnode_model_coef = new ArrayList<ArrayList<Double>>();

			if(node_type.get(nodeNo).matches("model")) {
				// there is a model on the leaf
				BufferedReader model = new BufferedReader(new FileReader(opts.dir + "/Node_" + node_name.get(nodeNo) + "/model_polydegree_" + opts.poly_degree + ".txt"));
				String line = model.readLine();
				intercept_val[nodeNo] = Double.parseDouble(line.split("\t")[2]);
				line = model.readLine();
				while(line!=null){
					String[] data_model = line.split("\t");
					tempnode_model_col.add(Integer.parseInt(data_model[0]));
					ArrayList<Double> tempnode_tempattr_model_coef = new ArrayList<Double>();
					for(int i = 0; i < opts.poly_degree + 2; i++){
						// the last two items are (min and max) range of the attribute
						// the previous items are coefficients of the corresponding polynomial terms of the attribute
						tempnode_tempattr_model_coef.add(Double.parseDouble(data_model[i + 2]));
					}
					tempnode_model_coef.add(tempnode_tempattr_model_coef);
					line = model.readLine();
				}
				model_col.add(tempnode_model_col);
				model_coef.add(tempnode_model_coef);
				model.close();
			} else {
				// there is no model on the node
				model_col.add(tempnode_model_col);
				model_coef.add(tempnode_model_coef);
				if(node_type.get(nodeNo).matches("const")) {
					// there is a const on the node
					BufferedReader constRead = new BufferedReader(new FileReader(opts.dir + "/Node_" + node_name.get(nodeNo) + "/model_const.txt"));
					String line = constRead.readLine();
					const_val[nodeNo] = Double.parseDouble(line.split(": ")[1]); 
					constRead.close();
				} 
			}
		}

		// Load test data and predict
		timeStamp("Load data and generate predictions");
		BufferedReader testData = new BufferedReader(new FileReader(opts.testPath), 65535);
		BufferedWriter pred_fir_out = new BufferedWriter(new FileWriter(opts.outputPath));

		String line_test = testData.readLine();
		while(line_test != null) {
			String[] data = line_test.split("\t");
			int tree_index = 0;
			String next_node = new String();
			
			while(true){
				String current_node = node_name.get(tree_index);
				String current_type = node_type.get(tree_index);

				if(current_type.matches("split")) {
					// identify the data point falls in which node (L or R) on this split.
					int current_col = attr_col.get(tree_index);
					Double current_val = Double.parseDouble(data[current_col]);
					Double current_split = split_val.get(tree_index);
					if(current_val <= current_split) {
						next_node = current_node + "_L";
					} else {
						next_node = current_node + "_R";
					} 
					int next_index = node_name.indexOf(next_node);
					tree_index = next_index;
				} else {
					// predict for this data point based on the const or model on the leaf it falls in.
					double val = 0;
					if(current_type.matches("const")){
						// predict by the const on the leaf
						val = const_val[tree_index];
					} else {
						// predict by the model on the leaf
						val = intercept_val[tree_index]; // intercept
						ArrayList<Integer> current_col = model_col.get(tree_index);
						ArrayList<ArrayList<Double>> current_coef = model_coef.get(tree_index);
						for(int i_attr = 0; i_attr < current_col.size(); i_attr++){
							double current_x = Double.parseDouble(data[current_col.get(i_attr)]);
							double current_feature_min = current_coef.get(i_attr).get(opts.poly_degree);
							double current_feature_max = current_coef.get(i_attr).get(opts.poly_degree + 1);
							if(current_x < current_feature_min){
								current_x = current_feature_min;
							}
							if(current_x > current_feature_max){
								current_x = current_feature_max;
							}

							for(int i = 0; i < opts.poly_degree; i++){
								val += current_coef.get(i_attr).get(i)*Math.pow(current_x, i + 1); // each attribute
							}
						}
					}

					pred_fir_out.write(val + "\n");
					break;
				}
			}
			line_test = testData.readLine();
		}
		testData.close();
		pred_fir_out.flush();
		pred_fir_out.close();

		long end = System.currentTimeMillis();
		System.out.println("Finished all in " + (end - start) / 1000.0 + " (s).");
	}
	
	protected static void timeStamp(String msg){
		Date tmpDate = new Date();
		System.out.println("TIMESTAMP >>>> ".concat(tmpDate.toString()).concat(": ").concat(msg));
	}
}
