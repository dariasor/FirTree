/*
 * This code loads the original FirTree outputs (a tree that splits on top interactions and returns key features on each leaf)
 * and returns for each leaf, either a regression model with transformed features (no iterations found) or a constant (not enough data or no improvements in candidate splits).
 */

package firtree;

import java.io.*;
import java.util.*;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.io.AttrInfo;
import mltk.core.io.AttributesReader;
import smile.regression.*;

public class RegressionOnLeaves {

	static class Options {
		@Argument(name = "-d", description = "model directory", required = true)
		String dir = ""; //path up to FirTree/
		
		@Argument(name = "-r", description = "attribute file", required = true)
		String attPath = "";
		
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

		// load attrFile.  This helps identify corresponding column index of selected attributes 

		timeStamp("Load attrFile");
		AttrInfo ainfo_core = AttributesReader.read(opts.attPath);

		// Identify tree leaves that will (1) contain LR models, (2) predict constants; 

		timeStamp("Identify tree leaves");

		Vector<String> leavesConst = new Vector<String>();
		Vector<String> leavesModel = new Vector<String>();

		BufferedReader treelog = new BufferedReader(new FileReader(opts.dir + "/treelog.txt"), 65535);
		String line_tree = treelog.readLine();
		String current_node = new String();

		while(line_tree != null){
			if(line_tree.matches("Root(.*)")){
				current_node = line_tree;
			}
			if(line_tree.matches("Not enough data.") || line_tree.matches("Const")){
				leavesConst.add(current_node);
			}
			if(line_tree.matches("No (.*)d.")){
				leavesModel.add(current_node);
			}
			if(line_tree.matches("Best feature:(.*)")){
				line_tree = treelog.readLine();
			}
			line_tree = treelog.readLine();
		}
		treelog.close();


		// Train model on each leafModel
		timeStamp("-------------- Train model on each leafModel --------------");

		for(int i_leaf = 0; i_leaf < leavesModel.size(); i_leaf++){

			String dataNodePath = opts.dir + "/Node_" + leavesModel.get(i_leaf);
			System.out.println("------------Processing leaf "+ leavesModel.get(i_leaf) + "------------");

			long start_load = System.currentTimeMillis();
			timeStamp("Scan data");

			AttrInfo ainfo_leaf = AttributesReader.read(dataNodePath + "/fir.fs.fs.attr");

			// read the data, save labels and values of selected features
			BufferedReader br_dta = new BufferedReader(new FileReader(dataNodePath + "/fir.dta"));
			
			int col_num = ainfo_leaf.attributes.size(); //col refers to the columns in the matrix, not in the data file
			List<List<Double>> xMat_arraylist = new ArrayList<List<Double>>(); //dynamic memory for temp data storage - features
			ArrayList<Double> y_double_arraylist = new ArrayList<Double>(); //dynamic memory for temp data storage - labels

			for(String line = br_dta.readLine(); line != null; line = br_dta.readLine()){
				String[] data = line.split("\t+");
				y_double_arraylist.add(Double.parseDouble(data[ainfo_core.getClsCol()]));
				ArrayList<Double> current_selected_attr = new ArrayList<Double>();
				for(int j = 0; j < col_num; j++){
					current_selected_attr.add(Double.parseDouble(data[ainfo_leaf.attributes.get(j).getColumn()]));
				}
				xMat_arraylist.add(current_selected_attr); 
			}
			br_dta.close();

			long end_load = System.currentTimeMillis();

			int row_num = y_double_arraylist.size(); // number of data points
			System.out.println(row_num);

			//copy the data into regular arrays, as required for regression model input
			double[] y_double = new double[row_num];
			double[][] xMat = new double[row_num][col_num];

			for(int i = 0; i < row_num; i++){
				y_double[i] = y_double_arraylist.get(i);
				for(int j = 0; j < col_num; j++){
					xMat[i][j] = xMat_arraylist.get(i).get(j);
				}
			}

			// Train OLS with polynomial terms
			timeStamp("Training OLS with transformed features");
			double[][] xMat_trans = new double[xMat.length][xMat[0].length * opts.poly_degree];
			for(int i = 0; i < xMat.length; i++){
				for(int j = 0; j < xMat[0].length; j++){
					for(int i_trans = 0; i_trans < opts.poly_degree; i_trans++){
						xMat_trans[i][j * opts.poly_degree + i_trans] = Math.pow(xMat[i][j], i_trans + 1);
					}
				}
			}

			OLS ols_trans = new OLS(xMat_trans, y_double); 

			double[] ols_trans_coef = ols_trans.coefficients();
			double ols_trans_intercept = ols_trans.intercept();

			// get the range of each selected feature for thresholding
			ArrayList<ArrayList<Double>> attr_range = new ArrayList<ArrayList<Double>>();
			for(int j = 0; j < col_num; j++){
				ArrayList<Double> current_attr_range = new ArrayList<Double>();
				double current_attr_min = Double.POSITIVE_INFINITY; 
				double current_attr_max = Double.NEGATIVE_INFINITY;
				for(int i = 0; i < row_num; i++){
					if(xMat[i][j] < current_attr_min){
						current_attr_min = xMat[i][j];
					}
					if(xMat[i][j] > current_attr_max){
						current_attr_max = xMat[i][j];
					}
				}
				current_attr_range.add(current_attr_min);
				current_attr_range.add(current_attr_max);
				attr_range.add(current_attr_range);
			}

			timeStamp("Saving the model");

			BufferedWriter modelTrans_out = new BufferedWriter(new FileWriter(dataNodePath + "/model_polydegree_" + opts.poly_degree + ".txt"));

			modelTrans_out.write("intercept\t" + ols_trans_intercept + "\n");

			for (int i_attr = 0; i_attr < col_num; i_attr++) {
				modelTrans_out.write(ainfo_leaf.idToName(i_attr) + "\t");
				for(int i_poly = 0; i_poly < opts.poly_degree; i_poly++){
					modelTrans_out.write(ols_trans_coef[i_attr * opts.poly_degree + i_poly] + "\t" );
				}
				modelTrans_out.write(attr_range.get(i_attr).get(0) + "\t" + attr_range.get(i_attr).get(1) + "\n");
			}
			modelTrans_out.flush();
			modelTrans_out.close();

			long end_train = System.currentTimeMillis();

			System.out.println("Finished training OLS on this node in " + (end_train - start_load) / 1000.0 + " (s).");
			System.out.println("Without loading data, the model training step takes " + (end_train - end_load) / 1000.0 + " (s).");
		}

		// Get constant for each leafConst;

		timeStamp("-------------- Get constant for each leafConst ---------------");

		for(int i_const = 0; i_const < leavesConst.size(); i_const++){

			String dataNodePath = opts.dir + "/Node_" + leavesConst.get(i_const);
			System.out.println("------------Processing leaf " + leavesConst.get(i_const) + "------------");

			// read dta file, only need to read the target column

			timeStamp("Load dta data");

			BufferedReader br_dta = new BufferedReader(new FileReader(dataNodePath + "/fir.dta"));
			ArrayList<Double> y_double_arraylist = new ArrayList<Double>();
			double y_sum = 0;
			
			for(String line = br_dta.readLine(); line != null; line = br_dta.readLine()){
				String[] data = line.split("\t+");
				double y_current = Double.parseDouble(data[ainfo_core.getClsCol()]);
				y_double_arraylist.add(y_current);
				y_sum += y_current;				
			}
			br_dta.close();

			double y_mean = y_sum / y_double_arraylist.size();
			System.out.println(y_mean);

			timeStamp("Saving the const");

			BufferedWriter const_out = new BufferedWriter(new FileWriter(dataNodePath + "/model_const.txt"));
			const_out.write("Const: " + y_mean);
			const_out.flush();
			const_out.close();
		}

		long end = System.currentTimeMillis();
		System.out.println("Finished all in " + (end - start) / 1000.0 + " (s).");
	}

	protected static void timeStamp(String msg){
		Date tmpDate = new Date();
		System.out.println("TIMESTAMP >>>> ".concat(tmpDate.toString()).concat(": ").concat(msg));
	}
}
