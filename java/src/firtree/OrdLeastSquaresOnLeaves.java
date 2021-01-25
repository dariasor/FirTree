/*
 * This code loads the original FirTree outputs (a tree that splits on top interactions and returns key features on each leaf)
 * and returns for each leaf, either a regression model with transformed features (no iterations found) or a constant (not enough data or no improvements in candidate splits).
 */

package firtree;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.io.AttrInfo;
import mltk.core.io.AttributesReader;
import smile.regression.*;

public class OrdLeastSquaresOnLeaves {

	static class Options {
//		@Argument(name="-d", description="FirTree directory output by InteractionTreeLearnerGAMMC", required=true)
//		String dir = ""; // Usually path up to "FirTree" inclusive

		@Argument(name="-l", description="(cropped) treelog.txt which specifies a tree structure", required=true)
		String logPath = "";

		@Argument(name = "-r", description = "attribute file", required = true)
		String attPath = "";

		@Argument(name = "-y", description = "polynomial degree")
		int polyDegree = 2;
		
		// The group id is used for subsampling if necessary
		@Argument(name = "-g", description = "name of the attribute with the group id")
		String group = "";
		
		@Argument(name = "-m", description = "Prefix of name of output parameter files (default: ols)")
		String modelPrefix = "ols_y2";
		
		@Argument(name = "-o", description = "Override the results of OLS")
		int override = 1;
	}


	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(OrdLeastSquaresOnLeaves.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}

		long start = System.currentTimeMillis();

		//Load attrFile and tree structure.
		timeStamp("Load attrFile");
		AttrInfo ainfo = AttributesReader.read(opts.attPath);
		FirTree model = new FirTree(ainfo, opts.logPath, opts.polyDegree, opts.modelPrefix, opts.override);
		ArrayList<String> leavesModel = model.getRegressionLeaves();
		ArrayList<String> leavesConst = model.getConstLeaves();
		
		for(int i_leaf = 0; i_leaf < leavesModel.size(); i_leaf++){
			String leafName = leavesModel.get(i_leaf);
			int nodeIndex = model.nodeIndexes.get(leafName);
			if (model.lr_attr_ids.size() > 0) {
				for (int j = 0; j < model.lr_attr_ids.get(nodeIndex).size(); j ++) {
					int id = model.lr_attr_ids.get(nodeIndex).get(j);
					if (id != model.nodeAttIdList.get(nodeIndex).get(j)) {
						System.err.println("Variables lr_attr_ids and nodeAttIdList are inconsistent");
						System.exit(1);
					}
				}				
			}
		}
		
		// Train model on each regression leaf
		timeStamp("-------------- Train model on each regression leaf --------------");

		for(int i_leaf = 0; i_leaf < leavesModel.size(); i_leaf++){
			String leafName = leavesModel.get(i_leaf);
			
			int nodeIndex = model.nodeIndexes.get(leafName);
			
			timeStamp("------------Processing leaf "+ leafName + "------------");

			long start_load = System.currentTimeMillis();
			timeStamp("Scan data");

			String dataPath = getNodeDir(model.dir, leafName).toString() + "/fir.dta";
			// read the data, save labels and values of selected features
			BufferedReader br_dta = new BufferedReader(new FileReader(dataPath));

			
			int col_num = model.nodeAttIdList.get(nodeIndex).size(); //col refers to the columns in the matrix, not in the data file

			boolean crash = checkCrash(col_num, ainfo, opts, dataPath);
			Set<String> groupIdSet = null;
			if (crash) {
				groupIdSet = subsample(col_num, ainfo, opts, dataPath);
			} else {
				timeStamp("No need to subsample data points");
			}
			
			List<List<Double>> xMat_arraylist = new ArrayList<List<Double>>(); //dynamic memory for temp data storage - features
			ArrayList<Double> y_double_arraylist = new ArrayList<Double>(); //dynamic memory for temp data storage - labels

			for(String line = br_dta.readLine(); line != null; line = br_dta.readLine()) {
				String[] data = line.split("\t+");

				if (crash) {
					// Skip the group ids that are not subsampled, i.e., not included in the set
					String groupId = data[ainfo.nameToCol.get(opts.group)];
					if (! groupIdSet.contains(groupId))
						continue;
				}
				
				y_double_arraylist.add(Double.parseDouble(data[ainfo.getClsCol()]));
				ArrayList<Double> current_selected_attr = new ArrayList<Double>();
				for(int attIndex = 0; attIndex < col_num; attIndex ++){
					int id = model.nodeAttIdList.get(nodeIndex).get(attIndex);
					double value = Double.parseDouble(data[ainfo.idToCol(id)]);
					current_selected_attr.add(model.truncate(nodeIndex, attIndex, value));
				}
				xMat_arraylist.add(current_selected_attr);
			}
			br_dta.close();

			long end_load = System.currentTimeMillis();

			int row_num = y_double_arraylist.size(); // number of data points
			timeStamp("Number of data points: "  + row_num);

			//copy the data into regular arrays, as required for regression model input
			double[] y_double = new double[row_num];
			double[][] xMat = new double[row_num][col_num];

			for(int i = 0; i < row_num; i++){
				y_double[i] = y_double_arraylist.get(i);
				for(int j = 0; j < col_num; j++){
					xMat[i][j] = xMat_arraylist.get(i).get(j);
				}
			}
			
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

			// Train OLS with polynomial terms
			timeStamp("Training OLS with transformed features");
			double[][] xMat_trans = new double[xMat.length][xMat[0].length * opts.polyDegree];
			for(int i = 0; i < xMat.length; i++){
				for(int j = 0; j < xMat[0].length; j++){
					for(int i_trans = 0; i_trans < opts.polyDegree; i_trans++){
						xMat_trans[i][j * opts.polyDegree + i_trans] = Math.pow(xMat[i][j], i_trans + 1);
					}
				}
			}

			// Setting the last parameter to True to use SVD decomposition as part of the regression
			OLS ols_trans = new OLS(xMat_trans, y_double, true);

			double[] ols_trans_coef = ols_trans.coefficients();
			double ols_trans_intercept = ols_trans.intercept();

			timeStamp("Save the model");

			String paramPath = model.getParamPath(leafName);
			BufferedWriter modelTrans_out = new BufferedWriter(new FileWriter(paramPath));

			modelTrans_out.write("intercept\t" + ols_trans_intercept + "\n");

			for (int i_attr = 0; i_attr < col_num; i_attr++) {
				int id = model.nodeAttIdList.get(nodeIndex).get(i_attr);
				modelTrans_out.write(ainfo.idToName(id) + "\t");
				for(int i_poly = 0; i_poly < opts.polyDegree; i_poly++){
					modelTrans_out.write(ols_trans_coef[i_attr * opts.polyDegree + i_poly] + "\t" );
				}
				modelTrans_out.write(attr_range.get(i_attr).get(0) + "\t" + attr_range.get(i_attr).get(1) + "\n");
			}
			modelTrans_out.flush();
			modelTrans_out.close();

			long end_train = System.currentTimeMillis();

			timeStamp("Finished training OLS on this node in " + (end_train - start_load) / 1000.0 + " (s).");
			timeStamp("Without loading data, the model training step takes " + (end_train - end_load) / 1000.0 + " (s).");
		}

		// Get constant for each leafConst;

		timeStamp("-------------- Get constant for each leafConst ---------------");

		for(int i_const = 0; i_const < leavesConst.size(); i_const++){
			String leafName = leavesConst.get(i_const);

			String dataNodePath = model.dir + "/Node_" + leafName;
			System.out.println("------------Processing leaf " + leafName + "------------");

			// read dta file, only need to read the target column

			timeStamp("Load data");

			BufferedReader br_dta = new BufferedReader(new FileReader(dataNodePath + "/fir.dta"));
			ArrayList<Double> y_double_arraylist = new ArrayList<Double>();
			double y_sum = 0;
			for(String line = br_dta.readLine(); line != null; line = br_dta.readLine()){
				String[] data = line.split("\t+");
				double y_current = Double.parseDouble(data[ainfo.getClsCol()]);
				y_double_arraylist.add(y_current);
				y_sum += y_current;
			}
			br_dta.close();

			double y_mean = y_sum / y_double_arraylist.size();
			System.out.println(y_mean);

			timeStamp("Saving the const");

			String paramPath = model.getParamPath(leafName);
			BufferedWriter const_out = new BufferedWriter(new FileWriter(paramPath));
			const_out.write("Const: " + y_mean);
			const_out.flush();
			const_out.close();
		}
		
		long end = System.currentTimeMillis();
		timeStamp("Finished all in " + (end - start) / 1000.0 + " (s).");
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
	
	static boolean checkCrash(int nAtt, AttrInfo ainfo, Options opts, String dataPath) 
			throws Exception {
		/*// Crash that needs to look into org.netlib.lapack.Dgeqrf.dgeqrf(lapack.f)
		int max = Integer.MAX_VALUE / (nAtt * opts.polyDegree + 1);
		*///
		// Use this magic number instead of using max integer value 2147483647
		int max = 2147000000 / (nAtt * opts.polyDegree + 1);
		
		BufferedReader br = new BufferedReader(new FileReader(dataPath));
		int num = 0;
		while (br.readLine() != null) num ++;
		br.close();
		
		if (num > max)
			return true;
		else
			return false;
	}

	static Set<String> subsample(int nAtt, AttrInfo ainfo, Options opts, String dataPath) 
			throws Exception {
		Set<String> groupIdSet = new HashSet<String>();
		
		/*// Crash that needs to look into org.netlib.lapack.Dgeqrf.dgeqrf(lapack.f)
		int max = Integer.MAX_VALUE / (nAtt * opts.polyDegree + 1);
		*///
		// Use this magic number instead of using max integer value 2147483647
		int max = 2147000000 / (nAtt * opts.polyDegree + 1);
		
		Map<String, Integer> groupSizes = new HashMap<String, Integer>();
		BufferedReader br = new BufferedReader(new FileReader(dataPath));
		for (String line = br.readLine(); line != null; line = br.readLine()) {
			String[] data = line.split("\t");
			String groupId = data[ainfo.nameToCol.get(opts.group)];
			groupSizes.put(groupId, groupSizes.getOrDefault(groupId, 0) + 1);
		}
		br.close();
		List<String> groupIdList = new ArrayList<String>(groupSizes.keySet());
		Collections.shuffle(groupIdList);
		int num = 0;
		for (String groupId : groupIdList) {
			if (num + groupSizes.get(groupId) > max)
				break;
			num += groupSizes.get(groupId);
			groupIdSet.add(groupId);
		}
		timeStamp(String.format(
				"Subsample a set of %d out of %d group ids", 
				groupIdSet.size(), 
				groupIdList.size()
				));
		return groupIdSet;
	}

	protected static void timeStamp(String msg){
		Date tmpDate = new Date();
		System.out.println("TIMESTAMP >>>> ".concat(tmpDate.toString()).concat(": ").concat(msg));
	}
}
