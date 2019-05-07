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
		timeStamp("Load model");
		AttrInfo ainfo = AttributesReader.read(opts.attPath);
		
		FirTree model = new FirTree(ainfo, opts.dir, opts.poly_degree);

		// Load test data and predict
		timeStamp("Load data and generate predictions");
		BufferedReader testData = new BufferedReader(new FileReader(opts.testPath), 65535);
		BufferedWriter pred_fir_out = new BufferedWriter(new FileWriter(opts.outputPath));

		String line_test = testData.readLine();
		while(line_test != null) {
			double val = model.predict(line_test); 
			pred_fir_out.write(val + "\n");
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
