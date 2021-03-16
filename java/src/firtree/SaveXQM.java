package firtree;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;

public class SaveXQM {
	
	static class Options {
		@Argument(name="-l", description="(cropped) treelog.txt which specifies a tree structure", required=true)
		String logPath = "";

		@Argument(name = "-m", description = "Prefix of name of output parameter files (default: model)", required=true)
		String modelPrefix = "model";

		@Argument(name = "-o", description = "output file with java code", required = true)
		String outputPath = "";

	}
	
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(SaveXQM.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		
		String log = Files.readString(Path.of(opts.logPath));
		String[] nodes = log.strip().split("\n\n");
		
		if (nodes.length == 1) {
			List<String> expressions = getExpressions(nodes[0], opts);
			String expression = String.join("+", expressions);
			String encoded = URLEncoder.encode(expression, StandardCharsets.UTF_8.toString());
        	System.out.printf("Query time rank expression            : %s\n", expression);
        	System.out.printf("URL-encoded query time rank expression: %s\n", encoded);
		}
		
		if (nodes.length != 3) {
			System.err.println("SaveXQM only supports a FirTree with depth 2 and 3 nodes");
			System.exit(1);
		}
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(opts.outputPath));
		writeLine(bw, "<virtual-fields>", 0);
		
		// Write root node
		writeLine(bw, "<virtual-field function=\"range\" query-name=\"relevance_fast\" type=\"computed\" indexes=\"[query_index]\">", 1);
		writeLine(bw, "<comment></comment>", 2);
		writeLine(bw, String.format("<field>%s</field>", getSplitField(nodes[0])), 2);
		writeLine(bw, String.format("<range-1>0:firtree_[marketplace]_[query_index]_[date]_node_l</range-1>"), 2);
		writeLine(bw, String.format("<range-2>%d:firtree_[marketplace]_[query_index]_[date]_node_r</range-2>", getSplitValue(nodes[0]) + 1), 2);
		writeLine(bw, "</virtual-field>", 1);
		
		// Write left leaf
		writeLine(bw, "<virtual-field function=\"expression\" query-name=\"firtree_[marketplace]_[query_index]_[date]_node_l\" type=\"computed\" indexes=\"[query_index]\">", 1);
		writeLeaf(bw, nodes[1], opts);
		writeLine(bw, "</virtual-field>", 1);		
		
		// Write right leaf
		writeLine(bw, "<virtual-field function=\"expression\" query-name=\"firtree_[marketplace]_[query_index]_[date]_node_r\" type=\"computed\" indexes=\"[query_index]\">", 1);
		writeLeaf(bw, nodes[2], opts);
    	writeLine(bw, "</virtual-field>", 1);
		
    	writeLine(bw, "</virtual-fields>", 0);
		bw.close();
		
		System.out.println("Output XQM to " + opts.outputPath);
	}
	
	static void writeLeaf(BufferedWriter bw, String node, Options opts) throws Exception {
    	for (String coreField : getCoreFields(node)) {
    		writeLine(bw, String.format("<query-field>%s</query-field>", coreField), 2);
    	}
    	writeLine(bw, "<expression>", 2);
    	List<String> expressions = getExpressions(node, opts);
    	for (int i = 0; i < expressions.size(); i ++) {
    		if (i == 0) {
    	    	writeLine(bw, expressions.get(i), 3);
    		} else {
    			writeLine(bw, "+" + expressions.get(i), 3);
    		}
    	}
    	writeLine(bw, "</expression>", 2);
	}
	
	static void writeLine(BufferedWriter bw, String line, int level) throws Exception {
		for (int i = 0; i < level; i ++) {
			bw.write("  ");
		}
		bw.write(line + "\n");
	}
	
	static List<String> getExpressions(String node, Options opts) throws Exception {
		List<String> expressions = new ArrayList<>();
		
		String nodeName = "Node_" + node.strip().split("\n")[0].strip();
		Path modelPath = Paths.get((new File(opts.logPath)).getParent(), nodeName, opts.modelPrefix + ".txt");
		String model = Files.readString(modelPath);
		for (String line : model.strip().split("\n")) {
			String[] splits = line.split("\t");
			
			// Ignore the intercept of a linear model
			if (splits.length == 2) {
				continue;
			}
			
			expressions.add(getExpression(splits));
		}
		
		return expressions;
	}
	
	static String getExpression(String[] splits) {
		String feature = splits[0];
		int polyDegree = splits.length - 1 - 4;
		
		double[] coefficients = new double[polyDegree];
		for (int i = 0; i < polyDegree; i ++) {
			coefficients[i] = Math.pow(10, 5) * Double.parseDouble(splits[i + 1]);
		}
		
		int minValue = (int) Double.parseDouble(splits[splits.length - 4]);
		double minPrediction = getPrediction(coefficients, minValue, polyDegree);

		int maxValue = (int) Double.parseDouble(splits[splits.length - 3]);
		double maxPrediction = getPrediction(coefficients, maxValue, polyDegree);
		
		String expression = "(";
		
		if (minValue > 0) {
			expression += "(";
			expression += String.format("%d >= %s", minValue, feature);
			expression += String.format(") ? %s : ", sixSignificant(minPrediction));
		}
		expression += "(";
		expression += String.format("%s >= %d", feature, maxValue);
		expression += String.format(") ? %s : ", sixSignificant(maxPrediction));
		
		expression += String.join("+", getExpressions(feature, coefficients, polyDegree));
		
		expression += ")";
		
		return expression;
	}
	
	static String[] getExpressions(String feature, double[] coefficients, int polyDegree) {
		String[] expressions = new String[polyDegree];
		for (int i = 0; i < polyDegree; i ++) {
			expressions[i] = String.format("%s * %s", feature, sixSignificant(Math.pow(coefficients[i], i + 1)));
		}
		return expressions;
	}
	
	static double getPrediction(double[] coefficients, int value, int polyDegree) {
		double prediction = 0.;
		for (int i = 0; i < polyDegree; i ++) {
			prediction += coefficients[i] * Math.pow(value, i + 1);
		}
		return prediction;
	}
	
	static List<String> getCoreFields(String node) {
		List<String> coreFields = new ArrayList<>();
		for (String line : node.strip().split("\n")) {
			if (line.startsWith("\t")) {
				coreFields.add(line.strip());
			}
		}
		return coreFields;
	}
	
	static String getSplitField(String node) {
		String field = null;
		for (String line : node.strip().split("\n")) {
			if (line.startsWith("Best feature:")) {
				field = line.strip().split(": ")[1];
			}
		}
		if (field == null) {
			System.err.printf("Cannot get split field from '''\n%s\n'''\n", node);
			System.exit(1);
		}
		return field;
	}
	
	static int getSplitValue(String node) {
		int value = -1;
		for (String line : node.strip().split("\n")) {
			if (line.startsWith("Best split:")) {
				value = (int) Double.parseDouble(line.strip().split(": ")[1]);
			}
		}
		if (value == -1) {
			System.err.printf("Cannot get split value from '''\n%s\n'''\n", node);
			System.exit(1);
		}
		return value;
	}	

	static String sixSignificant(double num) {
		String str = String.format("%E", num);
		int power = Integer.parseInt(str.substring(str.indexOf("E") + 1));
		// 6 -> 0, 5 -> 0, 4 -> 1, ..., -3 -> 8
		int digit = Math.max(5 - power, 0);
		return String.format("%." + digit + "f", num);
	}
}
