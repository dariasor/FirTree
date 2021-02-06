package firtree;

import java.io.File;
import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.Attr;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

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
			String expression = getExpression(nodes[0], opts);
			String encoded = URLEncoder.encode(expression, StandardCharsets.UTF_8.toString());
        	System.out.printf("Query time rank expression            : %s\n", expression);
        	System.out.printf("URL-encoded query time rank expression: %s\n", encoded);
		}
		
		if (nodes.length != 3) {
			System.err.println("SaveXQM only supports a FirTree with depth 2 and 3 nodes");
			System.exit(1);
		}
				
        DocumentBuilderFactory documentFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder documentBuilder = documentFactory.newDocumentBuilder();

        Document doc = documentBuilder.newDocument();
        doc.setXmlVersion("1.0");
        doc.setXmlStandalone(false);

        // <query-metadata> element
        Element queryMetadata = doc.createElement("query-metadata");
        queryMetadata.setAttribute("version", "2010-03-01");
        doc.appendChild(queryMetadata);
        
        // <virtual-fields> element
        Element virtualFields = doc.createElement("virtual-fields");
        queryMetadata.appendChild(virtualFields);
        
        String indexes = "[query_index]";
        
        // <virtual-field> element for the root of a FirTree
        Element root = doc.createElement("virtual-field");
        root.setAttribute("function", "range");
        root.setAttribute("query-name", "relevance_fast");
        root.setAttribute("type", "computed");
        root.setAttribute("indexes", indexes);
        virtualFields.appendChild(root);
        //// <comment>
        Element rootComment = doc.createElement("comment");
        rootComment.appendChild(doc.createTextNode("Placeholder"));
        root.appendChild(rootComment);
        //// <field>
        Element rootField = doc.createElement("field");
        rootField.appendChild(doc.createTextNode(getSplitField(nodes[0])));
        root.appendChild(rootField);
        //// <range-1> and <range-2>
        Integer[] lowerBounds = new Integer[] {-1, 0, getSplitValue(nodes[0]) + 1};
        String prefix = "firtree_[marketplace]_[query_index]_[date]";
        String[] queryNames = new String[] {null, prefix + "_node_l", prefix + "node_r",};
        for (int i = 1; i < 3; i ++) {        	
        	Element rootRange = doc.createElement(String.format("range-%d", i));
        	rootRange.appendChild(doc.createTextNode(String.format("%d:%s", lowerBounds[i], queryNames[i])));
        	root.appendChild(rootRange);
        }
        
        // <virtual-field> elements for left node and right node of the root
        for (int i = 1; i < 3; i ++) {
        	Element node = doc.createElement("virtual-field");
        	node.setAttribute("function", "expression");
        	node.setAttribute("query-name", queryNames[i]);
        	node.setAttribute("type", "computed");
        	node.setAttribute("indexes", indexes);
        	//// <query-field>
        	for (String coreField : getCoreFields(nodes[i])) {
        		Element nodeQueryField = doc.createElement("query-field");
        		nodeQueryField.appendChild(doc.createTextNode(coreField));
        		node.appendChild(nodeQueryField);
        	}
        	//// <expression>
        	Element nodeExpression = doc.createElement("expression");
        	String expression = getExpression(nodes[i], opts);
        	System.out.printf("Expression of %s: %s\n", queryNames[i], expression);
        	nodeExpression.appendChild(doc.createTextNode(expression));
        	node.appendChild(nodeExpression);
        	virtualFields.appendChild(node);
        }
        
        //transform the DOM Object to an XML File
        TransformerFactory transformerFactory = TransformerFactory.newInstance();
        Transformer transformer = transformerFactory.newTransformer();
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");
        transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "2");
        transformer.setOutputProperty(OutputKeys.METHOD, "html");
        
        // create the xml file
        DOMSource domSource = new DOMSource(doc);
        StreamResult streamResult = new StreamResult(new File(opts.outputPath));
        transformer.transform(domSource, streamResult);
        System.out.printf("Write expressions to %s\n", opts.outputPath);

        // Output to the standard output for debugging
//        transformer.transform(domSource, new StreamResult(System.out));
	}
	
	static String getExpression(String node, Options opts) throws Exception {
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
		
		return String.join("+", expressions);
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
			expression += String.format("%d>=%s", minValue, feature);
			expression += String.format(")?%s:", sixSignificant(minPrediction));
		}
		expression += "(";
		expression += String.format("%s>=%d", feature, maxValue);
		expression += String.format(")?%s:", sixSignificant(maxPrediction));
		
		expression += String.join("+", getExpressions(feature, coefficients, polyDegree));
		
		expression += ")";
		
		return expression;
	}
	
	static String[] getExpressions(String feature, double[] coefficients, int polyDegree) {
		String[] expressions = new String[polyDegree];
		for (int i = 0; i < polyDegree; i ++) {
			expressions[i] = String.format("%s*%s", feature, sixSignificant(Math.pow(coefficients[i], i + 1)));
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
