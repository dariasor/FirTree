package firtree;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.io.AttrInfo;
import mltk.core.io.AttributesReader;

public class SplitDataIntoLeaves {
	
	static class Options {
		@Argument(name="-l", description="(cropped) treelog.txt which specifies a tree structure", required=true)
		String logPath = "";
		
		@Argument(name = "-r", description = "attribute file", required = true)
		String attPath = "";
		
		@Argument(name = "-t", description = "training set", required = true)
		String trainPath = "";
	}
	
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(SplitDataIntoLeaves.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		
		// Load attribute file
		AttrInfo ainfo = AttributesReader.read(opts.attPath);
		
		// Load tree structure and initial parameter values
		FirTree model = new FirTree(ainfo, opts.logPath, -1, "");
		
		System.out.printf("Read data from %s\n", opts.trainPath);
		Map<String, List<String>> leafToLines = new HashMap<>();
		BufferedReader br = new BufferedReader(new FileReader(opts.trainPath));
		for (String line = br.readLine(); line != null; line = br.readLine()) {
			String[] data = line.split("\t");
			
			// Predict index of the node (must be a leaf) the instance falls in
			int nodeIndex = model.indexLeaf(data);
			String leaf = model.getNodeName(nodeIndex);
			if (! leafToLines.containsKey(leaf)) {
				leafToLines.put(leaf, new ArrayList<>());
			}
			leafToLines.get(leaf).add(line);
		}
		br.close();
		
		for (String leaf : leafToLines.keySet()) {
			List<String> lines = leafToLines.get(leaf);
			File dir = new File(CoorAscentOnLeaves.getNodeDir(model.dir, leaf));
			if (! dir.exists()) {
				dir.mkdirs();
			}
			
			Path outPath = Paths.get(dir.getAbsolutePath(), "fir.dta");
			System.out.printf("Save %d instances into %s\n", lines.size(), outPath);
			Files.deleteIfExists(outPath);
			Files.write(outPath, lines, StandardCharsets.UTF_8, StandardOpenOption.CREATE);
		}
	}

}
