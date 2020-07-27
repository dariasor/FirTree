package firtree.data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;

public class S3DataPreprocessor {
	
	static class Options {
		@Argument(name = "-i", description = "S3 data dir", required = true)
		String dir = null;
		
		@Argument(name = "-r", description = "Attribute file", required = true)
		String attPath = "";
		
		@Argument(name = "-t", description = "Data file", required = true)
		String trainPath = "";
	}
	
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(S3DataPreprocessor.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		
		Map<String, Integer> attToCol = new HashMap<String, Integer>();
		File dir = new File(opts.dir);
		for (String name : dir.list()) {
			if (! name.endsWith(".csv"))
				continue;

			String file = Paths.get(opts.dir, name).toString();
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line = br.readLine();
			String[] data = line.strip().split("\t");
			if (attToCol.size() == 0) {
				timeStamp(String.format("Load header from %s", name));
				int col = 0;
				for (String att : data) {
					attToCol.put(att, col);
					col += 1;
				}
			} else {
				int col = 0;
				for (String att : data) {
					if (attToCol.get(att) != col) {
						System.err.printf("Header of %s is not consistent\n", name);
						System.exit(1);
					}
					col += 1;
				}
			}
			br.close();
		}
		
		// A list of columns corresponding to selected attributes
		List<Integer> selCols = new ArrayList<Integer>();
		for (String att : attToCol.keySet())
			selCols.add(attToCol.get(att));
		timeStamp(String.format("Select %d out of %d attributes", selCols.size(), attToCol.size()));
		
		int total = 0;
		BufferedWriter bwT = new BufferedWriter(new FileWriter(opts.trainPath));
		for (String name : dir.list()) {
			if (! name.endsWith(".csv"))
				continue;
			
			String file = Paths.get(opts.dir, name).toString();
			BufferedReader br = new BufferedReader(new FileReader(file));
			br.readLine(); // Skip header
			List<String> lines = new ArrayList<String>();
			for (String line = br.readLine(); line != null; line = br.readLine()) {
				String[] data = line.strip().split("\t");
				List<String> selData = new ArrayList<String>();
				for (Integer col : selCols)
					selData.add(data[col]);
				lines.add(String.join("\t", selData));
			}
			bwT.write(String.join("\n", lines) + "\n");
			br.close();
			total += lines.size();
			timeStamp(String.format("Find %d instances in %s", lines.size(), name));
		}	
		bwT.close();
		timeStamp(String.format("Find %d instances in total", total));
		
		BufferedWriter bwR = new BufferedWriter(new FileWriter(opts.attPath));
		Map<Integer, String> colToAtt = new HashMap<Integer, String>();
		for (String att : attToCol.keySet())
			colToAtt.put(attToCol.get(att), att);
		for (Integer col : selCols)
			bwR.write(String.format("%s\n", colToAtt.get(col)));
		bwR.close();
	}

	static void timeStamp(String msg) {
		Date date = new Date();
		System.out.println("TIMESTAMP >>>> ".concat(date.toString()).concat(": ").concat(msg));
	}
}
