package firtree;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.io.AttrInfo;
import mltk.core.io.AttributesReader;

public class SaveCPP {

	static class Options {
		@Argument(name = "-d", description = "model directory", required = true)
		String dir = ""; //usually path up to "FirTree" inclusive
		
		@Argument(name = "-r", description = "attribute file", required = true)
		String attPath = "";
		
		@Argument(name = "-y", description = "polynomial degree")
		int poly_degree = 2;

		@Argument(name = "-o", description = "output file with c++ code", required = true)
		String outputPath = ""; 		

	}
	
	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(SaveCPP.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}
		
		AttrInfo ainfo = AttributesReader.read(opts.attPath);
		FirTree model = new FirTree(ainfo, opts.dir, opts.poly_degree);
		model.outcpp(opts.outputPath);
	}
	
}
