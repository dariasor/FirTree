package firtree;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.io.AttrInfo;
import mltk.core.io.AttributesReader;

public class SaveCPP {

	static class Options {
		@Argument(name="-l", description="(cropped) treelog.txt which specifies a tree structure", required=true)
		String logPath = "";
		
		@Argument(name = "-r", description = "attribute file", required = true)
		String attPath = "";
		
		@Argument(name = "-y", description = "polynomial degree", required = true)
		int polyDegree = 2;

		@Argument(name = "-m", description = "Prefix of name of output parameter files (default: model)")
		String modelPrefix = "model";
		
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
		FirTree model = new FirTree(ainfo, opts.logPath, opts.polyDegree, opts.modelPrefix);
		model.outcpp(opts.outputPath);
	}
	
}
