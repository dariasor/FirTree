package firtree;

import mltk.cmdline.Argument;
import mltk.cmdline.CmdLineParser;
import mltk.core.io.AttrInfo;
import mltk.core.io.AttributesReader;

public class SaveJava {

	static class Options {
		@Argument(name = "-d", description = "model directory", required = true)
		String dir = ""; //usually path up to "FirTree" inclusive

		@Argument(name = "-r", description = "attribute file", required = true)
		String attPath = "";

		@Argument(name = "-y", description = "polynomial degree")
		int polyDegree = 2;

		@Argument(name = "-o", description = "output file with java code", required = true)
		String outputPath = "";

		@Argument(name = "-m", description = "Prefix of name of output parameter files (default: model)")
		String modelPrefix = "model";
	}

	public static void main(String[] args) throws Exception {
		Options opts = new Options();
		CmdLineParser parser = new CmdLineParser(SaveJava.class, opts);
		try {
			parser.parse(args);
		} catch (IllegalArgumentException e) {
			parser.printUsage();
			System.exit(1);
		}

		AttrInfo ainfo = AttributesReader.read(opts.attPath);
		FirTree model = new FirTree(ainfo, opts.dir, opts.polyDegree, opts.modelPrefix);
		model.outjava(opts.outputPath);
	}
}
