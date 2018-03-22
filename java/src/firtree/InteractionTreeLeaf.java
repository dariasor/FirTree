package firtree;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.List;

import mltk.predictor.gam.GAM;


/**
 * Class for interaction tree leaves.
 * 
 * @author ylou
 *
 */
public class InteractionTreeLeaf extends InteractionTreeNode {
	
	protected GAM gam;

	@Override
	public boolean isLeaf() {
		return true;
	}

	@Override
	public void writeStructure(PrintWriter out) throws Exception {
		out.println("Leaf");
	}

	@Override
	public void readStructure(BufferedReader in) throws Exception {
		
	}

	@Override
	public void writeModel(PrintWriter out) throws Exception {
		out.println("Leaf");
		gam.write(out);
	}

	@Override
	public void readModel(BufferedReader in) throws Exception {
		gam = new GAM();
		in.readLine();
		gam.read(in);
	}

	@Override
	public void getLeaves(List<InteractionTreeLeaf> leaves) {
		leaves.add(this);
	}

	@Override
	public void dfs(List<InteractionTreeNode> nodes) {
		nodes.add(this);
	}

}
