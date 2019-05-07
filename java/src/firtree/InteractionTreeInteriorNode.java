package firtree;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.List;

import mltk.core.Instance;


/**
 * Class for interaction tree interior nodes.
 * 
 * @author ylou
 *
 */
public class InteractionTreeInteriorNode extends InteractionTreeNode {
	
	int attIndex;
	double splitPoint;

	@Override
	public boolean isLeaf() {
		return false;
	}
	
}
