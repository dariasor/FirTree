package firtree;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.List;

/**
 * Class for interaction tree nodes.
 * 
 * @author ylou
 *
 */
public abstract class InteractionTreeNode {

	/**
	 * Returns <code>true</code> if the node is a leaf.
	 * 
	 * @return <code>true</code> if the node is a leaf.
	 */
	public abstract boolean isLeaf();
		
}
