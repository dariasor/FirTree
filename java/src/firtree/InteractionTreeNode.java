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
	
	public abstract void writeStructure(PrintWriter out) throws Exception;

	public abstract void readStructure(BufferedReader in) throws Exception;
	
	public abstract void writeModel(PrintWriter out) throws Exception;

	public abstract void readModel(BufferedReader in) throws Exception;
	
	public abstract void getLeaves(List<InteractionTreeLeaf> leaves);
	
	public abstract void dfs(List<InteractionTreeNode> nodes);
	
}
