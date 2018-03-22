package firtree;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;

import mltk.core.Instance;
import mltk.util.Queue;

/**
 * Class for piecewise linear (additive) tree models.
 * 
 * @author ylou
 *
 */
public class InteractionTree {

	protected InteractionTreeNode root;
	
	public void readStructure(BufferedReader in) throws Exception {
		String line = in.readLine();
		if (line.equalsIgnoreCase("Leaf")) {
			root = new InteractionTreeLeaf();
		} else {
			root = new InteractionTreeInteriorNode();
		}
		root.readStructure(in);
	}
	
	public void readModel(BufferedReader in) throws Exception {
		String line = in.readLine();
		if (line.equalsIgnoreCase("Leaf")) {
			root = new InteractionTreeLeaf();
		} else {
			root = new InteractionTreeInteriorNode();
		}
		root.readModel(in);
	}
	
	public void writeStructure(PrintWriter out) throws Exception {
		root.writeStructure(out);
	}
	
	public void writeModel(PrintWriter out) throws Exception {
		root.writeModel(out);
	}
	
	public InteractionTreeLeaf getLeaf(Instance instance) {
		if (root.isLeaf()) {
			return (InteractionTreeLeaf) root;
		} else {
			return ((InteractionTreeInteriorNode) root).getLeaf(instance);
		}
	}
	
	public void getLeaves(List<InteractionTreeLeaf> leaves) {
		root.getLeaves(leaves);
	}
	
	public void computePaths(Map<InteractionTreeNode, String> paths) {
		paths.put(root, "Root");
		Queue<InteractionTreeNode> q = new Queue<>();
		q.enqueue(root);
		
		while (!q.isEmpty()) {
			InteractionTreeNode node = q.dequeue();
			String path = paths.get(node);
			if (!node.isLeaf()) {
				InteractionTreeInteriorNode interiorNode = 
						(InteractionTreeInteriorNode) node;
				paths.put(interiorNode.left, path + "_L");
				paths.put(interiorNode.right, path + "_R");
				
				q.enqueue(interiorNode.left);
				q.enqueue(interiorNode.right);
			}
		}
	}
	
	public void dfs(List<InteractionTreeNode> nodes) {
		root.dfs(nodes);
	}
	
}
