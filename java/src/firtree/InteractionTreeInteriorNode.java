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
	
	protected InteractionTreeNode left;
	protected InteractionTreeNode right;
	
	int attIndex;
	double splitPoint;

	@Override
	public boolean isLeaf() {
		return false;
	}
	
	InteractionTreeInteriorNode() {
		
	}
	
	public InteractionTreeLeaf getLeaf(Instance instance) {
		double v = instance.getValue(attIndex);
		if (v <= splitPoint) {
			if (!left.isLeaf()) {
				return ((InteractionTreeInteriorNode) left).getLeaf(instance);
			} else {
				return (InteractionTreeLeaf) left;
			}
		} else {
			if (!right.isLeaf()) {
				return ((InteractionTreeInteriorNode) right).getLeaf(instance);
			} else {
				return (InteractionTreeLeaf) right;
			}
		}
	}
	
	public InteractionTreeInteriorNode(int attIndex, double splitPoint) {
		this.attIndex = attIndex;
		this.splitPoint = splitPoint;
	}
	
	public InteractionTreeNode getLeftChild() {
		return left;
	}
	
	public InteractionTreeNode getRightChild() {
		return right;
	}
	
	public int getSplitAttributeIndex() {
		return attIndex;
	}
	
	public double getSplitPoint() {
		return splitPoint;
	}

	@Override
	public void writeStructure(PrintWriter out) throws Exception {
		out.println("InteriorNode");
		out.println(attIndex);
		out.println(splitPoint);
		out.println();
		left.writeStructure(out);
		out.println();
		right.writeStructure(out);
	}

	@Override
	public void readStructure(BufferedReader in) throws Exception {
		attIndex = Integer.parseInt(in.readLine());
		splitPoint = Double.parseDouble(in.readLine());
		in.readLine();
		String line = in.readLine();
		if (line.equalsIgnoreCase("Leaf")) {
			left = new InteractionTreeLeaf();
			left.readStructure(in);
		} else {
			left = new InteractionTreeInteriorNode();
			left.readStructure(in);
		}
		in.readLine();
		line = in.readLine();
		if (line.equalsIgnoreCase("Leaf")) {
			right = new InteractionTreeLeaf();
			right.readStructure(in);
		} else {
			right = new InteractionTreeInteriorNode();
			right.readStructure(in);
		}
	}

	@Override
	public void writeModel(PrintWriter out) throws Exception {
		out.println("InteriorNode");
		out.println(attIndex);
		out.println(splitPoint);
		out.println();
		left.writeModel(out);
		out.println();
		right.writeModel(out);
	}

	@Override
	public void readModel(BufferedReader in) throws Exception {
		attIndex = Integer.parseInt(in.readLine());
		splitPoint = Double.parseDouble(in.readLine());
		in.readLine();
		String line = in.readLine();
		if (line.equalsIgnoreCase("Leaf")) {
			left = new InteractionTreeLeaf();
			left.readModel(in);
		} else {
			left = new InteractionTreeInteriorNode();
			left.readModel(in);
		}
		in.readLine();
		line = in.readLine();
		if (line.equalsIgnoreCase("Leaf")) {
			right = new InteractionTreeLeaf();
			right.readModel(in);
		} else {
			right = new InteractionTreeInteriorNode();
			right.readModel(in);
		}
	}

	@Override
	public void getLeaves(List<InteractionTreeLeaf> leaves) {
		left.getLeaves(leaves);
		right.getLeaves(leaves);
	}

	@Override
	public void dfs(List<InteractionTreeNode> nodes) {
		nodes.add(this);
		left.dfs(nodes);
		right.dfs(nodes);
	}

}
