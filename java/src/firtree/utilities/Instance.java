package firtree.utilities;

import java.util.Map;

import mltk.core.io.AttrInfo;

/**
 * Extends {@link mltk.core.Instance} to be used in the coordinate ascent algorithm
 * 
 * @author Xiaojie Wang
 */
public class Instance extends mltk.core.Instance {

	Map<String, Integer> nameToId;
	// Cache the previous prediction for an instance to speed up training
	protected double prediction;
	// Instances with the same group id are collectively called a query group
	protected String groupId;
	// Index in the list of all nodes in a FirTree model, but must be a leaf node's index
	protected int nodeIndex = -1;
	
	// Used for evaluation
	public Instance(double target) {
		super(target);
	}
	
	public Instance(mltk.core.Instance instance, Map<String, Integer> nameToId) {
		super(instance);
		this.nameToId = nameToId;
	}
	
	public int getAttId(String attName) {
		return nameToId.get(attName);
	}
	
	public boolean isIndexed() {
		if (nodeIndex == -1)
			return false;
		return true;
	}

	// The following getters and setters are automatically generated by Eclipse
	public double getPrediction() {
		return prediction;
	}

	public void setPrediction(double prediction) {
		this.prediction = prediction;
	}

	public int getNodeIndex() {
		return nodeIndex;
	}

	public void setNodeIndex(int nodeIndex) {
		this.nodeIndex = nodeIndex;
	}

	public String getGroupId() {
		return groupId;
	}

	public void setGroupId(String groupId) {
		this.groupId = groupId;
	}
}