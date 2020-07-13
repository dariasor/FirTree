package firtree.utilities;

import java.util.ArrayList;
import java.util.List;

/**
 * A rank list of instances with the same group id
 * 
 * @author Xiaojie Wang
 */
public class RankList {

	String groupId;
	protected List<Instance> instances;
	protected double weight;
	
	// Cache the previous score of a rank list by a certain metric
	protected double score;

	public RankList(String groupId) {
		this.groupId = groupId;
		this.instances = new ArrayList<>();
	}

    public RankList(RankList rankList, int[] indexes) {
    		this.groupId = rankList.getGroupId();
    		this.instances = new ArrayList<>();
        for (int i = 0; i < indexes.length; i ++) {
            this.instances.add(rankList.get(indexes[i]));
        }
    }
    
	public Instance get(int index) {
		return instances.get(index);
	}
	
	public void add(Instance instance) {
		this.instances.add(instance);
	}
	
	public int size() {
		return this.instances.size();
	}
	
	public String getGroupId() {
		return groupId;
	}

	public void setGroupId(String groupId) {
		this.groupId = groupId;
	}

	public List<Instance> getInstances() {
		return instances;
	}

	public void setInstances(List<Instance> instances) {
		this.instances = instances;
	}

	
	public double getWeight() {
		return weight;
	}
	

	public void setWeight(double weight) {
		this.weight = weight;
	}
	

	public double getScore() {
		return score;
	}
	

	public void setScore(double score) {
		this.score = score;
	}
	
}
