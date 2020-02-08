package mltk.core;

/**
 * @author Xiaojie Wang
 * 
 */
public class Pointer {

	protected int index;
	protected int weight;
	
	public Pointer(int index, int weight) {
		this.index = index;
		this.weight = weight;
	}
	
	public Pointer(int index) {
		this(index, 1);
	}
	
	public int getIndex() {
		return index;
	}
	
	public int getWeight() {
		return weight;
	}
	
}
