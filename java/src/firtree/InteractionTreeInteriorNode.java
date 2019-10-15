package firtree;




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
	
	public InteractionTreeInteriorNode(int attIndex, double splitPoint) {
		this.attIndex = attIndex;
		this.splitPoint = splitPoint;
	}
}
