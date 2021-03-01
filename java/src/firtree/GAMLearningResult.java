package firtree;

public class GAMLearningResult {
	public boolean isParent;
	
	// The variable is set when learning a parent GAM
	public double parentScore;
	
	// These variables are set when learning two child GAMs
	public int attIndex;
	public double splitPoint;
	public double splitScore;
	
	public GAMLearningResult(boolean isParent) {
		this.isParent = isParent;
	}
}
