package firtree;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.List;

import mltk.predictor.gam.GAM;


/**
 * Class for interaction tree leaves.
 * 
 * @author ylou
 *
 */
public class InteractionTreeLeaf extends InteractionTreeNode {
	
	@Override
	public boolean isLeaf() {
		return true;
	}

}
