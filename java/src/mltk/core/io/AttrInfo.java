package mltk.core.io;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import mltk.core.Attribute;

/**
 * Structure for information on attributes in the data: 
 * list of active attributes, class attribute,  
 * map from names to columns in the data file for all attributes. Columns start with zero
 * map from names to ids for active attributes
 * 
 * @author Daria Sorokina, modified by Xiaojie Wang
 * 
 */

public class AttrInfo {
	public List<Attribute> attributes;
	public Attribute clsAttr;
	public Attribute wtAttr;
	public HashMap<String, Integer> nameToCol;	//defined for all attributes
	public HashMap<String, Integer> nameToId;	//defined for active attributes only (inactive don't have id)
	public Set<String> leafNames;
	public Set<String> splitNames;
	
	public static int GROUP_UNSET = -1;
	public int groupCol = GROUP_UNSET;
	
	/**
	 * Default constructor 
	 */
	public AttrInfo() {
		this.attributes = new ArrayList<Attribute>();
		this.nameToCol = new HashMap<String, Integer>();
		this.nameToId = new HashMap<String, Integer>();
		this.leafNames = new HashSet<String>();
		this.splitNames = new HashSet<String>();
	}
	
	public int getClsCol() {
		return clsAttr.getColumn();
	}
	
	public int getWtCol() {
		if (wtAttr == null) {
			System.out.println("Warning: weight column is not well defined");
			return -1;
		}
		return wtAttr.getColumn();
	}

	public int idToCol(int id) {
		return attributes.get(id).getColumn();
	}
	
	public String idToName(int id) {
		return attributes.get(id).getName();
	}
	
	public int getColN() {
		return nameToCol.size();
	}
}
