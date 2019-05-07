package mltk.core.io;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import mltk.core.Attribute;

/**
 * Structure for information on attributes in the data: 
 * list of active attributes, class attribute,  
 * map from names to columns in the data file for all attributes. Columns start with zero
 * map from names to ids for active attributes
 * 
 * @author Daria Sorokina
 * 
 */

public class AttrInfo {
	public List<Attribute> attributes;
	public Attribute clsAttr;
	public HashMap<String, Integer> nameToCol;	//defined for all attributes
	public HashMap<String, Integer> nameToId;	//defined for active attributes only (inactive don't have id)
	
	/**
	 * Default constructor 
	 */
	public AttrInfo() {
		this.attributes = new ArrayList<Attribute>();
		this.nameToCol = new HashMap<String, Integer>();
		this.nameToId = new HashMap<String, Integer>();
	}
	
	public int getClsCol() {
		return clsAttr.getColumn();
	}
	
	public int idToCol(int id) {
		return attributes.get(id).getColumn();
	}
	
	public String idToName(int id) {
		return attributes.get(id).getName();
	}
}