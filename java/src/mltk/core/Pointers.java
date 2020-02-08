package mltk.core;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author Xiaojie Wang
 * 
 */

public class Pointers implements Iterable<Pointer> {

	List<Pointer> pointers;
	
	public Pointers() {
		pointers = new ArrayList<>();
	}
	
	public void add(Pointer pointer) {
		pointers.add(pointer);
	}

	@Override
	public Iterator<Pointer> iterator() {
		return pointers.iterator();
	}
	
	public final int size() {
		return pointers.size();
	}
	
	public Pointer get(int index) {
		return pointers.get(index);
	}
	
}
