package firtree;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class Feature {

	public String name;
	public double[] centers;
	public double[] counts;
	
	public static Feature read(String file) throws Exception {
		BufferedReader br = new BufferedReader(new FileReader(file), 65535);
		String[] data = br.readLine().split("\\s+");
		Feature feature = new Feature();
		feature.name = data[1].replaceAll("_values", "");
		
		br.readLine();
		
		List<Double> centers = new ArrayList<>();
		List<Double> counts = new ArrayList<>();
		
		for (;;) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			data = line.split("\\s+");
			counts.add(Double.parseDouble(data[0]));
			if(data[1].equals("?"))
				centers.add(Double.NaN);				
			else 
				centers.add(Double.parseDouble(data[1]));
		}
		br.close();
		
		feature.centers = new double[centers.size()];
		feature.counts = new double[centers.size()];
		for (int i = 0; i < feature.centers.length; i++) {
			feature.centers[i] = centers.get(i);
			feature.counts[i] = counts.get(i);
		}
		
		return feature;
	}
	
}
