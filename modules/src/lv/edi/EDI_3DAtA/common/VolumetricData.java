package lv.edi.EDI_3DAtA.common;

import java.util.ArrayList;

import org.ejml.data.DenseMatrix64F;


/**
 * Class for volumetric data representation
 * @author Ričards Cacurs
 *
 */
public class VolumetricData{
	
	private ArrayList<DenseMatrix64F> layers;
	
	/**
	 * Initializes Volumetric Data with initial capacity of layers 10
	 */
	public VolumetricData(){
		layers = new ArrayList<DenseMatrix64F>(10);
	}
	
	/**
	 * Initializes Volumetric Data with specific initial capacity
	 * @param initSize Number of initial layers for volumetric data
	 */
	public VolumetricData(int initSize){
		layers = new ArrayList<DenseMatrix64F>(initSize);
	}
	
	/**
	 * Ads layer to volumetric data.
	 * @param layer Layer data 2D matrix.
	 */
	public void addLayer(DenseMatrix64F layer){
		layers.add(layer);
	}
	/**
	 * Method for getting one layer data
	 * @param index index of the layer
	 * @return DenseMatrix64F one layer data
	 */
	public DenseMatrix64F getLayer(int index){
		return layers.get(index);
	}
	
	/**
	 * Method return number of layers
	 * @return int number of layers
	 */
	public int size(){
		return layers.size();
	}
	
	/**
	 * Method return string representation of one layer, formatted as .csv.
	 * @param layer integer representing layer index. If index number exceeds actual layer count empty string is returned.
	 * @return String string representation of specified layer in CSV format. 
	 */
	public String layerToString(int layer){
		if(layer>=0 && layer<layers.size()){
			StringBuilder sb = new StringBuilder();
			DenseMatrix64F layerI = layers.get(layer);
			for(int i=0; i<layerI.numRows; i++){
				for(int j=0; j<layerI.numCols; j++){
					sb.append(layerI.get(i, j));
					if(j<(layerI.numCols-1)){
						sb.append(",");
					}
				}
				sb.append("\n");
			}
			return sb.toString();
		}else{
			return "";
		}
	}
	
	/**
	 * Returns values of volumetric data at cube vertices with origin at specified coordinates
	 * Usefull in marching cubes algorithm. 
	 * @param x - x coordinate index
	 * @param y - y coodinate index
	 * @param z - z coordinate index
	 * @return double[] Values on cube vertices are returned as double[] array of size 8
	 *  order of values are corresponding:
	 *  double[] = {Value(i, j, k), Value(i+1, j, k), Value(i+1, j, k+1),
	 *              Value(i, j, k+1), Value(i, j+1, k), Value(i+1, j+1, k), 
	 *              Value(i+1, j+1, k+1), Value(i, j+1, k+1)}
	 */
	public double[] getValuesAtCubeVertices(int i, int j, int k){
		double[] values = new double[8];
		if((i>0)&&
		   (i<layers.get(0).numCols-1)&&
		   (j>0)&&
		   (j<layers.get(0).numRows-1)&&
		   (k>0)&&
		   (j<layers.size()-1)){
			
			values[0]=layers.get(k).get(j, i);
			values[1]=layers.get(k).get(j, i+1);
			values[2]=layers.get(k).get(j+1, i+1);
			values[3]=layers.get(k).get(j+1, i);
			values[4]=layers.get(k+1).get(j, i);
			values[5]=layers.get(k+1).get(j, i+1);
			values[6]=layers.get(k+1).get(j+1, i+1);
			values[7]=layers.get(k+1).get(j+1, i);
			
		}
		return values;
	}
	
	
}
