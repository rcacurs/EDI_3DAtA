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
	
	
}
