package lv.edi.EDI_3DAtA.common;

import java.util.ArrayList;

import boofcv.struct.image.ImageUInt8;

/**
 * Class for volumetric data representation
 * @author Ričards Cacurs
 *
 */
public class VolumetricData{
	
	private ArrayList<ImageUInt8> layers;
	
	/**
	 * Initializes Volumetric Data with initial capacity of layers 10
	 */
	public VolumetricData(){
		layers = new ArrayList<ImageUInt8>(10);
	}
	
	/**
	 * Initializes Volumetric Data with specific initial capacity
	 * @param initSize Number of initial layers for volumetric data
	 */
	public VolumetricData(int initSize){
		layers = new ArrayList<ImageUInt8>(initSize);
	}
	
	/**
	 * Ads layer to volumetric data.
	 * @param layer Layer data 2D matrix.
	 */
	public void addLayer(ImageUInt8 layer){
		layers.add(layer);
	}
	/**
	 * Method for getting one layer data
	 * @param index index of the layer
	 * @return ImageUInt8 one layer data
	 */
	public ImageUInt8 getLayer(int index){
		return layers.get(index);
	}
	
	/**
	 * Method return number of layers
	 * @return int number of layers
	 */
	public int size(){
		return layers.size();
	}
	
	
}
