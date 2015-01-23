package lv.edi.EDI_3DAtA.common;

import java.util.ArrayList;

import boofcv.struct.image.ImageFloat32;
import boofcv.struct.image.ImageUInt8;

/**
 * Class for volumetric data representation
 * @author Ričards Cacurs
 *
 */
public class VolumetricData{
	
	private ArrayList<ImageFloat32> layers;
	
	/**
	 * Initializes Volumetric Data with initial capacity of layers 10
	 */
	public VolumetricData(){
		layers = new ArrayList<ImageFloat32>(10);
	}
	
	/**
	 * Initializes Volumetric Data with specific initial capacity
	 * @param initSize Number of initial layers for volumetric data
	 */
	public VolumetricData(int initSize){
		layers = new ArrayList<ImageFloat32>(initSize);
	}
	
	/**
	 * Ads layer to volumetric data.
	 * @param layer Layer data 2D matrix.
	 */
	public void addLayer(ImageFloat32 layer){
		layers.add(layer);
	}
	/**
	 * Method for getting one layer data
	 * @param index index of the layer
	 * @return ImageFloat32 one layer data
	 */
	public ImageFloat32 getLayer(int index){
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
			ImageFloat32 layerI = layers.get(layer);
			for(int i=0; i<layerI.height; i++){
				for(int j=0; j<layerI.width; j++){
					sb.append(layerI.get(j, i));
					if(j<(layerI.width-1)){
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
	
	
}
