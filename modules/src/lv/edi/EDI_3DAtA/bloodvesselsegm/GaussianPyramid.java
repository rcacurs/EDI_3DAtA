package lv.edi.EDI_3DAtA.bloodvesselsegm;

import java.util.ArrayList;
import org.ejml.data.DenseMatrix64F;


/**
 * Class representing Gaussian pyramid of image. Each layer of pyramid is two times smaller.
 * @author Ricards Cacurs
 *
 */
public class GaussianPyramid {
	private ArrayList<DenseMatrix64F> layers;
	
	/** Constructor that creates Gaussian pyramid with specified parameters.
	 * 
	 * @param src input image, gaussian pyramid layer 0 specified in DenseMatrix64F
	 * @param numberOfLayers  number of layers for pyramid
	 * @param kernelWidth size for the Gaussian filter kernel in pixels
	 * @param kernelSigmaSQ sigma parameter for Gaussian filter (STD)
	 */
	public GaussianPyramid(DenseMatrix64F src, int numberOfLayers, int kernelWidth, float kernelSigmaSQ){
		layers = new ArrayList<DenseMatrix64F>(numberOfLayers);
		DenseMatrix64F output = FilteringOperations.gaussianBlur(src,1,5);
		layers.add(output);
		for(int i=1; i<numberOfLayers; i++){
			output = FilteringOperations.gaussianBlur(layers.get(i-1), kernelSigmaSQ, kernelWidth);;
			DenseMatrix64F outputDownsample = new DenseMatrix64F(layers.get(i-1).numRows/2, layers.get(i-1).numCols/2);
			for(int j=0; j<outputDownsample.numRows; j++){
				for(int k=0; k<outputDownsample.numCols; k++){
					outputDownsample.set(j, k, output.get(j*2, k*2));
				}
			}
			layers.add(outputDownsample);
		}
	}
	
	/**
	 * Function for accessing pyramid layer
	 * @param layer layer index
	 * @return DenseMatrix64F returns specific layer. Returns null if index 
	 * larger that number of pyramid layers
	 */
	public DenseMatrix64F getLayer(int layer){
		if(layer<layers.size() && layer>=0){
			return layers.get(layer);
		} else{
			return null;
		}
	}
	
	/**
	 *  Returns number of layers in Gaussian pyramid
	 * @return int Layer count in Gaussian pyramid
	 */
	public int size(){
		return layers.size();
	}
}
