package lv.edi.EDI_3DAtA.bloodvesselsegm;

import java.util.ArrayList;

import boofcv.alg.filter.convolve.GConvolveImageOps;
import boofcv.factory.filter.kernel.FactoryKernelGaussian;
import boofcv.struct.convolve.Kernel2D_F64;
import boofcv.struct.convolve.Kernel2D_I32;
import boofcv.struct.image.ImageUInt8;

/**
 * Class representing Gaussian pyramid of image. Each layer of pyramid is two times smaller
 * @author Ričards Cacurs
 *
 */
public class GaussianPyramid {
	private ArrayList<ImageUInt8> layers;
	
	/** Constructor that creates Gaussian pyramid with specified parameters.
	 * 
	 * @param src input image, gaussian pyramid layer 0
	 * @param numberOfLayers  number of layers for pyramid
	 * @param kernelWidth size for the Gaussian filter kernel in pixels
	 * @param kernelSigma sigma parameter for Gaussian filter
	 */
	public GaussianPyramid(ImageUInt8 src, int numberOfLayers, int kernelWidth, float kernelSigma){
		layers = new ArrayList<ImageUInt8>(numberOfLayers);
		Kernel2D_I32 kernel = FactoryKernelGaussian.gaussian2D(ImageUInt8.class, 1.0, 5);
		System.out.println("Kernel width"+kernel.getWidth());
		System.out.println(kernel.toString());
		layers.add(src);
		for(int i=1; i<numberOfLayers; i++){
			ImageUInt8 output = new ImageUInt8(layers.get(i-1).width, layers.get(i-1).height);
			GConvolveImageOps.convolveNormalized(kernel, layers.get(i-1), output);
			ImageUInt8 outputDownsample = new ImageUInt8(layers.get(i-1).width/2, layers.get(i-1).height/2);
			for(int j=0; j<outputDownsample.height; j++){
				for(int k=0; k<outputDownsample.width; k++){
					outputDownsample.set(k, j, output.get(k*2, j*2));
				}
			}
			layers.add(outputDownsample);
		}
	}
	
	/**
	 * Function for accessing pyramid layer
	 * @param layer layer index
	 * @return ImageUInt8 returns specific layer. Returns null if index 
	 * larger that number of pyramid layers
	 */
	public ImageUInt8 getLayer(int layer){
		if(layer<layers.size() && layer>=0){
			return layers.get(layer);
		} else{
			return null;
		}
	}
	
	/**
	 *  Returns number of layers in Gaussian pyramid
	 * @return Layer count in Gaussian pyramid
	 */
	public int size(){
		return layers.size();
	}
}
